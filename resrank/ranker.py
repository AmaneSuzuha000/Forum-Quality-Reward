# -*- coding: utf-8 -*-
"""Ranker —— 把四步封成一个对象：fit / residual / score / allocate / save / load。

    r = Ranker(objective={...}, context=[...], content=[...], text=("title","text"))
    r.fit(df)
    r.residual(df)     # L1+L2 结算量（窗口已关闭）
    r.score(df)        # L1+L2+L3 预测分（窗口未关闭 / 冷启动）
    r.allocate(df)     # L4 档位 + 积分
    r.save("model.pkl")

交付物是**一个文件**：网络系数 + 标准化常数 + TF-IDF 词表 + 超参，
全部装在同一个 pickle 里。换台机器只要 pip install 那四行。
"""
from __future__ import annotations

import pickle
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .allocate import allocate
from .distill import Distiller, build_docs
from .signal import Debiaser, compose


class Ranker:
    def __init__(self,
                 objective: dict[str, float],
                 context: Sequence[str] = (),
                 content: Sequence[str] = (),
                 text: tuple[str, ...] | None = None,
                 time_col: str | None = None,
                 train_end: str | None = None,
                 alpha_debias: float = 5.0,
                 alpha_distill: float = 30.0,
                 max_features: int = 8000,
                 ngram: tuple[int, int] = (2, 3),
                 min_df: int = 10,
                 log1p: bool = True,
                 seasonal: bool = False,
                 verbose: bool = True,
                 strict: bool = False,
                 crossfit_splits: int = 5,
                 crossfit_seed: int = 17):
        """
        objective    {列名: 权重} —— L1 目标。正=鼓励，负=惩罚
        context      L2 去混杂用的上下文列（谁 / 何时 / 什么主题）
        content      L3 蒸馏用的数值特征列（发帖时即可得）
        text         (标题列, 正文列)；只要标题就传 (标题列,)
        time_col     时间列，用于切训练集
        train_end    训练截止日（含）；None = 全部参与训练
        seasonal     True = 把月份周期项并入 L2 上下文。
                     校园/职场场景的季节性是真实混杂：考试月、假期、开学季
                     的发帖量与被看到的机会都系统性不同，不剥掉会误判。
        """
        self.objective = dict(objective)
        self.context = list(context)
        self.content = list(content)
        self.text = tuple(text) if text else None
        self.time_col = time_col
        self.train_end = train_end
        self.alpha_debias = alpha_debias
        self.alpha_distill = alpha_distill
        self.max_features = max_features
        self.ngram = ngram
        self.min_df = min_df
        self.log1p = log1p
        self.seasonal = seasonal
        self.verbose = verbose
        self.strict = strict
        self.crossfit_splits = int(crossfit_splits)
        self.crossfit_seed = int(crossfit_seed)

    # ------------------------------------------------------------------ 内部
    def _say(self, *a):
        if self.verbose:
            print(*a)

    def _required(self, purpose: str):
        if purpose == "fit":
            names = set(self.objective) | set(self.context) | set(self.content)
            if self.text:
                names.update(self.text)
            if self.time_col:
                names.add(self.time_col)
        elif purpose == "residual":
            names = set(self.objective) | set(self.context)
            if self.seasonal and self.time_col:
                names.add(self.time_col)
        elif purpose == "score":
            names = set(self.content)
            if self.text:
                names.update(self.text)
        else:
            names = set()
        return sorted(names)

    def schema(self) -> dict:
        """返回训练、结算和预排名的输入契约。"""
        return {
            "fit": self._required("fit"),
            "residual": self._required("residual"),
            "score": self._required("score"),
        }

    def _validate_schema(self, df, purpose: str):
        if not self.strict:
            return
        missing = [c for c in self._required(purpose) if c not in df.columns]
        if missing:
            raise ValueError(f"{purpose} 输入缺少必需列: {', '.join(missing)}")

    def _cols(self, df, names):
        missing = [c for c in names if c not in df.columns]
        if self.strict and missing:
            raise ValueError(f"数值特征缺少必需列: {', '.join(missing)}")
        names = [c for c in names if c in df.columns]
        if not names:
            return None
        M = df[names].apply(pd.to_numeric, errors="coerce").astype(float)
        return M.fillna(0.0).to_numpy(np.float32)

    def _docs(self, df):
        if not self.text or self.text[0] not in df.columns:
            return None
        t = df[self.text[0]].fillna("")
        b = (df[self.text[1]].fillna("") if len(self.text) > 1
             and self.text[1] in df.columns else None)
        return build_docs(t.tolist(), None if b is None else b.tolist())

    def _ctx(self, df):
        """L2 的上下文矩阵 = 显式上下文列 ⊕ (可选) 月份周期项。"""
        C = self._cols(df, self.context)
        if self.seasonal and self.time_col and self.time_col in df.columns:
            t = pd.to_datetime(df[self.time_col], utc=True, errors="coerce")
            ang = 2.0 * np.pi * (t.dt.month.to_numpy(float) - 1.0) / 12.0
            cyc = np.nan_to_num(np.c_[np.sin(ang), np.cos(ang)])
            C = cyc if C is None else np.hstack([C, cyc])
        return C

    def _train_mask(self, df) -> np.ndarray:
        if not self.train_end or not self.time_col or self.time_col not in df.columns:
            return np.ones(len(df), dtype=bool)
        t = pd.to_datetime(df[self.time_col], utc=True, errors="coerce")
        cutoff = pd.to_datetime(self.train_end, utc=True)
        # 日期字符串表示“含整天”，避免把 cutoff 当成当天 00:00 而漏掉当天数据。
        if len(str(self.train_end)) <= 10:
            cutoff = cutoff.normalize() + pd.Timedelta(days=1)
            return (t < cutoff).to_numpy()
        return (t <= cutoff).to_numpy()

    def _oof_residual(self, C, u):
        """用 OOF nuisance 预测生成蒸馏目标，避免把 in-sample 残差当真值。"""
        u = np.asarray(u, dtype=float)
        if C is None or len(u) < 2 or self.crossfit_splits < 2:
            return Debiaser(self.alpha_debias).fit(C, u).transform(C, u)
        C = np.asarray(C, dtype=float)
        n_splits = min(self.crossfit_splits, len(u))
        out = np.empty(len(u), dtype=float)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.crossfit_seed)
        for fit_idx, val_idx in kf.split(C):
            nuisance = Debiaser(self.alpha_debias).fit(C[fit_idx], u[fit_idx])
            out[val_idx] = nuisance.transform(C[val_idx], u[val_idx])
        return out

    # ------------------------------------------------------------------ 训练
    def fit(self, df: pd.DataFrame) -> Ranker:
        self._validate_schema(df, "fit")
        tr = self._train_mask(df)
        if not tr.any():
            raise ValueError("train_end 之前没有可用训练行")
        u = compose(df, self.objective, self.log1p)
        C = self._ctx(df)
        Ctr = None if C is None else C[tr]
        utr = u[tr]
        # 最终 nuisance 只在历史训练期拟合；未来行只能 transform。
        self.debias_ = Debiaser(self.alpha_debias).fit(Ctr, utr)
        r_all = self.debias_.transform(C, u)
        # 蒸馏目标用 OOF residual，避免 nuisance 对训练行过拟合后泄漏进 L3。
        r_oof = self._oof_residual(Ctr, utr)
        # 只留标量诊断量。整条训练集的 u / r 存进对象会让交付物膨胀 ~1.5 MB
        # （占整个文件的 83%），而它们随时可以用 residual() 重算。
        self.diag_ = {
            "n_fit": len(df), "n_train": int(tr.sum()), "n_oof": len(r_oof),
            "u_mean_train": float(utr.mean()), "u_std_train": float(utr.std()),
            "r_std_train": float(r_oof.std()), "r_std_all": float(r_all.std()),
            "rho_u_r_train": float(np.corrcoef(utr, r_oof)[0, 1]),
            "crossfit_splits": int(min(self.crossfit_splits, int(tr.sum())))
        }

        docs, X = self._docs(df), self._cols(df, self.content)
        self.distill_ = Distiller(
            alpha=self.alpha_distill, max_features=self.max_features,
            ngram=self.ngram, min_df=self.min_df,
        ).fit(None if docs is None else np.asarray(docs, dtype=object)[tr],
              None if X is None else X[tr], r_oof)

        self.feature_names = [c for c in self.content if c in df.columns]
        self._say(f"[L1] 目标 {len(self.objective)} 路信号 -> u  "
                  f"(训练均值 {utr.mean():.3f}  标准差 {utr.std():.3f})")
        nctx = 0 if C is None else C.shape[1]
        self._say(f"[L2] 去混杂 {nctx} 维上下文 -> OOF 残差 r  "
                  f"(训练标准差 {r_oof.std():.3f}  与 u 相关 {np.corrcoef(utr, r_oof)[0, 1]:.3f})")
        self._say(f"[L3] 蒸馏 词表 {self.distill_.vocab_size:,} + 特征 {len(self.feature_names)} "
                  f"-> 训练 {int(tr.sum()):,} 行")
        return self

    # ------------------------------------------------------------------ 推理
    def residual(self, df: pd.DataFrame) -> np.ndarray:
        """L1+L2：结算量。窗口已关闭时用它，别用模型分。"""
        self._validate_schema(df, "residual")
        u = compose(df, self.objective, self.log1p)
        return self.debias_.transform(self._ctx(df), u)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """L1+L2+L3：预测分。窗口未关闭 / 冷启动时用它。"""
        self._validate_schema(df, "score")
        return self.distill_.predict(self._docs(df), self._cols(df, self.content))

    def allocate(self, df: pd.DataFrame, scores=None, scheme=None, blocked=None,
                 rankable=None, extra=None) -> pd.DataFrame:
        """L4：档位 + 积分。默认按模型分排名。"""
        if scores is None:
            scores = self.score(df)
        return allocate(scores, scheme=scheme, blocked=blocked,
                        rankable=rankable, extra=extra)

    def explain(self, df: pd.DataFrame, top: int = 12) -> pd.DataFrame:
        """看哪些词/特征把分数推高或压低 —— 线性模型的免费福利。"""
        m = self.distill_.model_
        names = list(getattr(self.distill_.tfidf_, "get_feature_names_out", lambda: [])()) \
            + self.feature_names
        coef = np.asarray(m.coef_).ravel()
        d = pd.DataFrame({"term": names[:len(coef)], "coef": coef})
        d["abs"] = d["coef"].abs()
        return d.sort_values("abs", ascending=False).head(top).drop(columns="abs")

    # ------------------------------------------------------------------ 持久化
    def save(self, path: str) -> str:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)
        try:
            from .artifact import write_manifest
            write_manifest(self, path)
        except Exception as exc:
            self._say(f"[manifest] 写入失败: {exc}")
        return path

    @staticmethod
    def load(path: str) -> Ranker:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        # 兼容 P0 之前导出的 provisional pickle；新模型会显式写入这些字段。
        if not hasattr(obj, "strict"):
            obj.strict = False
        if not hasattr(obj, "crossfit_splits"):
            obj.crossfit_splits = 1
        if not hasattr(obj, "crossfit_seed"):
            obj.crossfit_seed = 17
        return obj

    @property
    def vocab_size(self) -> int:
        return self.distill_.vocab_size
