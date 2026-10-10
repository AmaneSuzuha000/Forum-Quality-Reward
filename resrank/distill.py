# -*- coding: utf-8 -*-
"""L3 蒸馏 —— 学一个只吃内容的打分器 ŝ(x) ≈ r。

为什么需要它：残差 r 要等 7 天窗口关闭才算得出来，而奖励要在窗口内发。
蒸馏把「已结算的残差」压进内容特征，得到冷启动即可用的打分器。

为什么是线性 + TF-IDF：在同一个域内排序（同一个吧、同一批人、同一个话题域），
语料自身的词频结构就够用 —— 实测 8000 词表 char n-gram 的稀疏 TF-IDF
接 Ridge，比 512 维通用预训练向量、比双塔神经网络都更准，
工件小两个数量级，没有随机性，CPU 上万帖/秒。
"""
from __future__ import annotations

import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

TITLE_CHARS = 160
BODY_CHARS = 400
_WS = re.compile(r"\s+")


def clean(s, limit: int) -> str:
    """折叠空白并截断 —— 训练与线上必须走同一个函数。"""
    return _WS.sub(" ", str(s if s is not None else "").strip())[:limit]


def build_doc(title, body=None) -> str:
    """文本契约。**改了它就必须重新拟合**，否则线上与训练分布不一致。"""
    parts = ["标题：" + clean(title, TITLE_CHARS)]
    b = clean(body, BODY_CHARS)
    if b:
        parts.append("正文：" + b)
    return "\n".join(parts)


def build_docs(titles, bodies=None) -> list:
    if bodies is None:
        bodies = [None] * len(titles)
    return [build_doc(t, b) for t, b in zip(titles, bodies, strict=False)]


class Distiller:
    """稀疏 TF-IDF(char n-gram) ⊕ 数值特征 → Ridge。

    文本与数值特征都可选，但不能同时为空。
    标准化常数与词表都在训练子集上拟合 —— 这一点必须守住，
    否则测试期的统计量会渗进训练。
    """

    def __init__(self, alpha: float = 30.0, max_features: int = 8000,
                 ngram: tuple[int, int] = (2, 3), min_df: int = 10):
        self.alpha = alpha
        self.max_features = max_features
        self.ngram = ngram
        self.min_df = min_df

    # ---- 内部 ----
    def _stack(self, blocks):
        return hstack(blocks).tocsr() if len(blocks) > 1 else blocks[0]

    # ---- 拟合 / 预测 ----
    def fit(self, docs, X, y) -> Distiller:
        """docs / X 传**训练子集**即可；两者分别可以为 None。"""
        self.tfidf_ = None
        self.mu_ = self.sd_ = None
        blocks = []
        if docs is not None:
            self.tfidf_ = TfidfVectorizer(
                analyzer="char_wb", ngram_range=self.ngram, min_df=self.min_df,
                max_features=self.max_features, sublinear_tf=True, dtype=np.float32)
            blocks.append(self.tfidf_.fit_transform(list(docs)))
        if X is not None:
            X = np.asarray(X, dtype=float)
            self.mu_ = X.mean(0)
            sd = X.std(0)
            self.sd_ = np.where(sd == 0.0, 1.0, sd)
            blocks.append(csr_matrix((X - self.mu_) / self.sd_))
        if not blocks:
            raise ValueError("docs 与 X 不能同时为空")
        self.model_ = Ridge(alpha=self.alpha, solver="lsqr")
        self.model_.fit(self._stack(blocks), np.asarray(y, dtype=float))
        return self

    def predict(self, docs, X) -> np.ndarray:
        blocks = []
        if getattr(self, "tfidf_", None) is not None:
            blocks.append(self.tfidf_.transform(list(docs)))
        if getattr(self, "mu_", None) is not None:
            X = np.asarray(X, dtype=float)
            blocks.append(csr_matrix((X - self.mu_) / self.sd_))
        if not blocks:
            raise ValueError("docs 与 X 不能同时为空")
        return self.model_.predict(self._stack(blocks))

    @property
    def vocab_size(self) -> int:
        return 0 if getattr(self, "tfidf_", None) is None else len(self.tfidf_.vocabulary_)
