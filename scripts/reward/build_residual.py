# -*- coding: utf-8 -*-
"""L1+L2：去混杂目标与残差构造。

直接拿互动量当奖励目标，学到的是「谁发的」而不是「写得好不好」——
实测中仅作者身份一项就解释了 ln(1+reply_num) 方差的 R²=0.6488。

本脚本先合成 utility（多路信号 → 单标量目标），再用岭回归拟合
「谁发的 / 何时发的 / 什么主题」这组 nuisance 变量，取残差作为目标：

    residual = utility − E[utility | 作者先验, 版面活跃度, 主题]

残差回答「这篇比同等条件下的平均水平好多少」，是 L3 蒸馏的标签。

输出:
    data/reward/thread_residual.parquet          终生口径
    data/reward/thread_residual_weekly.parquet   7 天窗口口径 (--weekly)

用法:
    python scripts/reward/build_residual.py --weekly
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resrank.runtime import ROOT, log, setup_stdout

setup_stdout()

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

DATA = ROOT / "data" / "reward"
TRAIN_END = "2022-12-31"

# 周窗口口径的合成权重：创作侧压过互动侧，自顶为负。
W_LONG, W_DEEP, W_BUMP = 1.15, 0.80, -0.60
W_REPLY, W_COMMENT, W_BREADTH, W_ACTDAYS = 0.60, 0.30, 0.30, 0.18
# 纯长度权重必须为 0：len_text 本身是模型特征，放进目标即构成目标泄漏。
W_LEN = 0.0

AUTHOR_PRIOR_W = 5.0    # 作者先验向全局均值的收缩强度
TOPIC_SMOOTH = 20.0     # 主题目标编码的平滑强度
RIDGE_ALPHA = 5.0

# 周窗口里从 window7d.parquet 并入的列
WINDOW_COLS = ["tid", "reply_7d", "reply_3d", "self_7d", "breadth_7d", "actdays_7d",
               "comment_7d", "comment_life", "cb_7d", "late_ratio_7d",
               "first_reply_min", "window_complete", "reply_life_others"]


def _num(frame, col):
    """取一列为非负数值 Series；列不存在时返回等长的 0。"""
    if col not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(0.0).clip(lower=0)


def load_features() -> pd.DataFrame:
    """载入特征表。优先用并入互动指标的那张，缺则退回基础表。"""
    enriched = DATA / "thread_features_enriched.parquet"
    src = enriched if enriched.exists() else DATA / "thread_features.parquet"
    df = pd.read_parquet(src)
    log(f"loaded {src.name}: {len(df):,} x {len(df.columns)} cols")
    return df


def merge_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """并入 7 天窗口指标与楼主创作量，并剔除窗口未满的帖子。"""
    window = DATA / "window7d.parquet"
    if not window.exists():
        raise SystemExit("需要先运行 build_window_features.py 产出 window7d.parquet")
    w = pd.read_parquet(window)
    df = df.merge(w[[c for c in WINDOW_COLS if c in w.columns]], on="tid", how="left")

    self_content = DATA / "self_content.parquet"
    if self_content.exists():
        sc = pd.read_parquet(self_content)
        use = ["tid"] + [c for c in ("self_long_uniq", "self_deep_uniq", "self_bump")
                         if c in sc.columns]
        df = df.merge(sc[use], on="tid", how="left")
        for c in ("self_long_uniq", "self_deep_uniq", "self_bump"):
            df[c] = _num(df, c)
        log(f"  merged self_content: long_uniq>0 = {(df['self_long_uniq'] > 0).sum():,}")
    else:
        log("  WARN 缺少 self_content.parquet，创作侧信号退化为全 0")
        for c in ("self_long_uniq", "self_deep_uniq", "self_bump"):
            df[c] = 0.0

    # 窗口未满 7 天的帖子必须剔除，否则会被系统性低估。
    before = len(df)
    df = df[df["window_complete"].fillna(0).astype(bool)].reset_index(drop=True)
    log(f"[weekly] 剔除窗口未满 7 天的帖子: {before:,} -> {len(df):,}")
    return df


def compose_utility(df: pd.DataFrame, weekly: bool) -> pd.DataFrame:
    """按窗口口径或终生口径合成 utility。"""
    if weekly:
        reply = _num(df, "reply_7d")
        comment = _num(df, "comment_7d")
        breadth = _num(df, "breadth_7d")
        actdays = _num(df, "actdays_7d")

        # 保留纯「他人反馈」口径作为对照列
        df["utility_engage_only"] = (np.log1p(reply)
                                     + 0.5 * np.log1p(comment)
                                     + 0.5 * np.log1p(breadth)
                                     + 0.3 * np.log1p(actdays))

        engage = (W_REPLY * np.log1p(reply)
                  + W_COMMENT * np.log1p(comment)
                  + W_BREADTH * np.log1p(breadth)
                  + W_ACTDAYS * np.log1p(actdays))
        # 自顶权重是负数，必须**相加**。写成减号会让顶帖变成加分项。
        # 上界检查：utility 不得超过「去掉顶帖项」的和 —— 超过即说明符号写反了。
        no_bump = (W_LONG * np.log1p(_num(df, "self_long_uniq"))
                   + W_DEEP * np.log1p(_num(df, "self_deep_uniq")))
        creation = no_bump + W_BUMP * np.log1p(_num(df, "self_bump"))
        df["utility_creation"] = creation
        df["utility"] = creation + engage
        if (df["utility"] > no_bump + engage + 1e-9).any():
            raise SystemExit("[BUG] 顶帖惩罚符号错误：自顶反而提高了 utility")
        log(f"  utility = {W_LONG}*long + {W_DEEP}*deep + {W_BUMP}*bump "
            f"+ {W_REPLY}/{W_COMMENT}/{W_BREADTH}/{W_ACTDAYS}*参与7d; "
            f"mean={df['utility'].mean():.3f}")
    elif "utility" in df.columns:
        # 特征表已带效用列，直接沿用，避免与上游口径漂移。
        df["utility"] = pd.to_numeric(df["utility"], errors="coerce").fillna(0.0)
    else:
        df["utility"] = (np.log1p(_num(df, "reply_num"))
                         + 0.5 * np.log1p(_num(df, "n_comments"))
                         + 0.5 * np.log1p(_num(df, "n_distinct_authors"))
                         + 0.3 * np.log1p(_num(df, "survival_days")))

    # 合成目标前必须确认：目标里的加项不得出现在模型内容特征里。
    from apps.tieba_reward import CONTENT as MODEL_CONTENT
    used = {"self_long_uniq", "self_deep_uniq", "self_bump"}
    if W_LEN:
        used.add("len_text")
    overlap = used & set(MODEL_CONTENT)
    if overlap:
        raise SystemExit(f"[LEAK] utility 的加项与模型内容特征重叠: {sorted(overlap)}")
    log(f"  防泄漏检查通过（W_LEN={W_LEN}，目标加项不在模型特征内）")
    return df


def use_engagement_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """终生口径下，已抓到互动指标的帖子改用含浏览/点赞/分享的效用。"""
    if "utility_engagement" not in df.columns:
        return df
    has = _num(df, "has_engagement")
    if int(has.sum()) <= 1000:
        return df
    df["utility_lifetime"] = df["utility"]
    df["utility"] = np.where(has.astype(bool), df["utility_engagement"], df["utility"])
    # 把「互动指标缺失」显式标出来，免得模型把无数据的帖当成低质。
    df["eng_missing"] = (1 - has).astype(float)
    log(f"  utility -> utility_engagement for {int(has.sum()):,} posts")
    return df


def add_nuisance(df: pd.DataFrame) -> tuple:
    """构造 nuisance 设计矩阵：作者先验 / 版面活跃度 / 主题目标编码。"""
    df = df.sort_values("ct").reset_index(drop=True)
    global_mean = df["utility"].mean()

    # 作者先验用 as-of-time：只用该帖之前的历史，既是真实部署条件也无泄漏。
    df = df.sort_values(["author_id", "ct"]).reset_index(drop=True)
    by_author = df.groupby("author_id")["utility"]
    prior_sum = by_author.transform(lambda s: s.shift().expanding().sum())
    prior_cnt = by_author.transform(lambda s: s.shift().expanding().count())
    df["author_prior"] = ((prior_sum + AUTHOR_PRIOR_W * global_mean)
                          / (prior_cnt + AUTHOR_PRIOR_W)).astype(float)
    df["author_prior_n"] = prior_cnt.fillna(0).astype(float)
    df["log_author_prior_n"] = np.log1p(df["author_prior_n"])
    cold = df["author_prior_n"] == 0
    df.loc[cold, "author_prior"] = global_mean
    log(f"  cold-start posts: {cold.sum():,} ({100 * cold.mean():.1f}%)")

    # 版面活跃度不涉及目标，测试期同样可得。
    df["month"] = df["ct"].dt.tz_convert("Asia/Shanghai").dt.to_period("M").astype(str)
    df["month_activity"] = np.log1p(df.groupby("month")["tid"].transform("size"))
    df["month_idx"] = (df["ct"].dt.year - 2006) * 12 + df["ct"].dt.month
    df["day_activity"] = np.log1p(
        df.groupby(df["ct"].dt.strftime("%Y-%m-%d"))["tid"].transform("size"))

    # 主题目标编码只在训练期拟合，测试期用不到新主题。
    is_train = (df["ct"] <= pd.Timestamp(TRAIN_END, tz="UTC")).to_numpy()
    is_val = (df["ct"].dt.year == 2023).to_numpy()
    is_test = (df["ct"].dt.year >= 2024).to_numpy()
    log(f"split train={is_train.sum():,} val={is_val.sum():,} test={is_test.sum():,}")

    df["topic_id_f"] = df["topic_id"].fillna(-1).astype("int64")
    train = df[is_train]
    train_mean = train["utility"].mean()
    stat = train.groupby("topic_id_f")["utility"].agg(["sum", "count"])
    encoding = (stat["sum"] + TOPIC_SMOOTH * train_mean) / (stat["count"] + TOPIC_SMOOTH)
    df["topic_te"] = df["topic_id_f"].map(encoding).fillna(train_mean).astype(float)

    # 设计矩阵不含与目标编码冗余的 one-hot，避免共线。
    cols = ["author_prior", "log_author_prior_n", "author_level",
            "month_activity", "day_activity", "topic_te"]
    if "eng_missing" in df.columns:
        cols.append("eng_missing")
    X = df[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.fillna(X.median())
    trend = df["month_idx"].to_numpy(dtype=float).reshape(-1, 1) / 12.0
    design = np.hstack([X.to_numpy(), trend])
    names = [*cols, "month_idx"]

    mu, sd = design[is_train].mean(0), design[is_train].std(0)
    sd[sd == 0] = 1.0
    log(f"  design matrix: {design.shape}")
    return df, (design - mu) / sd, names, is_train, is_val, is_test


def report(df, y, yhat, model, names, is_train, is_val, is_test) -> None:
    """打印解释力、系数与残差分布。"""
    print("\n=== 解释力 ===")
    for label, mask in (("train", is_train), ("val", is_val), ("test", is_test)):
        print(f"  R2({label:5}) = {r2_score(y[mask], yhat[mask]):+.4f}")

    print("\n=== 对照：单变量解释力 ===")
    print("  [A] 全样本均值 —— 含泄漏，表示混杂的信息上限")
    for label, series in (("作者", df.groupby("author_id")["utility"].transform("mean")),
                          ("月份", df.groupby("month")["utility"].transform("mean")),
                          ("主题", df.groupby("topic_id_f")["utility"].transform("mean"))):
        print(f"      R2({label}全样本均值)   = {r2_score(y, series.to_numpy()):+.4f}")
    print("  [B] 仅用过去 / 仅用训练集 —— 无泄漏，表示可部署能力")
    print(f"      R2(作者 as-of 先验)  = {r2_score(y, df['author_prior'].to_numpy()):+.4f}")
    print(f"      R2(版面活跃度 单独)  = {r2_score(y, df['month_activity'].to_numpy()):+.4f}")

    print("\n=== nuisance 系数（标准化） ===")
    coef = pd.Series(model.coef_, index=names)
    for k, v in coef.sort_values(key=np.abs, ascending=False).items():
        print(f"  {k:24} {v:+.4f}")

    print("\n=== residual 分布 ===")
    for label, mask in (("train", is_train), ("val", is_val), ("test", is_test)):
        r = df.loc[mask, "residual"]
        print(f"  {label:5} mean={r.mean():+.4f} sd={r.std():.4f} "
              f"p10={r.quantile(.1):+.3f} p50={r.median():+.3f} p90={r.quantile(.9):+.3f}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="去混杂残差目标构造")
    ap.add_argument("--weekly", action="store_true",
                    help="周度模式：效用改用发帖后 7 天窗口，而非终生累计值")
    args = ap.parse_args()

    df = load_features()
    if args.weekly:
        df = merge_weekly(df)
    df = compose_utility(df, args.weekly)
    if not args.weekly:
        df = use_engagement_snapshot(df)

    df["ct"] = pd.to_datetime(df["create_time"], utc=True)
    df, design, names, is_train, is_val, is_test = add_nuisance(df)

    y = df["utility"].to_numpy(dtype=float)
    model = Ridge(alpha=RIDGE_ALPHA, random_state=0)
    model.fit(design[is_train], y[is_train])
    yhat = model.predict(design)
    df["utility_hat"] = yhat
    df["residual"] = y - yhat
    report(df, y, yhat, model, names, is_train, is_val, is_test)

    out = DATA / ("thread_residual_weekly.parquet" if args.weekly
                  else "thread_residual.parquet")
    df.drop(columns=["ct"]).to_parquet(out, index=False)
    log(f"WROTE {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
