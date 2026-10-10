# -*- coding: utf-8 -*-
"""应用：西南交大吧 每周创作激励。

这是 ResRank 的一个应用 —— 换掉下面这几行配置，同一套算法就能用在
别的内容场景（视频、专栏、问答、商品评价）。

我们奖励的是**创作**，不是流量：
    + 自建楼（self_long_uniq / self_deep_uniq）—— 真在写东西
    - 自顶（self_bump）—— 把帖子顶上去，不是内容
互动量（回复/评论/广度/活跃天数）只作为**次要**信号，且必须先去混杂。

用法
----
    python -m apps.tieba_reward train --input data/reward/thread_residual_weekly.parquet
    python -m apps.tieba_reward plan  --input data/reward/thread_residual_weekly.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resrank import Ranker, allocate, summarize

# --------------------------------------------------------------------- 配置
# L1 目标：权重是**价值判断**，不是可以随便拟合的参数。
# 创作侧权重（1.15 / 0.80）压过互动侧（0.60 / 0.30），自顶为负。
OBJECTIVE = {
    "reply_7d": 0.60, "comment_7d": 0.30, "breadth_7d": 0.30, "actdays_7d": 0.18,
    "self_long_uniq": 1.15, "self_deep_uniq": 0.80,
    "self_bump": -0.60,
}

# L2 上下文：谁发的 / 何时发的 / 什么主题 —— 暴露偏差的全部来源
CONTEXT = ["author_prior", "log_author_prior_n", "author_level",
           "month_activity", "day_activity", "topic_te"]

# L3 服务时特征：内容结构 + 创建时可得的校准特征。
# author_level 不是“创作质量”本身，只允许在 residual 目标下作为校准变量；
# 发布前必须检查作者相关性、新作者覆盖和作者集中度。
CONTENT = ["len_title", "len_text", "info_density", "n_digits", "n_entities",
           "has_bracket_prefix", "is_question", "low_effort", "n_emoji",
           "n_image", "n_link", "n_at", "n_video", "emoji_ratio", "image_ratio",
           "has_image", "has_video", "has_link", "img_bytes_log", "img_px_log",
           "media_richness", "author_level"]

TRAIN_END = "2022-12-31"
DUP_SAT, DUP_W = 8.0, 0.50     # 重复项饱和点 / 权重
INTEGRITY_GATE = 0.60

# 官方精华率仅 0.2344%，所以闸门要松、分档要严 —— 让绝大多数人得 0 分
TIER_MIN, TIER_MID = 250, 100


def build_design(df: pd.DataFrame) -> pd.DataFrame:
    """补齐算法需要的列。已有列原样保留，缺哪列补哪列。"""
    from resrank.features import media_features, text_features

    df = df.copy()
    for _name, fn in (("text", text_features), ("media", media_features)):
        add = fn(df)
        for c in add.columns:
            if c not in df.columns:
                df[c] = add[c]
    if "n_entities" not in df.columns:
        ent = ["交大", "西南交大", "犀浦", "九里", "峨眉", "校区", "专业", "学院",
               "考研", "保研", "选课", "宿舍", "食堂", "就业", "校招", "实习",
               "绩点", "考试", "军训", "校庆"]
        both = df.get("title", pd.Series([""] * len(df))).fillna("").astype(str) + " " + \
               df.get("text", pd.Series([""] * len(df))).fillna("").astype(str)
        df["n_entities"] = both.apply(lambda s: sum(1 for e in ent if e in s))
    # 周队列：奖励是按周发的，分位就必须在周内算，否则「这周发帖少」会被惩罚
    if "week" not in df.columns and "create_time" in df.columns:
        iso = pd.to_datetime(df["create_time"], utc=True, errors="coerce").dt.isocalendar()
        df["week"] = (iso["year"].astype("Int64").astype("string") + "-W"
                      + iso["week"].astype("Int64").astype("string").str.zfill(2))
    return df


def _closed_mask(df: pd.DataFrame) -> np.ndarray:
    """读取唯一窗口状态；缺失时拒绝静默猜测。"""
    col = "window_complete" if "window_complete" in df.columns else "closed"
    if col not in df.columns:
        raise ValueError("输入必须包含 window_complete（或兼容字段 closed）")
    raw = df[col]
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(False).to_numpy(bool)
    numeric = pd.to_numeric(raw, errors="coerce")
    out = numeric.eq(1).to_numpy(bool).copy()
    text = raw.astype("string").str.strip().str.lower()
    out |= text.isin(["true", "yes", "complete", "closed", "已闭合"]).to_numpy(bool)
    return out


def gate(df: pd.DataFrame, settled: np.ndarray | None = None) -> tuple:
    """L4 闸门：区分低质量、重复、数据不足和可排名对象。

    用 AND 口径（短正文 **且** 短标题 **且** 无媒体）而不是 OR ——
    OR 会误伤「正文当引子、内容在回复里」的连载帖，实测会拦掉官方精华。
    """
    n = len(df)
    if settled is None:
        settled = _closed_mask(df)
    settled = np.asarray(settled, dtype=bool)
    dup = pd.to_numeric(df.get("dup_count_global", pd.Series(1, index=df.index)),
                        errors="coerce").fillna(1).clip(lower=1).astype(float)
    d = np.where(dup > 1, np.minimum(1.0, np.log1p(dup) / np.log1p(DUP_SAT)), 0.0)

    if "op_self_share" in df.columns:
        s = pd.to_numeric(df["op_self_share"], errors="coerce").fillna(0).clip(0, 1).to_numpy()
    else:
        s = np.zeros(n)

    len_text = pd.to_numeric(df.get("len_text", pd.Series(0, index=df.index)),
                             errors="coerce").fillna(0).to_numpy()
    len_title = pd.to_numeric(df.get("len_title", pd.Series(0, index=df.index)),
                              errors="coerce").fillna(0).to_numpy()
    has_media = np.zeros(n, dtype=bool)
    for c in ("n_image", "n_link", "n_video"):
        if c in df.columns:
            has_media |= pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy() > 0
    low_effort = (len_text <= 5) & (len_title <= 3) & ~has_media

    integrity = np.clip(1.0 - (DUP_W * d + 0.3 * s + 0.3 * low_effort), 0.0, 1.0)
    blocked = integrity < INTEGRITY_GATE
    # 已闭合行没有创作结果 -> 无法结算，标为数据不足而非拦下。
    outcome_cols = list(OBJECTIVE)
    missing_outcome = np.zeros(n, dtype=bool)
    for c in outcome_cols:
        if c not in df.columns:
            missing_outcome |= settled
        else:
            missing_outcome |= pd.to_numeric(df[c], errors="coerce").isna().to_numpy()
    insufficient = settled & missing_outcome
    reason = np.full(n, "eligible", dtype=object)
    reason[blocked & insufficient] = "blocked_and_insufficient"
    reason[blocked & ~insufficient] = "blocked_integrity"
    reason[~blocked & insufficient] = "insufficient_settlement_data"
    return blocked, ~insufficient, integrity, reason


def make_ranker() -> Ranker:
    return Ranker(objective=OBJECTIVE, context=CONTEXT, content=CONTENT,
                  text=("title", "text"), time_col="create_time", train_end=TRAIN_END,
                  seasonal=True, alpha_debias=15.0, alpha_distill=30.0,
                  strict=True, crossfit_splits=5, crossfit_seed=17)


def plan_dataframe(df: pd.DataFrame, ranker: Ranker, cohort: str = "week",
                   preview: bool = False) -> tuple:
    """生成不写外部状态的奖励计划,供 CLI、shadow 和回放共用。"""
    df = df.reset_index(drop=True)
    closed = np.zeros(len(df), dtype=bool) if preview else _closed_mask(df)
    blocked, rankable, integrity, reason = gate(df, settled=closed)
    scores = ranker.score(df) if preview else np.zeros(len(df), dtype=float)
    if not preview:
        # 闭合但缺结果的行已由 gate 标记为 insufficient，不得触发 strict residual。
        settled_ready = closed & rankable
        if settled_ready.any():
            scores[settled_ready] = ranker.residual(df.loc[settled_ready])
        if (~closed).any():
            scores[~closed] = ranker.score(df.loc[~closed])

    if cohort and cohort in df.columns:
        parts = []
        for _, group in df.groupby(cohort, sort=False, dropna=False):
            idx = group.index.to_numpy()
            part = allocate(scores[idx], blocked=blocked[idx], rankable=rankable[idx])
            part.index = idx
            parts.append(part)
        plan = pd.concat(parts).sort_index() if parts else allocate(scores)
    else:
        plan = allocate(scores, blocked=blocked, rankable=rankable)
    plan["integrity"] = integrity
    plan["eligible"] = rankable & ~blocked
    plan["eligibility_reason"] = reason
    plan["settled"] = closed
    return scores, plan


# ----------------------------------------------------------------------- CLI
def cmd_train(args) -> int:
    df = build_design(pd.read_parquet(args.input))
    r = make_ranker().fit(df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    r.save(str(out))
    print(f"\nWROTE {out} ({out.stat().st_size / 1e6:.3f} MB)")
    print("\n=== 权重最高的 12 项 ===")
    print(r.explain(df).to_string(index=False))
    return 0


def cmd_plan(args) -> int:
    df = build_design(pd.read_parquet(args.input))
    r = Ranker.load(args.model)
    _, plan = plan_dataframe(df, r, cohort=args.cohort)
    if args.verbose and args.cohort and args.cohort in df.columns:
        for key, group in df.reset_index(drop=True).groupby(args.cohort, sort=False, dropna=False):
            s = summarize(plan.loc[group.index])
            print("  %-12s 队列 %6d  参评 %6d  获奖 %5d  积分 %8.0f  档位 %s"
                  % (str(key), s["n"], s["ranked"], s["paid"], s["points_total"], s["tiers"]))
    out = pd.concat([df[["tid", "title"]].reset_index(drop=True), plan.reset_index(drop=True)], axis=1)
    print("=== 分配总览 ===")
    for k, v in summarize(plan).items():
        print(f"  {k:14} {v}")
    print("\n=== 前 10 名 ===")
    top = out.sort_values("score", ascending=False).head(10)
    for _, row in top.iterrows():
        print(f"  T{row['tier']:<2} {row['points']:>5.0f}  {str(row['title'])[:46]}")
    if args.out:
        out.to_parquet(args.out, index=False)
        print(f"\nWROTE {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="贴吧每周创作激励")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="训练并导出单文件模型")
    t.add_argument("--input", required=True)
    t.add_argument("--out", default="models/reward/resrank.pkl")
    t.set_defaults(func=cmd_train)
    p = sub.add_parser("plan", help="对一批帖出奖励方案")
    p.add_argument("--input", required=True)
    p.add_argument("--model", default="models/reward/resrank.pkl")
    p.add_argument("--out", default="")
    p.add_argument("--cohort", default="", help="队列列名（如 week）；留空则整表当一个队列")
    p.add_argument("--quiet", dest="verbose", action="store_false", default=True)
    p.set_defaults(func=cmd_plan)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
