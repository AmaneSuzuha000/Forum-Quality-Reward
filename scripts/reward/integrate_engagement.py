# -*- coding: utf-8 -*-
"""把抓到的互动指标与官方质量标注并入特征表。

新增特征
--------
view_num      浏览量
agree_num     点赞数
disagree_num  踩数
share_num     分享数
is_good       ★ 官方精华标注   —— 真正的 ground truth, 可用于验证奖励模型
is_top        置顶
engagement_rate = (agree+reply+share) / view   —— 互动率(抗流量偏置)

同时产出 **官方标注评估集**: 有 is_good 的帖子可用于独立检验奖励模型
是否真的在识别"好帖", 而不是只在拟合互动量。

用法: python scripts/reward/integrate_engagement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resrank.runtime import ROOT, connect, forum_id, log, setup_stdout

setup_stdout()


import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

DATA = ROOT / "data" / "reward"
FID = forum_id()



def main() -> int:
    conn = connect()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT te.tid, te.view_num, te.agree_num, te.disagree_num, te.share_num,
               te.reply_num_live, te.is_good, te.is_top, te.is_share,
               te.create_time AS te_create
        FROM analysis.thread_engagement te WHERE te.fid = %s""", (FID,))
    eng = pd.DataFrame(cur.fetchall())
    conn.close()
    log(f"engagement rows: {len(eng):,}")

    feat = pd.read_parquet(DATA / "thread_features.parquet")
    log(f"features rows: {len(feat):,}")

    before = len(feat.columns)
    df = feat.merge(eng, on="tid", how="left")
    n_match = df["view_num"].notna().sum()
    log(f"matched: {n_match:,} / {len(df):,} ({100*n_match/len(df):.1f}%)")

    # ---- 派生互动特征 ----
    for c in ["view_num","agree_num","disagree_num","share_num","reply_num_live"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_good"] = df["is_good"].fillna(False).astype(bool)
    df["is_top"] = df["is_top"].fillna(False).astype(bool)
    df["has_engagement"] = df["view_num"].notna().astype(int)

    # 对数化(长尾极重: view 可达 6 万)
    df["log_view"]  = np.log1p(df["view_num"].clip(lower=0))
    df["log_agree"] = np.log1p(df["agree_num"].clip(lower=0))
    df["log_share"] = np.log1p(df["share_num"].clip(lower=0))

    # 互动率: 抗"流量大所以什么都高"的偏置
    v = df["view_num"].replace(0, np.nan)
    df["agree_rate"] = (df["agree_num"] / v).clip(0, 1)
    df["reply_rate"] = (df["reply_num_live"] / v).clip(0, 5)
    df["share_rate"] = (df["share_num"] / v).clip(0, 1)
    df["engagement_rate"] = (df["agree_num"].fillna(0) + df["reply_num_live"].fillna(0)
                             + df["share_num"].fillna(0)) / v
    df["engagement_rate"] = df["engagement_rate"].clip(0, 5)
    # 赞踩比 (争议度反向指标)
    df["like_ratio"] = df["agree_num"] / (df["agree_num"] + df["disagree_num"]).replace(0, np.nan)

    # 官方标注同时更新 utility: 现在有真实互动了
    # 新效用 = 旧的(回复/评论/广度/存活) + 浏览/点赞/分享
    df["utility_engagement"] = (df["utility"]
                        + 0.6 * df["log_view"].fillna(0)
                        + 0.8 * df["log_agree"].fillna(0)
                        + 0.4 * df["log_share"].fillna(0))

    out = DATA / "thread_features_enriched.parquet"
    df.to_parquet(out, index=False)
    log(f"WROTE {out} ({len(df):,} x {len(df.columns)}, was {before} cols)")

    print("\n=== 新增互动特征 (仅已抓到的帖子) ===")
    sub = df[df["has_engagement"] == 1]
    for c in ["view_num","agree_num","share_num","reply_num_live",
              "agree_rate","engagement_rate","like_ratio"]:
        s = pd.to_numeric(sub[c], errors="coerce")
        print(f"  {c:18} n={s.notna().sum():>7,} mean={s.mean():>10.2f} "
              f"p50={s.median():>9.2f} p90={s.quantile(.9):>11.2f} max={s.max():>12.0f}")

    print("\n=== 官方标注 ===")
    print(f"  is_good=True : {df['is_good'].sum():,}")
    print(f"  is_top=True  : {df['is_top'].sum():,}")

    # 官方标注 vs 我们已有的 reward_score —— 独立检验
    try:
        rs = pd.read_parquet(DATA / "reward_scores.parquet")
        col = "reward_score"
        m = df.merge(rs[["tid", col]], on="tid", how="left")
        m2 = m[m["has_engagement"] == 1]
        if m2["is_good"].sum() > 5:
            from scipy.stats import mannwhitneyu
            g = m2.loc[m2["is_good"], col].dropna()
            b = m2.loc[~m2["is_good"], col].dropna()
            if len(g) > 5 and len(b) > 5:
                u, p = mannwhitneyu(g, b, alternative="greater")
                auc = u / (len(g) * len(b))
                print("\n=== 独立检验: 奖励模型 vs 官方精华标注 ===")
                print(f"  精华帖 reward_score 中位={g.median():.1f}, 普通帖={b.median():.1f}")
                print(f"  Mann-Whitney U p={p:.3e}   AUC={auc:.4f}  (>0.5 说明模型确实识别好帖)")
    except Exception as e:
        print("  独立检验跳过:", type(e).__name__, str(e)[:90])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
