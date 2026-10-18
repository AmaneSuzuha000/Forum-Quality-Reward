# -*- coding: utf-8 -*-
"""决策规则层: 把 reward_score 转成可发放的奖励。

设计要点(针对 作者 R2 混杂 63.5% 的现实):
  以 **作者内相对进步** 为主, 跨作者绝对排名为辅 ——
  否则新人永远拿不到奖励(马太效应)。

用法:
  python scripts/reward/score_posts.py --top 20
  python scripts/reward/score_posts.py --tid 10284430088
  python scripts/reward/score_posts.py --author 1669907408
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resrank.runtime import ROOT, setup_stdout

setup_stdout()

import argparse

import numpy as np
import pandas as pd

DATA = ROOT / "data" / "reward"
INTEGRITY_GATE = 0.60     # 低于此不进入奖励池
# 分位边界: 6 条边 -> 5 档。T0 既是"诚信不达标"也是"分位不足"的落点。
TIER_EDGES = [0.0, 0.60, 0.75, 0.90, 0.97, 1.0001]
TIER_LABELS = ["T0不奖", "T1", "T2", "T3", "T4顶级"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tid", type=int); ap.add_argument("--author", type=int)
    ap.add_argument("--top", type=int, default=0); ap.add_argument("--year", type=int, default=0)
    a = ap.parse_args()

    f = pd.read_parquet(DATA / "thread_residual.parquet")
    s = pd.read_parquet(DATA / "reward_scores.parquet")
    score_col = "reward_score_raw" if "reward_score_raw" in s.columns else "reward_score"
    df = f.merge(s[["tid", score_col]].rename(columns={score_col: "reward_score_raw"}),
                 on="tid", how="left")

    # ---- 标定 (关键) --------------------------------------------------
    # 网络输出只有**排序**可信(pairwise acc 0.60), 量纲未标定:
    # 实测有 residual=+0.26 却 score=+9.4 的帖子 —— 模型在尾部外推。
    # 所以对外分数用 **百分位** 而非原始标量:
    #   - 完全保留排序信息(没有 winsorize 造成的人为并列)
    #   - 天然有界 [-100, +100], 可直接跨批次比较
    #   - 奖励分档本来就是按名次发的, 所以这是与决策一致的表示
    raw = df["reward_score_raw"].to_numpy(dtype=float)
    med = np.nanmedian(raw)
    pct = df["reward_score_raw"].rank(pct=True).to_numpy()
    df["reward_z"] = np.clip((raw - med) / (1.4826 * (np.nanmedian(np.abs(raw - med)) or 1.0)),
                             -6, 6)                       # 仅供诊断, 已限幅
    df["reward_score"] = (pct - 0.5) * 200.0              # ← 对外分: [-100, +100]
    df["score_raw"] = raw

    df["eligible"] = df["integrity"] >= INTEGRITY_GATE

    # ---- 奖励决策 ----
    # 1) 绝对分位 (跨作者)
    df["pct_global"] = df["reward_score"].rank(pct=True)
    # 2) 作者内分位 (相对自己) —— 主通道
    df["pct_in_author"] = df.groupby("author_id")["reward_score"].rank(pct=True)
    n_a = df.groupby("author_id")["tid"].transform("size")
    df["author_n"] = n_a

    # 作者内比较需要足够历史, 否则退回绝对分位
    MIN_HIST = 5
    df["pct_used"] = np.where(df["author_n"] >= MIN_HIST, df["pct_in_author"], df["pct_global"])
    df["basis"] = np.where(df["author_n"] >= MIN_HIST, "作者内进步", "跨作者分位")

    # 新人加成: 作者历史少 => 用绝对分位但略放宽
    df["tier"] = pd.cut(df["pct_used"], TIER_EDGES, labels=TIER_LABELS)
    df["tier"] = df["tier"].astype(object)
    # 诚信不达标 -> 直接出局(覆盖分位结果)
    df.loc[~df["eligible"], "tier"] = "T0不奖"
    POINTS = {"T0不奖": 0, "T1": 10, "T2": 30, "T3": 60, "T4顶级": 100}
    df["reward_points"] = df["tier"].map(POINTS).astype(int)

    print("=== 标定 ===")
    print(f"  原始标量    min={np.nanmin(raw):+.2f} max={np.nanmax(raw):+.2f} median={med:+.3f}")
    print("  对外分      百分位映射到 [-100,+100] (保留完整排序, 无人工并列)")
    print(f"              p10={df['reward_score'].quantile(.1):+.1f} "
          f"p50={df['reward_score'].median():+.1f} p90={df['reward_score'].quantile(.9):+.1f}")

    print("\n=== 奖励池统计 ===")
    print(f"  总帖数        {len(df):,}")
    print(f"  诚信达标      {df['eligible'].sum():,} ({100*df['eligible'].mean():.1f}%)")
    print(f"  走作者内进步  {(df['basis']=='作者内进步').sum():,}")
    print(f"  走跨作者分位  {(df['basis']=='跨作者分位').sum():,}")
    print("\n=== 分档发放 ===")
    g = df.groupby("tier", observed=False).agg(
        帖数=("tid", "size"), 均回复=("reply_num", "mean"), 均诚信=("integrity", "mean"),
        均奖励=("reward_points", "sum"))
    g["均奖励"] = (g["均奖励"] / g["帖数"].replace(0, np.nan)).round(1)
    print(g.to_string())
    print(f"\n  总发放积分: {df['reward_points'].sum():,}")

    print("\n=== 分年度 (检验是否只奖励老帖) ===")
    ya = df.groupby("year").agg(帖数=("tid", "size"), 均奖励=("reward_points", "mean"),
                                获T3以上占比=("tier", lambda x: (x.isin(["T3", "T4顶级"])).mean()))
    print(ya.tail(8).to_string())

    if a.tid:
        r = df[df["tid"] == a.tid]
        print("\n=== 指定帖 ===")
        for _, x in r.iterrows():
            print(f"  tid={x['tid']} 标题={(str(x['title'])[:40])}")
            print(f"  reward_score={x['reward_score']:+.1f} (raw={x['score_raw']:+.3f})  分位={x['pct_used']:.3f}  "
                  f"档={x['tier']}  积分={x['reward_points']}")
            print(f"  依据={x['basis']}  诚信={x['integrity']:.3f}  "
                  f"实际回复={x['reply_num']} 参与人数={x['n_distinct_authors']}")
    if a.author:
        r = df[df["author_id"] == a.author].sort_values("reward_score", ascending=False)
        print(f"\n=== 作者 {a.author} (共 {len(r)} 帖) ===")
        for _, x in r.head(10).iterrows():
            print(f"  {x['reward_score']:+.3f} {x['tier']:<8} 回复{x['reply_num']:>5}  {(str(x['title'])[:46])}")
    if a.top:
        r = df[df["eligible"]].nlargest(a.top, "reward_score")
        print(f"\n=== Top {a.top} (诚信达标) ===")
        for _, x in r.iterrows():
            print(f"  {x['reward_score']:>+7.3f}分 (raw={x['score_raw']:+.2f}) {x['tier']:<8} "
                  f"回复{x['reply_num']:>5} 人{x['n_distinct_authors']:>4.0f}  {(str(x['title'])[:44])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
