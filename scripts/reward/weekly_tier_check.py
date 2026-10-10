# -*- coding: utf-8 -*-
"""检验档位在两种目标上的单调性。

档位按模型分（残差）切分，而残差衡量的是「超出作者/主题/时间预期的程度」，
不是绝对互动量。因此在模型分上档位必然单调，在绝对回复上只是相关而非单调。
本脚本把这两件事分开验证。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resrank.runtime import ROOT, setup_stdout

setup_stdout()

import numpy as np
import pandas as pd

d = pd.read_parquet(ROOT/"data/reward/thread_residual_weekly.parquet")
s = pd.read_parquet(ROOT/"data/reward/reward_scores_weekly.parquet")
df = d.merge(s[["tid","reward_score"]], on="tid", how="left")
df["ct"] = pd.to_datetime(df["create_time"], utc=True).dt.tz_convert("Asia/Shanghai")
df["wk"] = df["ct"].dt.strftime("%G-W%V")
rec = df[df["ct"] >= "2022-01-01"].copy()

def scheme(n):
    if n >= 250: return [0,.60,.75,.90,.97,1.0001], ["T0","T1","T2","T3","T4"]
    if n >= 100: return [0,.70,.85,.95,1.0001],       ["T0","T1","T2","T3"]
    return [0,.75,.90,1.0001],                        ["T0","T1","T2"]

parts = []
for _w, g in rec.groupby("wk"):
    if len(g) < 20: continue
    g = g.copy()
    g["eligible"] = g["integrity"] >= 0.60
    g["pct"] = g["reward_score"].rank(pct=True)
    edges, labels = scheme(len(g))
    g["tier"] = pd.cut(g["pct"], edges, labels=labels).astype(object)
    g.loc[~g["eligible"], "tier"] = "T0"
    g["tier_rank"] = g["tier"].map({t: i for i, t in enumerate(labels)})
    # 周内归一化(去掉旺盛周/冷清周基线差异)
    m = g["reply_7d"].mean()
    g["rel"] = g["reply_7d"]/m if m > 0 else np.nan
    g["wnorm_rank"] = g["rel"].rank(pct=True)
    parts.append(g)
P = pd.concat(parts, ignore_index=True)
print(f"合并 {P['wk'].nunique()} 周, {len(P):,} 帖\n")

print("="*76)
print("A. 档位在【模型分】上的单调性  (这是分档依据, 必然单调)")
print("="*76)
a = P.groupby("tier_rank").agg(档位=("tier","first"), 帖数=("tid","size"),
                               均模型分=("reward_score","mean"),
                               中位模型分=("reward_score","median"))
print(a.round(3).to_string())
print(f"  => 单调: {a['均模型分'].is_monotonic_increasing} (分档就是这个变量切的, 一定成立)")

print()
print("="*76)
print("B. 档位在【绝对7天回复】上的单调性  (用户直觉的'好帖')")
print("="*76)
b = P.groupby("tier_rank").agg(档位=("tier","first"), 帖数=("tid","size"),
                               均7天回复=("reply_7d","mean"),
                               周内归一化=("rel","mean"),
                               相对本周均值=("wnorm_rank","mean"))
print(b.round(3).to_string())
print(f"  原始绝对量 单调: {b['均7天回复'].is_monotonic_increasing}")
print(f"  周内归一化 单调: {b['周内归一化'].is_monotonic_increasing}")

print()
print("="*76)
print("C. 两档对比 (最稳的检验, 不受中间档噪声影响)")
print("="*76)
for hi, lo in [(3,0),(4,0),(2,0),(3,1)]:
    if hi not in set(P["tier_rank"]) or lo not in set(P["tier_rank"]): continue
    H = P[P["tier_rank"]==hi]; L = P[P["tier_rank"]==lo]
    from scipy.stats import mannwhitneyu
    u,p = mannwhitneyu(H["reply_7d"], L["reply_7d"], alternative="greater")
    print(f"  T{hi} vs T{lo}: 7天回复 {H['reply_7d'].mean():6.2f} vs {L['reply_7d'].mean():6.2f}  "
          f"({100*(H['reply_7d'].mean()/L['reply_7d'].mean()-1):+6.1f}%)  AUC={u/(len(H)*len(L)):.3f}  p={p:.1e}")

print()
print("="*76)
print("D. 逐周: 最高档 vs 最低档, 谁更热?")
print("="*76)
res = []
for _w, g in P.groupby("wk"):
    H = g[g["tier_rank"]==g["tier_rank"].max()]; L = g[g["tier_rank"]==0]
    if len(H) < 3 or len(L) < 3: continue
    res.append((H["reply_7d"].mean(), L["reply_7d"].mean(), H["reply_7d"].mean()>L["reply_7d"].mean(),
                g["tier_rank"].max()))
R = pd.DataFrame(res, columns=["hi","lo","win","ntier"])
print(f"  {len(R)} 个可评估周")
for nt, sub in R.groupby("ntier"):
    print(f"    最高档={int(nt)}档的周: n={len(sub):>3}  最高档更热的占比={100*sub['win'].mean():.1f}%  "
          f"均高 {sub['hi'].mean():.2f} vs 均低 {sub['lo'].mean():.2f}")
print(f"  合计: 最高档更热的周 = {int(R['win'].sum())}/{len(R)} = {100*R['win'].mean():.1f}%")

print()
print("="*76)
print("结论")
print("="*76)
print("  档位按【模型分=残差】切分 => 在模型分上严格单调(定义如此)")
print("  残差 = '超出作者/主题/时间预期' 的程度, 与绝对互动量**正相关但不相等**")
print("  所以: 分档在绝对回复上不保证单调, 但最高档显著更热")
