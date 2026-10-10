# -*- coding: utf-8 -*-
"""检验模型在小队列上的有效性。

两个不同的目标必须分开测:
  (A) 残差排名 —— 模型实际学的目标（超出作者/主题/时间预期的程度），公平但非绝对质量
  (B) 绝对 7 天回复 —— 直觉上的「好帖」
若 (A) 有效而 (B) 无效，说明模型在做「相对进步」评估，而不是预测绝对热度。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resrank.runtime import ROOT, setup_stdout

setup_stdout()

import pandas as pd
from scipy.stats import binomtest, spearmanr

d = pd.read_parquet(ROOT/"data/reward/thread_residual_weekly.parquet")
s = pd.read_parquet(ROOT/"data/reward/reward_scores_weekly.parquet")
df = d.merge(s[["tid","reward_score"]], on="tid", how="left")
df["ct"] = pd.to_datetime(df["create_time"], utc=True).dt.tz_convert("Asia/Shanghai")
df["wk"] = df["ct"].dt.strftime("%G-W%V")

# 只看统计上可用的周 + 近年(与部署期相近)
recent = df[df["ct"] >= "2022-01-01"].copy()
print(f"近年帖 {len(recent):,} 覆盖 {recent['wk'].nunique()} 周")

def eval_cohort(g, label):
    """对单周队列算 (A)(B) 的秩相关与 top/bottom 对比"""
    if len(g) < 40: return None
    rho_a,_ = spearmanr(g["reward_score"], g["residual"])
    rho_b,_ = spearmanr(g["reward_score"], g["reply_7d"])
    k = max(1, int(len(g)*0.30))
    t = g.nlargest(k, "reward_score"); b = g.nsmallest(k, "reward_score")
    return {"n": len(g), "rho_res": rho_a, "rho_rep": rho_b,
                "top_rep": t["reply_7d"].mean(), "bot_rep": b["reply_7d"].mean(),
                "top_res": t["residual"].mean(), "bot_res": b["residual"].mean(),
                "top_gt_bot": t["reply_7d"].mean() > b["reply_7d"].mean()}

# ---- 单周队列 ----
rows = [eval_cohort(g, w) for w, g in recent.groupby("wk")]
rows = [r for r in rows if r]
R = pd.DataFrame(rows)
print(f"\n{'='*74}\n单周队列 (n={len(R)} 周, 中位 {R['n'].median():.0f} 帖/周)\n{'='*74}")
print(f"  (A) 与残差秩相关  mean={R['rho_res'].mean():+.4f}  "
      f"正的周占比={100*(R['rho_res']>0).mean():.1f}%")
print(f"  (B) 与绝对回复秩相关 mean={R['rho_rep'].mean():+.4f}  "
      f"正的周占比={100*(R['rho_rep']>0).mean():.1f}%")
print(f"\n  top30% vs bottom30% 的绝对7天回复: {R['top_rep'].mean():.2f} vs {R['bot_rep'].mean():.2f}")
bt = binomtest(int(R["top_gt_bot"].sum()), len(R), 0.5)
print(f"    top>bot 的周占比 = {100*R['top_gt_bot'].mean():.1f}%  (二项检验 p={bt.pvalue:.3f})")
print(f"  top30% vs bottom30% 的残差:       {R['top_res'].mean():+.2f} vs {R['bot_res'].mean():+.2f}")

# ---- 多周合并队列 ----
print(f"\n{'='*74}\n多周合并队列 (解决小样本)\n{'='*74}")
print(f"  {'窗口':>10} {'队列中位':>9} {'n(残差)':>9} {'n(绝对)':>9} {'top-bot绝对':>12} {'top-bot残差':>12}")
for weeks_per in (1, 2, 4, 8, 13, 26):
    rec = []
    wk_sorted = sorted(recent["wk"].unique())
    for i in range(0, len(wk_sorted), weeks_per):
        chunk = wk_sorted[i:i+weeks_per]
        g = recent[recent["wk"].isin(chunk)]
        r = eval_cohort(g, f"{weeks_per}w")
        if r: rec.append(r)
    if not rec: continue
    X = pd.DataFrame(rec)
    print(f"  {weeks_per:>8}w {X['n'].median():>9.0f} {X['rho_res'].mean():>+9.4f} "
          f"{X['rho_rep'].mean():>+9.4f} {X['top_rep'].mean()-X['bot_rep'].mean():>+12.2f} "
          f"{X['top_res'].mean()-X['bot_res'].mean():>+12.2f}")

# ---- 模型评分是否优于"随机选" ? ----
print(f"\n{'='*74}\n与随机基线对比 (单周, top30% 的绝对7天回复)\n{'='*74}")
obs = R["top_rep"].mean(); weekmed = recent.groupby("wk")["reply_7d"].mean()
print(f"  top30% 平均7天回复 = {obs:.2f}")
print(f"  全体周均          = {recent['reply_7d'].mean():.2f}")
print(f"  => 相对全体提升 {100*(obs/recent['reply_7d'].mean()-1):+.1f}%")
print("\n  (说明: 发奖时刻**看不到** reply_7d, 所以'随机选'的期望就是全体周均)")

# ---- 时间漂移对固定分数线的影响 ----
print(f"\n{'='*74}\n队列内 vs 全库 百分位 (为什么必须用队列内)\n{'='*74}")
recent["pct_global"] = recent["reward_score"].rank(pct=True)
recent["pct_week"]   = recent.groupby("wk")["reward_score"].rank(pct=True)
print(f"  全库百分位 vs 队列内百分位的秩相关 = {spearmanr(recent['pct_global'], recent['pct_week']).statistic:.4f}")
print(f"  若用全库 90% 分位当分数线: 通过率={(recent['pct_global']>=0.90).mean()*100:.1f}%")
print(f"  逐周通过率极差: "
      f"{(recent.groupby('wk').apply(lambda g:(g['pct_global']>=0.90).mean(), include_groups=False).max()*100):.0f}% ~ "
      f"{(recent.groupby('wk').apply(lambda g:(g['pct_global']>=0.90).mean(), include_groups=False).min()*100):.0f}%")
jg = recent.groupby("wk").apply(lambda g:(g['pct_global']>=0.90).mean(), include_groups=False)
print(f"  逐周通过率 标准差 = {jg.std()*100:.1f} 个百分点  <- 波动越大越说明全库分位不可用")
