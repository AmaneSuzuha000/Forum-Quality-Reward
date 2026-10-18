# -*- coding: utf-8 -*-
"""端到端示例：不需要数据库、不需要任何外部数据，直接跑。

    python examples/demo_reward_round.py

做的事：造一批「真实质量与作者名气无关、但互动量被名气污染」的帖子，
跑完整四步，打印本周获奖名单，并对比「按互动量直接排」会选出谁。

看完这个脚本就能明白算法在解决什么问题：朴素排序选出来的是一批
已经很有名的作者，ResRank 选出来的是一批真正写得好的帖子。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resrank import Ranker, allocate, pick_scheme, summarize


def make_posts(n=900, n_author=25, seed=7):
    """造一周的帖子。

    三个真值互相独立，方便事后检查算法有没有被带跑：

        q_true      真实质量      —— 算法**应该**学到的东西
        skill_true  作者名气      —— 算法**不该**学到的东西（暴露偏差的来源）
        互动量       = f(质量 + 名气 + 噪声)  —— 能直接观测到的、被污染的信号
    """
    rng = np.random.default_rng(seed)
    author = rng.integers(0, n_author, n)

    # 名气差异刻意做得比质量差异更大：这样「按互动量排」必然被名气带跑
    skill = rng.normal(0.0, 1.5, n_author)[author]
    q = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.4, n)

    # 观测互动 ≈ exp(质量 + 名气 + 噪声)
    engagement = np.exp(q + skill + noise)

    # 帖子正文：用词分布被真实质量平移，好内容用词不一样
    n_vocab = 300
    direction = rng.normal(0, 1, n_vocab)
    logits = q[:, None] * direction[None, :] * 0.9
    logits -= logits.max(1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(1, keepdims=True)
    words = []
    for i in range(n):
        k = int(rng.integers(12, 28))
        words.append(" ".join(f"w{j}" for j in rng.choice(n_vocab, k, p=prob[i])))

    # 一周内的时间戳：时间越靠后，版面越活跃（用来演示「版面活跃度」也是个混杂项）
    day = rng.integers(0, 7, n)
    stamp = pd.Timestamp("2024-06-03", tz="UTC") + pd.to_timedelta(day, unit="D")

    return pd.DataFrame({
        "tid": np.arange(n),
        "create_time": stamp,
        "author_id": author,
        "title": [f"标题 {w[:24]}" for w in words],
        "text": words,
        "reply_7d": engagement,
        # 作者历史水平：真实部署时由 L2 的 as-of 先验给出，这里直接给真值
        "author_prior": skill,
        "author_level": skill * 10,
        "q_true": q,
        "skill_true": skill,
    })


def main() -> int:
    df = make_posts()
    print(f"造出 {len(df)} 篇帖子，{df['author_id'].nunique()} 位作者\n")

    # ---- L1+L2+L3：训练打分器 ----
    r = Ranker(
        objective={"reply_7d": 1.0},
        context=["author_prior", "author_level"],     # 要剥掉的混杂项
        content=["author_level"],                     # 内容侧可用特征
        text=("title", "text"),
        time_col="create_time",
        verbose=False,
    ).fit(df)

    score = r.score(df)          # 只看内容就能算出的预测分
    residual = r.residual(df)    # 需要等窗口关闭才能算出的结算量

    # ---- L4：配额分配 ----
    scheme = pick_scheme(len(df))
    plan = allocate(score, scheme=scheme)
    out = df[["tid", "author_id", "title", "q_true", "skill_true"]].copy()
    out["score"] = score
    out["tier"] = plan["tier"].to_numpy()
    out["points"] = plan["points"].to_numpy()
    out = out.sort_values("score", ascending=False).reset_index(drop=True)

    info = summarize(plan)
    print(f"分档方案：{scheme.n} 档，切点 {scheme.edges}，各档积分 {scheme.points}")
    print(f"队列 {info['n']} 篇 -> 可排名 {info['ranked']} 篇 -> 获奖 {info['paid']} 篇，"
          f"共 {info['points_total']} 分\n")

    print("获奖名单（按预测分排序）：")
    print(f"  {'名次':<4} {'作者':<6} {'积分':<6} {'真质量':<8} {'作者名气':<8} 标题")
    for i, row in out[out["points"] > 0].head(12).iterrows():
        print(f"  {i + 1:<4} {row['author_id']:<6} {int(row['points']):<6} "
              f"{row['q_true']:>+7.2f}  {row['skill_true']:>+7.2f}  {row['title'][:26]}")

    # ---- 关键对比：朴素排序 vs 本算法 ----
    # 同样发 53 个名额，朴素做法是直接按互动量取前 N
    # 注意：out 已经过 sort + reset_index，它的 .index 是**名次**不是行号，
    # 所以只能按 tid 对齐回原表。直接用 .index 去索引 df 会静默取错行。
    k = int((plan["points"] > 0).sum())
    ours_ids = set(out.head(k)["tid"])
    naive_ids = set(df.nlargest(k, "reply_7d")["tid"])
    ours_top = df[df["tid"].isin(ours_ids)]
    naive_top = df[df["tid"].isin(naive_ids)]

    print(f"\n与「直接按互动量取前 {k} 名」对比：")
    print(f"  获奖者平均真质量    朴素 {naive_top['q_true'].mean():+.3f}"
          f"   本算法 {ours_top['q_true'].mean():+.3f}")
    print(f"  获奖者平均作者名气  朴素 {naive_top['skill_true'].mean():+.3f}"
          f"   本算法 {ours_top['skill_true'].mean():+.3f}")
    overlap = len(ours_ids & naive_ids)
    print(f"  两份名单重合 {overlap}/{k} 篇 —— 不重合的部分就是被名气挤掉的好帖")

    # ---- 自检：分数应该贴质量、不贴名气 ----
    from scipy.stats import spearmanr
    rho_q = spearmanr(score, df["q_true"]).statistic
    rho_s = spearmanr(score, df["skill_true"]).statistic
    rho_r = spearmanr(residual, df["q_true"]).statistic
    print(f"\n  rho(预测分, 真质量) = {rho_q:+.4f}   <- L3 蒸馏结果，冷启动即可用")
    print(f"  rho(残差,   真质量) = {rho_r:+.4f}   <- L1+L2 结算量，需等窗口关闭")
    print(f"  rho(预测分, 作者名气) = {rho_s:+.4f}   <- 低是对的")
    assert rho_q > rho_s, "预测分应该更贴质量而不是名气"
    assert rho_q > 0.5, "预测分应与真质量强相关"
    # L3 的意义：把结算量压进内容特征后，几乎不损失排序能力
    assert rho_q > rho_r - 0.05, "蒸馏不该显著损失排序能力"
    print("\n自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
