# -*- coding: utf-8 -*-
"""L4 配额分配 —— 队列内分位 → 档位 → 积分。

两个设计选择：

**分位而不是绝对阈值。** 奖励总额自动跟随队列规模。用绝对阈值的话，
发帖量翻倍的季度会直接把预算打爆，而且阈值要年年重调。

**闸门先于排名。** 重复顶帖、零字水帖这类不该参与竞争的对象先被拦下，
不占分位、不得分。它们仍然出现在名单里（可申诉），只是标为 blocked。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Scheme, pick_scheme


def allocate(scores, scheme: Scheme | None = None, blocked=None,
             rankable=None, extra=None, points_col: str = "points") -> pd.DataFrame:
    """把分数变成档位与积分。

    scores    打分，越大越好
    scheme    分档方案；None = 按可排名队列规模自动选
    blocked   布尔，True = 被闸门拦下（0 分，不参与分位）
    rankable  布尔，True = 有资格参与排名；None = 未被拦下的全部有资格
    extra     额外积分（如全勤奖），只在得分 > 0 时叠加

    返回 DataFrame: score / pct / tier / points / blocked
    tier = -1 表示未参与排名（拦下或数据不足）
    """
    s = np.asarray(scores, dtype=float)
    n = len(s)

    blk = np.zeros(n, dtype=bool) if blocked is None else np.asarray(blocked, dtype=bool)
    rk = (~blk) if rankable is None else (np.asarray(rankable, dtype=bool) & ~blk)

    scheme = scheme or pick_scheme(int(rk.sum()))

    pct = np.full(n, np.nan)
    tier = np.full(n, -1, dtype=int)
    if rk.any():
        v = s[rk]
        # 平均秩百分位 —— 并列同分给同一个分位，避免名次被人为拉开
        pct[rk] = pd.Series(v).rank(pct=True, method="average").to_numpy()
        tier[rk] = scheme.tier(pct[rk])

    pts = np.zeros(n, dtype=float)
    # tier=0 是「未获奖」档，不是「得 0 分的奖」——extra 不能加给它。
    ranked = tier >= 0
    paid = tier > 0
    if ranked.any():
        pts[ranked] = np.asarray(scheme.points, dtype=float)[tier[ranked]]
    if extra is not None:
        pts = pts + np.where(paid, np.asarray(extra, dtype=float), 0.0)
    pts[blk] = 0.0

    return pd.DataFrame({
        "score": s, "pct": pct, "tier": tier,
        points_col: pts, "blocked": blk,
    })


def summarize(plan: pd.DataFrame, points_col: str = "points") -> dict:
    """一眼看清这次分配：多少人、多少人拿钱、总共多少分。"""
    t = plan["tier"].to_numpy()
    return {
        "n": len(plan),
        "blocked": int(plan["blocked"].sum()),
        "ranked": int((t >= 0).sum()),
        "paid": int((t > 0).sum()),
        "points_total": float(plan[points_col].sum()),
        "tiers": {int(k): int(v) for k, v in pd.Series(t[t >= 0]).value_counts().sort_index().items()},
    }
