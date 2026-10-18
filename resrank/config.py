# -*- coding: utf-8 -*-
"""分档方案 —— 队列规模决定档位粒度，档位边界是队列内分位点。

用分位而不是绝对阈值：奖励总额自动跟随队列规模，
不需要每年重调预算，也不会因为某个季度发帖量暴涨而超支。

**预算是常数，粒度才是变量。** 三套方案的每档积分长期不变，
边界则被反解成「让每帖期望积分都等于同一个预算」——
否则 n=249 与 n=250 之间会跳出一倍以上的发放强度，
而队列规模差一帖本身不该改变任何人的奖金。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 每帖期望积分的目标值。三套方案都对齐到它（允许 ±5% 取整误差）。
BUDGET_PER_POST = 5.75


@dataclass(frozen=True)
class Scheme:
    """一套分档方案。

    edges   分位边界，单调递增，从 0.0 到 1.0；档位数 = len(edges) - 1
    points  每档积分，长度与档位数相同；第 0 档恒为 0 分（不获奖）
    """

    edges: tuple
    points: tuple

    @property
    def n(self) -> int:
        return len(self.edges) - 1

    @property
    def award(self) -> float:
        """获奖比例 = 得分 > 0 的那部分。"""
        return 1.0 - self.edges[1]

    @property
    def budget(self) -> float:
        """每帖期望积分 = Σ 档位宽度 × 该档积分。

        这是「队列规模变了，预算却该不变」的直接度量。
        """
        total = 0.0
        for i, p in enumerate(self.points):
            if p:
                total += (self.edges[i + 1] - self.edges[i]) * p
        return total

    def tier(self, q) -> np.ndarray:
        """分位 → 档位序号（0 = 未获奖）。"""
        return np.searchsorted(np.asarray(self.edges[1:-1]), np.asarray(q), side="right")


# 边界由 BUDGET_PER_POST 反解：先定每档积分（对外承诺，长期不变），
# 再让获奖比例满足预算。改积分请同时重算边界，否则预算会漂。
SCHEMES: dict[int, Scheme] = {
    3: Scheme((0.00, 0.70, 0.85, 1.00), (0, 10, 30)),                  # 预算 6.00
    4: Scheme((0.00, 0.70, 0.85, 0.95, 1.00), (0, 5, 20, 60)),         # 预算 5.75
    5: Scheme((0.00, 0.80, 0.90, 0.96, 0.99, 1.00), (0, 10, 30, 60, 100)),  # 预算 5.60
}

# 队列小于这个数时，分位没有统计意义，发奖等于随机撒钱 —— 不结算，顺延。
MIN_COHORT = 30


def pick_scheme(n: int) -> Scheme:
    """队列越大，档位越细 —— 小队列分太细会人人有奖，失去区分度。"""
    if n >= 250:
        return SCHEMES[5]
    if n >= 100:
        return SCHEMES[4]
    return SCHEMES[3]
