# -*- coding: utf-8 -*-
"""分档方案 —— 队列规模决定档位粒度，档位边界是队列内分位点。

用分位而不是绝对阈值：奖励总额自动跟随队列规模，
不需要每年重调预算，也不会因为某个季度发帖量暴涨而超支。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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

    def tier(self, q) -> np.ndarray:
        """分位 → 档位序号（0 = 未获奖）。"""
        return np.searchsorted(np.asarray(self.edges[1:-1]), np.asarray(q), side="right")


SCHEMES: dict[int, Scheme] = {
    3: Scheme((0.00, 0.75, 0.90, 1.00), (0, 10, 30)),
    4: Scheme((0.00, 0.70, 0.85, 0.95, 1.00), (0, 5, 20, 60)),
    5: Scheme((0.00, 0.60, 0.75, 0.90, 0.97, 1.00), (0, 10, 30, 60, 100)),
}


def pick_scheme(n: int) -> Scheme:
    """队列越大，档位越细 —— 小队列分太细会人人有奖，失去区分度。"""
    if n >= 250:
        return SCHEMES[5]
    if n >= 100:
        return SCHEMES[4]
    return SCHEMES[3]
