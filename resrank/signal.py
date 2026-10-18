# -*- coding: utf-8 -*-
"""L1 目标合成 与 L2 去混杂。

L1  把多路交互计数折成一个标量目标 u
L2  把 u 里「谁发的 / 何时发的 / 什么主题」能解释的部分减掉，得到残差 r

r 是**结算量**：它回答「这篇比同等条件下的平均水平好多少」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(float)


def compose(df: pd.DataFrame, weights: dict[str, float], log1p: bool = True) -> np.ndarray:
    """L1: u = Σ wₖ · f(yₖ)，f 默认 log1p。

    权重可正可负：正 = 鼓励，负 = 惩罚。
    log1p 不是可选项 —— 互动量跨越三个数量级，线性加权会被爆款主导，
    结果变成「谁火奖励谁」而不是「谁写得好奖励谁」。
    """
    u = np.zeros(len(df), dtype=float)
    for col, w in weights.items():
        if col not in df.columns:
            continue
        y = np.clip(_num(df, col), 0.0, None)
        u += w * (np.log1p(y) if log1p else y)
    return u


class Debiaser:
    """L2: r = u − E[u | 上下文]。

    上下文正是**暴露偏差**的来源。不剥掉这层，奖励就变成奖励「已经有的流量」。
    残差还有个副作用是好事：它把「作者历史表现」从目标里剔出去了，
    所以新人不会因为没历史而永远垫底。
    """

    def __init__(self, alpha: float = 5.0):
        self.alpha = alpha

    def fit(self, C: np.ndarray | None, u: np.ndarray) -> Debiaser:
        if C is None or np.asarray(C).size == 0:
            self.model_ = None
            self.mu_ = self.sd_ = None
            return self
        C = np.asarray(C, dtype=float)
        self.mu_ = C.mean(0)
        sd = C.std(0)
        self.sd_ = np.where(sd == 0.0, 1.0, sd)
        Z = (C - self.mu_) / self.sd_
        self.model_ = Ridge(alpha=self.alpha).fit(Z, np.asarray(u, dtype=float))
        return self

    def transform(self, C: np.ndarray | None, u: np.ndarray) -> np.ndarray:
        if getattr(self, "model_", None) is None:
            return np.asarray(u, dtype=float)
        C = np.asarray(C, dtype=float)
        # provisional pickle 没有标准化常数时仍按旧语义读取。
        mu = getattr(self, "mu_", None)
        sd = getattr(self, "sd_", None)
        Z = C if mu is None or sd is None else (C - mu) / sd
        return np.asarray(u, dtype=float) - self.model_.predict(Z)
