# -*- coding: utf-8 -*-
"""Tieba application contract tests; no model artifact or database required."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from apps.tieba_reward import _closed_mask, gate, plan_dataframe


class FakeRanker:
    def __init__(self):
        self.calls = []

    def score(self, df):
        self.calls.append(("score", tuple(df.index)))
        return df["preview_value"].to_numpy(float)

    def residual(self, df):
        self.calls.append(("residual", tuple(df.index)))
        return df["settled_value"].to_numpy(float)


def frame():
    return pd.DataFrame({
        "tid": [1, 2, 3, 4],
        "title": ["a", "b", "c", "d"],
        "text": ["long enough", "long enough", "long enough", "long enough"],
        "create_time": pd.to_datetime(["2023-01-01"] * 4, utc=True),
        "window_complete": [True, False, True, False],
        "week": ["2023-W01", "2023-W01", np.nan, np.nan],
        "reply_7d": [2, np.nan, 1, np.nan],
        "comment_7d": [1, np.nan, np.nan, np.nan],
        "breadth_7d": [1, np.nan, 1, np.nan],
        "actdays_7d": [1, np.nan, 1, np.nan],
        "self_long_uniq": [1, np.nan, 0, np.nan],
        "self_deep_uniq": [0, np.nan, 0, np.nan],
        "self_bump": [0, np.nan, 0, np.nan],
        "preview_value": [0.1, 0.4, 0.2, 0.3],
        "settled_value": [0.8, 0.0, 0.7, 0.0],
        "len_title": [10, 10, 10, 10],
        "len_text": [20, 20, 20, 20],
        "dup_count_global": [1, 1, 1, 1],
    })


class TiebaContractTest(unittest.TestCase):
    def test_status_is_required(self):
        with self.assertRaisesRegex(ValueError, "window_complete"):
            _closed_mask(pd.DataFrame({"tid": [1]}))

    def test_closed_missing_outcome_is_insufficient_not_blocked(self):
        df = frame()
        blocked, rankable, _, reason = gate(df, settled=_closed_mask(df))
        self.assertFalse(blocked[2])
        self.assertFalse(rankable[2])
        self.assertEqual(reason[2], "insufficient_settlement_data")
        self.assertTrue(rankable[1])
        self.assertEqual(reason[1], "eligible")

    def test_open_rows_are_deferred_not_mixed(self):
        """未闭合行必须顺延，不得与已结算行同池排序。

        混用 residual() 与 score() 不是精度差异而是两个不同的排序：
        真实数据上两者全局 Spearman 仅 0.2752，前 1% 名单重合 16.6%。
        所以默认口径是只结算已闭合窗口。
        """
        df = frame()
        model = FakeRanker()
        scores, plan = plan_dataframe(df, model, cohort="week")
        # 只有窗口闭合的行会被取 residual；未闭合行不调用 score
        self.assertEqual(model.calls, [("residual", (0,))])
        self.assertTrue(np.isnan(scores[1]) and np.isnan(scores[3]))
        self.assertEqual(len(plan), len(df))
        # 未闭合行被标记 deferred，不排名、不发分
        self.assertTrue(plan["deferred"].to_numpy()[1])
        self.assertTrue(plan["deferred"].to_numpy()[3])
        self.assertEqual(plan.loc[1, "eligibility_reason"], "deferred_window_open")
        self.assertEqual(plan.loc[1, "tier"], -1)
        self.assertEqual(plan.loc[1, "points"], 0)

    def test_allow_open_is_opt_in_and_flagged(self):
        """显式要求混排时才混排，且必须打上警告标记。"""
        df = frame()
        model = FakeRanker()
        _, plan = plan_dataframe(df, model, cohort="week", allow_open=True)
        self.assertIn("mixed_scale_warning", plan.columns)
        self.assertFalse(plan["deferred"].to_numpy()[1])

    def test_preview_does_not_require_settlement_columns(self):
        df = frame().drop(columns=[
            "reply_7d", "comment_7d", "breadth_7d", "actdays_7d",
            "self_long_uniq", "self_deep_uniq", "self_bump", "window_complete",
        ])
        model = FakeRanker()
        scores, plan = plan_dataframe(df, model, cohort="week", preview=True)
        self.assertEqual(model.calls, [("score", (0, 1, 2, 3))])
        self.assertEqual(len(plan), len(df))
        self.assertTrue(np.allclose(scores, df["preview_value"]))

    def test_nan_cohort_is_retained(self):
        df = frame()
        model = FakeRanker()
        _, plan = plan_dataframe(df, model, cohort="week")
        self.assertEqual(set(plan.index), set(df.index))
        self.assertEqual(len(plan), 4)


if __name__ == "__main__":
    unittest.main()
