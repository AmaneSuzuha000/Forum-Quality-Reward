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

    def test_open_rows_use_score_only(self):
        df = frame()
        model = FakeRanker()
        scores, plan = plan_dataframe(df, model, cohort="week")
        self.assertEqual(model.calls, [("residual", (0,)), ("score", (1, 3))])
        self.assertTrue(np.allclose(scores, [0.8, 0.4, 0.0, 0.3]))
        self.assertEqual(len(plan), len(df))
        self.assertEqual(plan.loc[2, "eligibility_reason"], "insufficient_settlement_data")
        self.assertEqual(plan.loc[2, "tier"], -1)

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
