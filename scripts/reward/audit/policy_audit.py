# -*- coding: utf-8 -*-
"""比较候选模型的奖励政策后果，而不是只比较预测指标。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from apps.tieba_reward import _closed_mask, build_design, gate
from resrank import Ranker, allocate


def _rate(mask, denom):
    denom = int(np.asarray(denom, dtype=bool).sum())
    return None if denom == 0 else float(np.asarray(mask, dtype=bool).sum() / denom)


def make_plan(df, model, preview=False):
    """同时支持真实 settled 模式和强制 score 预排名模式。"""
    closed = np.zeros(len(df), dtype=bool) if preview else _closed_mask(df)
    blocked, rankable, integrity, reason = gate(df, settled=closed)
    scores = model.score(df) if preview else np.empty(len(df), dtype=float)
    if not preview:
        if closed.any():
            scores[closed] = model.residual(df.loc[closed])
        if (~closed).any():
            scores[~closed] = model.score(df.loc[~closed])
    week = df.get("week", pd.Series(["__all__"] * len(df), index=df.index))
    parts = []
    for _, g in df.assign(_week=week).groupby("_week", sort=False, dropna=False):
        idx = g.index.to_numpy()
        p = allocate(scores[idx], blocked=blocked[idx], rankable=rankable[idx])
        p.index = idx
        parts.append(p)
    plan = pd.concat(parts).sort_index()
    plan["integrity"] = integrity
    plan["eligible"] = rankable & ~blocked
    plan["eligibility_reason"] = reason
    plan["settled"] = closed
    return scores, plan


def audit(df, model, name):
    future = pd.to_datetime(df["create_time"], utc=True).dt.year.to_numpy() >= 2023
    prior = pd.to_numeric(df["author_prior"], errors="coerce").to_numpy(float)
    finite = np.isfinite(prior)
    q25 = float(np.nanquantile(prior[finite], 0.25))
    q50 = float(np.nanquantile(prior[finite], 0.50))
    q75 = float(np.nanquantile(prior[finite], 0.75))
    # 用秩分桶处理 author_prior 的大量并列值,避免 q25=q50 导致空切片。
    prior_pct = pd.Series(prior).rank(method="average", pct=True).to_numpy().copy()
    prior_pct[~finite] = 0.5
    author_bin = np.select(
        [prior_pct <= 0.25, prior_pct <= 0.50, prior_pct <= 0.75],
        ["Q1_newer", "Q2", "Q3"], default="Q4_established"
    )
    group = df.get("author_id", pd.Series(np.arange(len(df)), index=df.index)).astype("string")
    dup = pd.to_numeric(df.get("dup_count_global", 1), errors="coerce").fillna(1).to_numpy(float)
    low = (
        pd.to_numeric(df.get("len_text", 0), errors="coerce").fillna(0).to_numpy(float) <= 5
    ) & (
        pd.to_numeric(df.get("len_title", 0), errors="coerce").fillna(0).to_numpy(float) <= 3
    )
    media = np.zeros(len(df), bool)
    for c in ("n_image", "n_link", "n_video"):
        if c in df.columns:
            media |= pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(float) > 0
    low &= ~media
    question = pd.to_numeric(df.get("is_question", 0), errors="coerce").fillna(0).to_numpy(float) > 0

    def summarize_plan(plan, label):
        ppoints = plan["points"].to_numpy(float)
        peligible = plan["eligible"].to_numpy(bool)
        def slice_stats(mask):
            m = np.asarray(mask, bool) & future
            return {
                "n": int(m.sum()),
                "eligible_rate": _rate(peligible[m], np.ones(m.sum(), bool)),
                "paid_rate": _rate(ppoints[m] > 0, np.ones(m.sum(), bool)),
                "points": float(ppoints[m].sum()),
            }
        group_points = pd.Series(ppoints[future], index=group[future]).groupby(level=0).sum()
        cutoff = group_points.quantile(0.99) if len(group_points) else 0
        return {
            "mode": label,
            "blocked": int(plan["blocked"].sum()),
            "paid": int((ppoints > 0).sum()),
            "points_total": float(ppoints.sum()),
            "top_1pct_author_points_share": float(group_points[group_points >= cutoff].sum() / group_points.sum()) if group_points.sum() else None,
            "author_slices": {
                label: slice_stats(author_bin == label)
                for label in ("Q1_newer", "Q2", "Q3", "Q4_established")
            },
            "content_slices": {
                "question": slice_stats(question),
                "low_effort": slice_stats(low),
                "duplicate_global_gt_1": slice_stats(dup > 1),
                "duplicate_global_ge_8": slice_stats(dup >= 8),
            },
            "eligibility_reasons": {
                str(k): int(v) for k, v in plan["eligibility_reason"].value_counts().items()
            },
        }

    settled_score, settled_plan = make_plan(df, model, preview=False)
    preview_score, preview_plan = make_plan(df, model, preview=True)
    return {
        "name": name,
        "artifact": getattr(model, "_artifact_path", None),
        "future_rows": int(future.sum()),
        "settled_score_author_prior_corr": float(spearmanr(settled_score[future], prior[future]).statistic),
        "preview_score_author_prior_corr": float(spearmanr(preview_score[future], prior[future]).statistic),
        "settled_plan": summarize_plan(settled_plan, "residual_when_closed"),
        "preview_plan": summarize_plan(preview_plan, "score_when_open"),
        "thresholds": {"author_prior_q25": q25, "q50": q50, "q75": q75},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data" / "reward" / "thread_residual_weekly.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "reward" / "policy_audit.json")
    args = ap.parse_args(argv)
    df = build_design(pd.read_parquet(args.input).reset_index(drop=True))
    models = {}
    for name in ("resrank", "resrank_p1_noauthor", "resrank_p1_author"):
        path = ROOT / "models" / "reward" / f"{name}.pkl"
        if path.exists():
            model = Ranker.load(str(path))
            model._artifact_path = str(path)
            models[name] = audit(df, model, name)
    result = {
        "protocol": {
            "future": "calendar_year >= 2023",
            "cohort": "week",
            "modes": ["residual_when_closed", "score_when_open"],
            "primary_guardrails": ["newer-author share", "top-author concentration", "duplicate/low-effort paid rate", "question slice"],
            "interpretation": "policy audit; not causal evidence that rewards change behavior",
        },
        "models": models,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
