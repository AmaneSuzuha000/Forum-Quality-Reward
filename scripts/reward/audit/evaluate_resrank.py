# -*- coding: utf-8 -*-
"""可信的 ResRank 评测入口。

主结果使用未来全量帖子和按周宏平均；pairs 只作为 residual-gap 诊断。
所有模型都在同一宽表、同一文本契约、同一队列上比较。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "reward"
REPORT = ROOT / "reports" / "reward"
sys.path.insert(0, str(ROOT))

from apps.tieba_reward import build_design, gate
from resrank import Ranker, compose


def _metric_corr(a, b, mask):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return None
    return float(spearmanr(a[m], b[m]).statistic)


def _top_lift(score, target, mask, q=0.10):
    score = np.asarray(score, dtype=float)
    target = np.asarray(target, dtype=float)
    m = np.asarray(mask, dtype=bool) & np.isfinite(score) & np.isfinite(target)
    if m.sum() < 3:
        return None
    k = max(1, int(np.ceil(m.sum() * q)))
    idx = np.flatnonzero(m)
    top = idx[np.argsort(score[idx])[-k:]]
    return float(np.mean(target[top]) - np.mean(target[idx]))


def _macro_week(score, target, week, mask):
    rows = []
    for key in pd.Series(week[mask]).dropna().unique():
        m = np.asarray(mask, dtype=bool) & (week == key)
        corr = _metric_corr(score, target, m)
        lift = _top_lift(score, target, m)
        if corr is not None:
            rows.append((corr, lift, int(m.sum())))
    if not rows:
        return {"n_weeks": 0, "spearman": None, "top10_lift": None}
    return {
        "n_weeks": len(rows),
        "spearman": float(np.mean([x[0] for x in rows])),
        "top10_lift": float(np.mean([x[1] for x in rows if x[1] is not None])),
        "min_week_n": int(min(x[2] for x in rows)),
    }


def _cluster_bootstrap(values, clusters, rounds=2000, seed=17):
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    codes, keys = pd.factorize(clusters, sort=False)
    if len(keys) < 2:
        return [float(np.mean(values))]
    groups = [np.flatnonzero(codes == k) for k in range(len(keys))]
    rng = np.random.default_rng(seed)
    out = np.empty(rounds, dtype=float)
    for i in range(rounds):
        chosen = rng.integers(0, len(groups), len(groups))
        out[i] = np.concatenate([values[groups[j]] for j in chosen]).mean()
    return out.tolist()


def _cluster_metric_bootstrap(score, target, clusters, mask, rounds=1000, seed=17):
    score = np.asarray(score, dtype=float)
    target = np.asarray(target, dtype=float)
    clusters = np.asarray(clusters)
    idx = np.flatnonzero(np.asarray(mask, dtype=bool) & np.isfinite(score) & np.isfinite(target))
    if len(idx) < 3:
        return {"spearman_ci95": None, "top10_lift_ci95": None}
    codes, keys = pd.factorize(clusters[idx], sort=False)
    if len(keys) < 2:
        return {"spearman_ci95": None, "top10_lift_ci95": None}
    groups = [idx[np.flatnonzero(codes == k)] for k in range(len(keys))]
    rng = np.random.default_rng(seed)
    rho, lift = [], []
    for _ in range(rounds):
        chosen = rng.integers(0, len(groups), len(groups))
        sample = np.concatenate([groups[j] for j in chosen])
        rho.append(_metric_corr(score[sample], target[sample], np.ones(len(sample), dtype=bool)))
        lift.append(_top_lift(score[sample], target[sample], np.ones(len(sample), dtype=bool)))
    def interval(values):
        finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
        return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))] if len(finite) else None
    return {"spearman_ci95": interval(rho), "top10_lift_ci95": interval(lift)}


def _pair_data(df, path):
    if not path.exists():
        return None
    p = pd.read_parquet(path).reset_index(drop=True)
    row = {int(t): i for i, t in enumerate(df["tid"])}
    year = pd.to_datetime(df["create_time"], utc=True).dt.year.to_numpy()
    week = df.get("week", pd.Series([None] * len(df))).astype("string").to_numpy()
    p = p[p["tid_a"].isin(row) & p["tid_b"].isin(row)].copy()
    ya = p["tid_a"].map({int(t): int(year[i]) for i, t in enumerate(df["tid"])}).to_numpy()
    yb = p["tid_b"].map({int(t): int(year[i]) for i, t in enumerate(df["tid"])}).to_numpy()
    same = ya == yb
    p = p.loc[same].reset_index(drop=True)
    ya = ya[same]
    ia = p["tid_a"].map(row).to_numpy(int)
    ib = p["tid_b"].map(row).to_numpy(int)
    clusters = week[ia]
    return {"p": p, "year": ya, "ia": ia, "ib": ib, "clusters": clusters}


def _pair_metrics(score, pairs):
    if pairs is None:
        return {}
    ia, ib, year = pairs["ia"], pairs["ib"], pairs["year"]
    delta = score[ia] - score[ib]
    win = np.where(delta > 0, 1.0, np.where(delta < 0, 0.0, 0.5))
    out = {"n": len(win)}
    for label, m in [("2023", year == 2023), ("2024_plus", year >= 2024), ("all", np.ones(len(year), bool))]:
        if m.any():
            vals = win[m]
            bs = _cluster_bootstrap(vals, pairs["clusters"][m], rounds=2000, seed=17)
            out[label] = {
                "n": int(m.sum()),
                "accuracy": float(vals.mean()),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            }
    return out


def evaluate(model_paths, input_path, output_path):
    df = build_design(pd.read_parquet(input_path).reset_index(drop=True))
    year = pd.to_datetime(df["create_time"], utc=True).dt.year.to_numpy()
    week = df.get("week", pd.Series([None] * len(df))).astype("string").to_numpy()
    validation = year == 2023
    locked_test = year >= 2024
    pairs = _pair_data(df, DATA / "pairs_train_weekly.parquet")
    blocked, rankable, integrity, reason = gate(
        df, settled=np.ones(len(df), dtype=bool)
    )
    # 这是观测数据完整时的政策闸门统计，不把它当模型质量指标。
    policy = {
        "n": len(df),
        "blocked": int(blocked.sum()),
        "blocked_rate": float(blocked.mean()),
        "rankable": int(rankable.sum()),
        "reason_counts": {str(k): int(v) for k, v in pd.Series(reason).value_counts().items()},
        "integrity_mean": float(np.mean(integrity)),
    }
    result = {
        "protocol": {
            "input": str(input_path),
            "n_rows": len(df),
            "train_cutoff": "2022-12-31",
            "selection_validation": "calendar_year == 2023 (used for configuration selection)",
            "locked_test": "calendar_year >= 2024 (not used for selection)",
            "primary": "locked-test full-row Spearman, top10 residual lift, macro-week metrics and cluster bootstrap CIs",
            "secondary": "same-year residual-gap pair diagnostic",
            "bootstrap": "week-cluster bootstrap, 2000 rounds, seed 17",
        },
        "policy": policy,
        "models": {},
    }
    utility = compose(df, {
        "reply_7d": 0.60, "comment_7d": 0.30, "breadth_7d": 0.30,
        "actdays_7d": 0.18, "self_long_uniq": 1.15,
        "self_deep_uniq": 0.80, "self_bump": -0.60,
    })
    for label, path in model_paths.items():
        model = Ranker.load(str(path))
        score = model.score(df)
        residual = model.residual(df)
        model_result = {
            "artifact": str(path),
            "artifact_bytes": int(path.stat().st_size),
            "diag": getattr(model, "diag_", {}),
            "selection_validation_2023": {
                "n": int(validation.sum()),
                "score_vs_settled_residual": {
                    "spearman": _metric_corr(score, residual, validation),
                    "top10_lift": _top_lift(score, residual, validation),
                },
            },
            "locked_test_2024_plus": {
                "n": int(locked_test.sum()),
                "score_vs_settled_residual": {
                    "spearman": _metric_corr(score, residual, locked_test),
                    "top10_lift": _top_lift(score, residual, locked_test),
                    "macro_week": _macro_week(score, residual, week, locked_test),
                    "week_cluster_bootstrap": _cluster_metric_bootstrap(score, residual, week, locked_test, seed=17),
                    "author_cluster_bootstrap": _cluster_metric_bootstrap(score, residual, df["author_id"], locked_test, seed=29)
                        if "author_id" in df.columns else None,
                },
                "score_vs_raw_utility": {
                    "spearman": _metric_corr(score, utility, locked_test),
                    "top10_lift": _top_lift(score, utility, locked_test),
                },
                "score_vs_author_prior": _metric_corr(score, df["author_prior"], locked_test)
                    if "author_prior" in df.columns else None,
            },
            "pair_diagnostic": _pair_metrics(score, pairs),
        }
        result["models"][label] = model_result
    if pairs is not None:
        result["pairs"] = {"n": len(pairs["year"]), "years": {
            str(y): int((pairs["year"] == y).sum())
            for y in sorted(set(pairs["year"]))
        }}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DATA / "thread_residual_weekly.parquet")
    ap.add_argument("--old", type=Path, default=ROOT / "models" / "reward" / "resrank_v0_provisional.pkl")
    ap.add_argument("--new", type=Path, default=ROOT / "models" / "reward" / "resrank_p0.pkl")
    ap.add_argument("--out", type=Path, default=REPORT / "resrank_p1_eval.json")
    args = ap.parse_args(argv)
    result = evaluate({args.old.stem: args.old, args.new.stem: args.new}, args.input, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nWROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
