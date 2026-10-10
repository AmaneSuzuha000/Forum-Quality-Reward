# -*- coding: utf-8 -*-
"""Ingest independent blind labels and evaluate the frozen candidate.

No labels means waiting_for_labels. Existing platform proxies are never used
as a substitute for human labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / "reports" / "reward" / "annotation"
DEFAULT_KEY = DEFAULT_DIR / "annotation_key.parquet"
DEFAULT_SCORES = ROOT / "data" / "reward" / "reward_scores_weekly.parquet"
DEFAULT_OUT = DEFAULT_DIR / "blind_label_evaluation.json"
ORDINAL = ("informativeness", "originality", "clarity", "actionability", "reward_worthy")
BINARY = ("low_effort", "uncertain")
ALL_LABELS = ORDINAL + BINARY


def _read_labels(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _valid_numeric(values, low, high):
    x = pd.to_numeric(values, errors="coerce")
    return x.notna() & x.between(low, high)


def _agreement(a: pd.Series, b: pd.Series, name: str) -> dict:
    if name in ORDINAL:
        valid = _valid_numeric(a, 1, 5) & _valid_numeric(b, 1, 5)
        weights = "quadratic"
    else:
        valid = _valid_numeric(a, 0, 1) & _valid_numeric(b, 0, 1)
        weights = None
    if int(valid.sum()) < 2:
        return {"n": int(valid.sum()), "kappa": None}
    av = pd.to_numeric(a[valid], errors="coerce").astype(int)
    bv = pd.to_numeric(b[valid], errors="coerce").astype(int)
    observed = float((av.to_numpy() == bv.to_numpy()).mean())
    if np.unique(np.r_[av.to_numpy(), bv.to_numpy()]).size < 2:
        return {"n": int(valid.sum()), "kappa": None, "observed_agreement": observed, "note": "constant_labels_kappa_undefined"}
    return {"n": int(valid.sum()), "kappa": float(cohen_kappa_score(av, bv, weights=weights)), "observed_agreement": observed}


def _quality_eval(score, y):
    score = np.asarray(score, float)
    y = np.asarray(y, float)
    m = np.isfinite(score) & np.isfinite(y)
    if m.sum() < 10 or np.unique(y[m]).size < 2:
        return {"n": int(m.sum()), "positive": int((y[m] >= 4).sum()), "spearman": None, "roc_auc": None, "average_precision": None}
    binary = (y[m] >= 4).astype(int)
    return {"n": int(m.sum()), "positive": int(binary.sum()), "spearman": float(spearmanr(score[m], y[m]).statistic), "roc_auc": float(roc_auc_score(binary, score[m])), "average_precision": float(average_precision_score(binary, score[m]))}


def evaluate(key_path: Path, scores_path: Path, labels_a: Path | None, labels_b: Path | None, out_path: Path) -> dict:
    result = {
        "schema_version": 1,
        "key": str(key_path),
        "scores": str(scores_path),
        "labels_a": str(labels_a) if labels_a else None,
        "labels_b": str(labels_b) if labels_b else None,
        "label_definition": "reward_worthy >= 4 is a secondary binary view; ordinal means remain primary",
        "platform_proxy_substitution": False,
    }
    if labels_a is None or labels_b is None or not labels_a.exists() or not labels_b.exists():
        result.update({"status": "waiting_for_labels", "blocking_reason": "需要两份独立标注文件;不会回退到 is_good", "expected_columns": ["annotation_id", *ALL_LABELS, "reason"]})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
    key = pd.read_parquet(key_path).copy()
    a = _read_labels(labels_a).copy()
    b = _read_labels(labels_b).copy()
    required = {"annotation_id", *ALL_LABELS}
    errors = []
    for label, frame in (("A", a), ("B", b)):
        missing = sorted(required - set(frame.columns))
        if missing:
            errors.append(f"labels_{label}_missing:{','.join(missing)}")
        if frame["annotation_id"].duplicated().any():
            errors.append(f"labels_{label}_duplicate_annotation_id")
    if errors:
        result.update({"status": "blocked_invalid_labels", "errors": errors})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
    merged = key[["annotation_id", "tid", "annotation_split"]].merge(a, on="annotation_id", how="inner", suffixes=("", "_a"))
    merged = merged.merge(b, on="annotation_id", how="inner", suffixes=("_a", "_b"))
    if merged.empty:
        result.update({"status": "blocked_invalid_labels", "errors": ["no_common_annotation_id"]})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
    result["coverage"] = {"common_rows": len(merged), "holdout_rows": int((merged["annotation_split"] == "holdout").sum()), "label_train_rows": int((merged["annotation_split"] == "label_train").sum())}
    result["agreement"] = {name: _agreement(merged[f"{name}_a"], merged[f"{name}_b"], name) for name in (*ORDINAL, *BINARY)}
    holdout = merged[merged["annotation_split"] == "holdout"].copy()
    scores = pd.read_parquet(scores_path).set_index("tid")
    for mode in ("base", "quality", "novelty", "uncertainty", "full"):
        col = f"score_{mode}"
        if col not in scores.columns:
            continue
        holdout[f"_score_{mode}"] = holdout["tid"].map(scores[col])
    result["holdout_evaluation"] = {}
    for name in ORDINAL:
        av = pd.to_numeric(holdout[f"{name}_a"], errors="coerce")
        bv = pd.to_numeric(holdout[f"{name}_b"], errors="coerce")
        holdout[f"{name}_mean"] = pd.concat([av, bv], axis=1).mean(axis=1)
        result["holdout_evaluation"][name] = {mode: _quality_eval(holdout[f"_score_{mode}"], holdout[f"{name}_mean"]) for mode in ("base", "quality", "novelty", "uncertainty", "full") if f"_score_{mode}" in holdout.columns}
    result["status"] = "ready"
    result["interpretation"] = "人工标签 holdout 评估;不得与原 is_good 代理结果混称"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", type=Path, default=DEFAULT_KEY)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--labels-a", type=Path, default=None)
    ap.add_argument("--labels-b", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    result = evaluate(args.key, args.scores, args.labels_a, args.labels_b, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
