# -*- coding: utf-8 -*-
"""Build a blinded, reproducible quality-label annotation packet.

The visible queue deliberately excludes author, engagement, platform labels,
settled outcomes, residuals, and reward scores.  The private key preserves
those fields for post-label evaluation; it must not be given to annotators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "data" / "reward" / "thread_residual_weekly.parquet"
DEFAULT_SCORES = ROOT / "data" / "reward" / "reward_scores_weekly.parquet"
DEFAULT_OUT = ROOT / "reports" / "reward" / "annotation"
SEED = 17
TARGET = 2400


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _media(df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=bool)
    for c in ("n_image", "n_link", "n_video"):
        if c in df.columns:
            out |= pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(float) > 0
    return out


def _quantile_bucket(values, labels):
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    rank = x.rank(method="average", pct=True).fillna(0.5).to_numpy(float)
    return np.select(
        [rank <= 0.2, rank <= 0.4, rank <= 0.6, rank <= 0.8],
        labels[:4],
        default=labels[4],
    )


def _year_bucket(year):
    year = np.asarray(year, dtype=int)
    return np.select(
        [year <= 2019, year <= 2022, year == 2023, year == 2024],
        ["early", "pre_lock", "selection_2023", "locked_2024"],
        default="locked_2025_plus",
    )


def _allocate_quota(frame: pd.DataFrame, group_col: str, n: int, rng) -> np.ndarray:
    groups = list(frame.groupby(group_col, sort=True, dropna=False).groups.items())
    if not groups or n <= 0:
        return np.array([], dtype=int)
    sizes = np.asarray([len(idx) for _, idx in groups], dtype=float)
    weights = np.sqrt(sizes)
    weights /= weights.sum()
    quotas = np.floor(weights * n).astype(int)
    quotas = np.minimum(quotas, sizes.astype(int))
    for i in np.argsort(-(weights * n - quotas)):
        if quotas.sum() >= min(n, int(sizes.sum())):
            break
        if quotas[i] < sizes[i]:
            quotas[i] += 1
    chosen = []
    for (_, idx), q in zip(groups, quotas, strict=False):
        idx = np.asarray(idx, dtype=int)
        if q:
            chosen.extend(rng.choice(idx, size=int(q), replace=False).tolist())
    return np.asarray(chosen, dtype=int)


def build_packet(input_path: Path, score_path: Path, out_dir: Path, target: int = TARGET, seed: int = SEED):
    df = pd.read_parquet(input_path).reset_index(drop=True)
    if "tid" not in df.columns or "title" not in df.columns or "text" not in df.columns:
        raise ValueError("输入必须包含 tid、title、text")
    score = pd.read_parquet(score_path).reset_index(drop=True) if score_path.exists() else pd.DataFrame()
    if len(score) and "score_full" in score.columns:
        score_by_tid = score.set_index("tid")["score_full"]
        model_score = df["tid"].map(score_by_tid).to_numpy(float)
    else:
        model_score = np.zeros(len(df), dtype=float)

    create = pd.to_datetime(df["create_time"], utc=True, errors="coerce")
    year = create.dt.year.fillna(0).to_numpy(int)
    media = _media(df)
    question = pd.to_numeric(df.get("is_question", 0), errors="coerce").fillna(0).to_numpy(float) > 0
    text_len = pd.to_numeric(df.get("len_text", 0), errors="coerce").fillna(0).to_numpy(float)
    title_len = pd.to_numeric(df.get("len_title", 0), errors="coerce").fillna(0).to_numpy(float)
    dup = pd.to_numeric(df.get("dup_count_global", 1), errors="coerce").fillna(1).to_numpy(float)
    author_prior = pd.to_numeric(df.get("author_prior", np.nan), errors="coerce").to_numpy(float)
    score_bucket = _quantile_bucket(model_score, ["score_q1", "score_q2", "score_q3", "score_q4", "score_q5"])
    prior_bucket = _quantile_bucket(author_prior, ["prior_q1", "prior_q2", "prior_q3", "prior_q4", "prior_q5"])
    year_bucket = _year_bucket(year)
    dup_bucket = np.select([dup <= 1, dup <= 3, dup <= 7], ["dup_1", "dup_2_3", "dup_4_7"], default="dup_ge8")
    low_effort = (text_len <= 5) & (title_len <= 3) & ~media

    work = pd.DataFrame({
        "row": np.arange(len(df)),
        "year_bucket": year_bucket,
        "score_bucket": score_bucket,
        "prior_bucket": prior_bucket,
        "dup_bucket": dup_bucket,
        "question": np.where(question, "question", "non_question"),
        "media": np.where(media, "media", "text_only"),
    })
    work["core_stratum"] = work["year_bucket"] + "|" + work["score_bucket"]
    rng = np.random.default_rng(seed)
    selected = set()
    # Main coverage: time x candidate-score strata.
    for i in _allocate_quota(work, "core_stratum", min(1200, target), rng):
        selected.add(int(i))
    # Deliberate safety slices, selected without exposing their reason.
    slices = [
        ("question_slice", question, 240),
        ("duplicate_slice", dup >= 2, 240),
        ("high_duplicate_slice", dup >= 8, 160),
        ("short_low_effort_slice", low_effort, 200),
        ("media_slice", media, 240),
        ("newer_author_slice", prior_bucket == "prior_q1", 160),
    ]
    for _, mask, quota in slices:
        candidates = np.flatnonzero(mask & ~np.isin(np.arange(len(df)), list(selected)))
        if len(candidates):
            selected.update(rng.choice(candidates, size=min(quota, len(candidates)), replace=False).tolist())
    # Fill the remaining budget with a reproducible population-stratified draw.
    remaining = target - len(selected)
    if remaining > 0:
        rest = work.loc[~work["row"].isin(selected)].copy()
        for i in _allocate_quota(rest, "year_bucket", remaining, rng):
            selected.add(int(i))
    selected = np.asarray(sorted(selected), dtype=int)
    if len(selected) > target:
        selected = np.sort(rng.choice(selected, size=target, replace=False))
    if len(selected) < target:
        pool = np.setdiff1d(np.arange(len(df)), selected)
        selected = np.sort(np.r_[selected, rng.choice(pool, size=min(target - len(selected), len(pool)), replace=False)])

    annotation_id = np.asarray([f"ng-{i:06d}" for i in range(1, len(selected) + 1)])
    selected_df = df.iloc[selected].reset_index(drop=True)
    split = np.full(len(selected), "label_train", dtype=object)
    holdout_n = int(np.ceil(len(selected) * 0.20))
    if holdout_n:
        split[rng.choice(len(selected), size=holdout_n, replace=False)] = "holdout"
    # Visible queue: no author, labels, interaction, residual, reward, or score columns.
    queue = pd.DataFrame({
        "annotation_id": annotation_id,
        "title": selected_df["title"].fillna("").astype(str),
        "text": selected_df["text"].fillna("").astype(str),
        "media_count": pd.to_numeric(selected_df.get("n_image", 0), errors="coerce").fillna(0).astype(int),
        "video_count": pd.to_numeric(selected_df.get("n_video", 0), errors="coerce").fillna(0).astype(int),
        "annotation_split": split,
    })
    key = pd.DataFrame({
        "annotation_id": annotation_id,
        "tid": selected_df["tid"].to_numpy(),
        "row_index": selected,
        "create_time": selected_df["create_time"].to_numpy(),
        "author_id": selected_df.get("author_id", pd.Series([None] * len(selected_df))).to_numpy(),
        "topic_id": selected_df.get("topic_id", pd.Series([None] * len(selected_df))).to_numpy(),
        "year_bucket": year_bucket[selected],
        "score_bucket": score_bucket[selected],
        "prior_bucket": prior_bucket[selected],
        "dup_bucket": dup_bucket[selected],
        "question_slice": question[selected],
        "media_slice": media[selected],
        "low_effort_slice": low_effort[selected],
        "annotation_split": split,
        "model_score_full": model_score[selected],
        "residual": pd.to_numeric(selected_df.get("residual", np.nan), errors="coerce").to_numpy(float),
        "is_good_original": selected_df.get("is_good", pd.Series([None] * len(selected_df))).to_numpy(),
        "is_top_original": selected_df.get("is_top", pd.Series([None] * len(selected_df))).to_numpy(),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    queue_path = out_dir / "annotation_queue.csv"
    template_path = out_dir / "annotation_template.csv"
    key_path = out_dir / "annotation_key.parquet"
    queue.to_csv(queue_path, index=False, encoding="utf-8-sig")
    template = queue.copy()
    for col in ("informativeness", "originality", "clarity", "actionability", "low_effort", "reward_worthy", "uncertain", "reason"):
        template[col] = ""
    template.to_csv(template_path, index=False, encoding="utf-8-sig")
    key.to_parquet(key_path, index=False)
    schema = {
        "schema_version": 1,
        "language": "zh-CN",
        "blindness": {
            "visible_queue_excludes": ["author_id", "topic_id", "engagement", "views", "platform_labels", "settled_outcomes", "residual", "reward_score"],
            "private_key_must_not_be_given_to_annotators": True,
        },
        "instructions": "独立阅读标题和正文后评分;不要依据作者身份、互动量、平台标签或奖励结果;不确定时使用“不确定”并写原因。",
        "fields": [
            {"name": "informativeness", "type": "ordinal_1_5", "anchors": {"1": "无实质信息", "3": "有部分可用信息", "5": "信息充分且具体"}},
            {"name": "originality", "type": "ordinal_1_5", "anchors": {"1": "模板/重复明显", "3": "有少量独特内容", "5": "明显原创或有独特分析"}},
            {"name": "clarity", "type": "ordinal_1_5", "anchors": {"1": "难以理解", "3": "基本可理解", "5": "结构清晰、表达准确"}},
            {"name": "actionability", "type": "ordinal_1_5", "anchors": {"1": "无法帮助读者", "3": "有限帮助", "5": "能直接帮助判断、解决或学习"}},
            {"name": "low_effort", "type": "binary", "anchors": {"0": "不是明显低努力", "1": "主要是空泛、重复、诱导或无实质内容"}},
            {"name": "reward_worthy", "type": "ordinal_1_5", "anchors": {"1": "不应奖励", "3": "边界案例", "5": "明确值得质量奖励"}},
            {"name": "uncertain", "type": "binary", "description": "信息不足、语境缺失或需要外部事实才能判断时勾选"},
            {"name": "reason", "type": "free_text", "description": "一句话说明主要依据"},
        ],
        "recommendation": {"annotators": 2, "adjudication": "分歧超过 1 分或任一方 uncertain 时复核", "minimum_agreement": "报告 weighted kappa 或 Krippendorff alpha"},
        "splits": {"label_train": "用于质量头开发", "holdout": "锁定人工质量评测,不得用于调参"},
    }
    schema_path = out_dir / "annotation_schema.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "ready_for_human_annotation",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "score_input": str(score_path),
        "score_input_sha256": sha256_file(score_path) if score_path.exists() else None,
        "sampling_seed": int(seed),
        "target_rows": int(target),
        "actual_rows": len(queue),
        "queue": str(queue_path),
        "template": str(template_path),
        "private_key": str(key_path),
        "schema": str(schema_path),
        "split_counts": {str(k): int(v) for k, v in queue["annotation_split"].value_counts().items()},
        "coverage": {
            "year_bucket": {str(k): int(v) for k, v in pd.Series(year_bucket[selected]).value_counts().items()},
            "question": int(question[selected].sum()),
            "media": int(media[selected].sum()),
            "low_effort": int(low_effort[selected].sum()),
            "duplicate_ge2": int((dup[selected] >= 2).sum()),
            "duplicate_ge8": int((dup[selected] >= 8).sum()),
        },
        "label_values_not_used_for_sampling": ["is_good", "is_top", "is_share"],
        "population_weighting_warning": "分层样本适合切片审计;若要估计总体质量比例,必须保存纳入概率并做加权估计。",
    }
    manifest_path = out_dir / "annotation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    result = build_packet(args.input, args.scores, args.out_dir, args.target, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
