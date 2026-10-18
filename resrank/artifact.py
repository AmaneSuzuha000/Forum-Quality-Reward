# -*- coding: utf-8 -*-
"""模型工件清单与完整性校验。"""
from __future__ import annotations

import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path


def _version(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest_for(model, artifact_path=None) -> dict:
    """生成不包含训练样本的可审计模型清单。"""
    out = {
        "manifest_version": 1,
        "algorithm": "ResRank",
        "algorithm_version": "4.1.0",
        "model_class": type(model).__name__,
        "objective": dict(getattr(model, "objective", {})),
        "context": list(getattr(model, "context", [])),
        "content": list(getattr(model, "content", [])),
        "text": list(getattr(model, "text", ()) or ()),
        "time_col": getattr(model, "time_col", None),
        "train_end": getattr(model, "train_end", None),
        "seasonal": bool(getattr(model, "seasonal", False)),
        "strict": bool(getattr(model, "strict", False)),
        "crossfit_splits": int(getattr(model, "crossfit_splits", 1)),
        "crossfit_seed": int(getattr(model, "crossfit_seed", 17)),
        "alpha_debias": float(getattr(model, "alpha_debias", 5.0)),
        "alpha_distill": float(getattr(model, "alpha_distill", 30.0)),
        "ngram": list(getattr(model, "ngram", (2, 3))),
        "min_df": int(getattr(model, "min_df", 10)),
        "max_features": int(getattr(model, "max_features", 8000)),
        "vocab_size": int(getattr(model, "vocab_size", 0)),
        "schema": model.schema() if hasattr(model, "schema") else {},
        "diagnostics": getattr(model, "diag_", {}),
        "runtime": {
            "python": platform.python_version(),
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scikit_learn": _version("scikit-learn"),
        },
    }
    if artifact_path is not None:
        p = Path(artifact_path)
        out["artifact"] = {
            "name": p.name,
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
    return out


def write_manifest(model, artifact_path, manifest_path=None) -> str:
    p = Path(manifest_path) if manifest_path else Path(str(artifact_path) + ".manifest.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest_for(model, artifact_path), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(p)


def verify_manifest(artifact_path, manifest_path=None) -> dict:
    artifact = Path(artifact_path)
    p = Path(manifest_path) if manifest_path else Path(str(artifact) + ".manifest.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    expected = data.get("artifact", {}).get("sha256")
    actual = sha256_file(artifact)
    return {
        "ok": expected == actual,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "artifact": str(artifact),
        "manifest": str(p),
    }
