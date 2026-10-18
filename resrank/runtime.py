# -*- coding: utf-8 -*-
"""运行时公共设施：仓库根、配置、数据库连接、输出编码、日志。

每个入口脚本原先各抄一份（6 份配置加载、6 份连接串、13 份 stdout 包装），
实现一旦分叉就会互相漂移。这里收成唯一一份。
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

# resrank/ 的上一级即仓库根
ROOT = Path(__file__).resolve().parents[1]

_ENV_MAP = {
    "host": "TIEBA_DB_HOST",
    "port": "TIEBA_DB_PORT",
    "username": "TIEBA_DB_USER",
    "password": "TIEBA_DB_PASSWORD",
    "dbname": "TIEBA_DB_NAME",
}


def load_config(path=None) -> dict:
    """读取 config/config.toml。数据库字段可被 TIEBA_DB_* 环境变量覆盖。"""
    import tomllib

    p = Path(path) if path else ROOT / "config" / "config.toml"
    cfg: dict = {}
    if p.exists():
        with open(p, "rb") as fh:
            cfg = tomllib.load(fh)
    db = dict(cfg.get("database", {}))
    for key, env in _ENV_MAP.items():
        value = os.environ.get(env)
        if value:
            db[key] = value
    cfg["database"] = db
    return cfg


def dsn(config=None) -> str:
    c = (config or load_config()).get("database", {})
    parts = [f"{k}={c[k]}" for k in ("host", "port", "dbname", "username", "password") if c.get(k) not in (None, "")]
    if not parts:
        raise RuntimeError("缺少数据库配置：请复制 config/config.example.toml 为 config/config.toml，或设置 TIEBA_DB_* 环境变量")
    return " ".join(parts)


def connect(config=None):
    """返回一个 PostgreSQL 连接。"""
    import psycopg2

    return psycopg2.connect(dsn(config))


def forum_id(config=None) -> int:
    """目标吧的 fid。必须显式配置 —— 不内置默认值，避免把单一吧写死进代码。"""
    cfg = config or load_config()
    raw = os.environ.get("TIEBA_FID") or cfg.get("report", {}).get("fid")
    if not raw:
        raise RuntimeError("未配置目标吧：请在 config/config.toml 的 [report] fid 填写，或设置 TIEBA_FID")
    return int(raw)


def setup_stdout() -> None:
    """Windows 控制台按 UTF-8 输出中文。"""
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def log(msg) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
