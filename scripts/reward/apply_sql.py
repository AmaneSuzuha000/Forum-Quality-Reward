# -*- coding: utf-8 -*-
"""应用 SQL 迁移文件。用法: python scripts/reward/apply_sql.py <file.sql> [...]"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resrank.runtime import ROOT, connect, setup_stdout

setup_stdout()


def main() -> int:
    names = sys.argv[1:]
    if not names:
        print("用法: python scripts/reward/apply_sql.py <file.sql> [...]")
        return 2
    conn = connect()
    cur = conn.cursor()
    for name in names:
        cur.execute((ROOT / name).read_text(encoding="utf-8"))
        conn.commit()
        print(f"applied: {name}", flush=True)
    cur.execute("SELECT * FROM analysis.v_engagement_coverage")
    print("coverage:", cur.fetchone())
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
