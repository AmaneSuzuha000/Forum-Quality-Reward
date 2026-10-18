# -*- coding: utf-8 -*-
"""构造「创作量」信号。

动机 (实测, 以吧务人工加精 `analysis.thread_engagement.is_good` 为外部真值):
  metric              jinghua AUC   rho(., author prior)
  self_n  (自回条数)      0.7822        +0.2727
  self_long (>=15字)      0.8041        +0.2131   <- 最强且偏差低
  self_long_uniq (去重)    0.8041        +0.2131
  self_bump (<=5字顶帖)     0.5304        +0.1477   <- 纯噪声, 必须剔除
  reply_7d (他人回复)      0.6982        +0.8017   <- 主要在学"谁发的"

结论: 顶帖(<=5字)占 OP 自回的 36.5%, 但判别力仅 0.53 (接近抛硬币)。
      剔除顶帖后信号更强 (0.7822 -> 0.8041), 作者偏差更低 (0.2727 -> 0.2131)。

输出: data/reward/self_content.parquet
  tid, self_n, self_bump, self_long, self_long_uniq, self_deep_uniq
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resrank.runtime import ROOT, connect, forum_id, log, setup_stdout

setup_stdout()

import time

import pandas as pd

FID = forum_id()
MIN_LONG = 15      # 实质内容门槛(字符)
MIN_DEEP = 30      # 深度内容门槛(更保守的备选)






def main() -> int:
    conn = connect()
    TW = ("p.create_time > t.create_time "
          "AND p.create_time <= t.create_time + interval '7 days' "
          "AND p.author_id = t.author_id")
    TXT = "btrim(coalesce(p.text,''))"
    q = f"""
    SELECT t.tid,
      count(p.pid) FILTER (WHERE {TW})                                  AS self_n,
      count(p.pid) FILTER (WHERE {TW} AND length({TXT}) <= 5)            AS self_bump,
      count(p.pid) FILTER (WHERE {TW} AND length({TXT}) >= {MIN_LONG})   AS self_long,
      count(DISTINCT {TXT}) FILTER (WHERE {TW}
            AND length({TXT}) >= {MIN_LONG})                             AS self_long_uniq,
      count(DISTINCT {TXT}) FILTER (WHERE {TW}
            AND length({TXT}) >= {MIN_DEEP})                             AS self_deep_uniq
    FROM public.thread t
    LEFT JOIN public.post p ON p.tid = t.tid
    WHERE t.fid = %s
    GROUP BY t.tid"""
    log("aggregating OP self-replies (7-day window) ...")
    t0 = time.time()
    df = pd.read_sql(q, conn, params=(FID,))
    conn.close()
    log(f"  {len(df):,} rows in {time.time()-t0:.1f}s")

    for c in ["self_n", "self_bump", "self_long", "self_long_uniq", "self_deep_uniq"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    tn = int(df["self_n"].sum())
    print()
    print("=== OP self-reply composition ===")
    print(f"  total self-replies        {tn:,}")
    print(f"  bump-only (<=5 chars)     {int(df['self_bump'].sum()):,}  "
          f"({100*df['self_bump'].sum()/max(tn,1):.1f}%)")
    print(f"  substantive (>={MIN_LONG} chars) {int(df['self_long'].sum()):,}  "
          f"({100*df['self_long'].sum()/max(tn,1):.1f}%)")
    print(f"  threads with creation     {int((df['self_long_uniq']>0).sum()):,}  "
          f"({100*(df['self_long_uniq']>0).mean():.1f}%)")

    out = ROOT / "data" / "reward" / "self_content.parquet"
    df.to_parquet(out, index=False)
    print()
    log(f"WROTE {out} ({len(df):,} x {len(df.columns)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
