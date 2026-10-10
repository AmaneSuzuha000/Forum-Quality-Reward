# -*- coding: utf-8 -*-
"""构造 **7 天观察窗口** 特征 (周度奖励计划专用)。

为什么必须重建
--------------
现有 utility 用的是 thread.reply_num —— 那是**抓取时刻的终生累计值**。
周度计划必须在**发帖后 7 天**关闭窗口, 口径不一致会导致:
  - 老帖的"终生回复"天然更大 -> 奖励偏向老帖
  - 新帖窗口未满 -> 被系统性低估
本脚本直接从 楼层/楼中楼 **时间戳** 算真正的 7 天窗口指标。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resrank.runtime import ROOT, connect, forum_id, log, setup_stdout

setup_stdout()

import time

import numpy as np
import pandas as pd

FID = forum_id()



def main() -> int:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT max(create_time) FROM public.post WHERE fid=%s", (FID,))
    censor = cur.fetchone()[0]
    log(f"截尾点 (库内最晚楼层时间) = {censor}")

    log("聚合楼层 (7天/3天/终生/广度/活跃天数/首复延迟) ...")
    t0 = time.time()
    cur.execute("""
    SELECT t.tid,
           count(p.pid) FILTER (WHERE p.create_time > t.create_time
                                  AND p.create_time <= t.create_time + interval '7 days'
                                  AND p.author_id <> t.author_id)                     AS reply_7d,
           count(p.pid) FILTER (WHERE p.create_time > t.create_time
                                  AND p.create_time <= t.create_time + interval '3 days'
                                  AND p.author_id <> t.author_id)                     AS reply_3d,
           count(p.pid) FILTER (WHERE p.create_time > t.create_time
                                  AND p.create_time <= t.create_time + interval '7 days'
                                  AND p.author_id = t.author_id)                      AS self_7d,
           count(DISTINCT p.author_id) FILTER (WHERE p.create_time > t.create_time
                                  AND p.create_time <= t.create_time + interval '7 days'
                                  AND p.author_id <> t.author_id)                     AS breadth_7d,
           count(DISTINCT date_trunc('day', p.create_time))
                                  FILTER (WHERE p.create_time > t.create_time
                                  AND p.create_time <= t.create_time + interval '7 days') AS actdays_7d,
           min(p.create_time)     FILTER (WHERE p.create_time > t.create_time)        AS first_reply_t,
           count(p.pid) FILTER (WHERE p.create_time > t.create_time
                                  AND p.author_id <> t.author_id)                     AS reply_life_others,
           count(DISTINCT p.author_id) FILTER (WHERE p.create_time > t.create_time
                                  AND p.author_id <> t.author_id)                     AS breadth_life,
           max(p.create_time)                                                          AS last_reply_t
    FROM public.thread t
    LEFT JOIN public.post p ON p.tid = t.tid
    WHERE t.fid = %s
    GROUP BY t.tid
    """, (FID,))
    cols = [d[0] for d in cur.description]
    floors = pd.DataFrame(cur.fetchall(), columns=cols)
    log(f"  floors: {len(floors):,} rows in {time.time()-t0:.1f}s")

    log("聚合楼中楼 ...")
    cur.execute("""
    SELECT t.tid,
           count(c.cid) FILTER (WHERE c.create_time > t.create_time
                                  AND c.create_time <= t.create_time + interval '7 days') AS comment_7d,
           count(DISTINCT c.author_id) FILTER (WHERE c.create_time > t.create_time
                                  AND c.create_time <= t.create_time + interval '7 days') AS cb_7d,
           count(c.cid) FILTER (WHERE c.create_time > t.create_time)                    AS comment_life
    FROM public.thread t
    LEFT JOIN public.comment c ON c.tid = t.tid
    WHERE t.fid = %s
    GROUP BY t.tid
    """, (FID,))
    cols = [d[0] for d in cur.description]
    cms = pd.DataFrame(cur.fetchall(), columns=cols)
    log(f"  comments: {len(cms):,} rows in {time.time()-t0:.1f}s")

    th = pd.read_sql("""SELECT tid, author_id, create_time, reply_num, last_time
                        FROM public.thread WHERE fid=%s""", conn, params=(FID,))
    conn.close()

    df = th.merge(floors, on="tid", how="left").merge(cms, on="tid", how="left")
    for c in df.columns:
        if c.startswith(("reply_","self_","breadth_","actdays_","comment_","cb_")):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # 首条回复延迟(分钟)
    df["first_reply_min"] = (pd.to_datetime(df["first_reply_t"], errors="coerce")
                             - pd.to_datetime(df["create_time"], errors="coerce")).dt.total_seconds()/60
    df["first_reply_min"] = df["first_reply_min"].clip(lower=0)

    # 持续性: 第 4-7 天回复占 7 天总量比
    df["late_ratio_7d"] = ((df["reply_7d"] - df["reply_3d"])
                           / df["reply_7d"].replace(0, np.nan)).clip(0, 1).fillna(0)

    # 窗口完整性: 发帖时间 + 7 天是否已过截尾点
    ct = pd.to_datetime(df["create_time"], errors="coerce")
    df["window_complete"] = (ct + pd.Timedelta(days=7) <= pd.Timestamp(censor)).astype(int)

    out = ROOT/"data"/"reward"/"window7d.parquet"
    df.to_parquet(out, index=False)
    log(f"WROTE {out}  ({len(df):,} x {len(df.columns)})")

    # ---------------- 报告 ----------------
    d = df[df["window_complete"] == 1]
    print("\n=== 窗口完整性 ===")
    print(f"  完整窗口 {len(d):,} / {len(df):,} ({100*len(d)/len(df):.1f}%)")
    print(f"  截尾(不足7天) {len(df)-len(d):,}  <- 部署时应排除或单独处理")

    print("\n=== 7天 vs 终生 (仅完整窗口) ===")
    print(f"  回复数     7天均={d['reply_7d'].mean():>7.2f}  终生(他人)均={d['reply_life_others'].mean():>7.2f}  "
          f"终生(官方reply_num)均={pd.to_numeric(d['reply_num'],errors='coerce').mean():>7.2f}")
    print(f"  参与广度   7天均={d['breadth_7d'].mean():>7.2f}  终生均={d['breadth_life'].mean():>7.2f}")
    print(f"  楼中楼     7天均={d['comment_7d'].mean():>7.2f}  终生均={d['comment_life'].mean():>7.2f}")

    with np.errstate(all="ignore"):
        frac = (d["reply_7d"] / d["reply_life_others"].replace(0, np.nan)).dropna()
    print(f"\n  ★ 7天捕获终生回复比例: 中位={frac.median():.3f}  均值={frac.mean():.3f}")

    lg = np.log1p(d["reply_7d"]); ll = np.log1p(d["reply_life_others"])
    print(f"  相关 log(7天) vs log(终生)  r={lg.corr(ll):.4f}")

    print("\n=== 零回复率 (关键: 小样本周度队列的区分度) ===")
    print(f"  7天零回复帖: {(d['reply_7d']==0).sum():,} ({100*(d['reply_7d']==0).mean():.1f}%)")
    print(f"  终生零回复帖: {(d['reply_life_others']==0).sum():,} ({100*(d['reply_life_others']==0).mean():.1f}%)")

    print("\n=== 分位数 (7天他人回复) ===")
    for q in (0.5, 0.75, 0.9, 0.95, 0.99):
        print(f"  p{int(q*100):<3} = {d['reply_7d'].quantile(q):>7.1f}")

    print("\n=== 首复延迟 (分钟) ===")
    f = d["first_reply_min"].dropna()
    print(f"  中位={f.median():.1f}  p90={f.quantile(.9):.1f}  "
          f"有回复帖占比={100*(d['reply_7d']>0).mean():.1f}%")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
