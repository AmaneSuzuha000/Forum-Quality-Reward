# -*- coding: utf-8 -*-
"""抓取最近 N 天的帖子 (aiotieba, 匿名) -> data/reward/recent_threads.parquet

用 sort=CREATE 按发帖时间倒序翻页, 直到整页都早于截止时间。
完整保留 contents/user 结构为 JSON, 便于离线派生特征(不丢信息)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resrank.runtime import ROOT, forum_id, log, setup_stdout

setup_stdout()

import argparse
import asyncio
import datetime as dt
import json

import pandas as pd

FID = forum_id()
CST = dt.timezone(dt.timedelta(hours=8))




def ser(o, depth=0):
    """通用序列化: 保留 protobuf-like 对象的全部字段。"""
    if depth > 5:
        return str(o)
    if o is None or isinstance(o, (int, float, bool, str)):
        return o
    if isinstance(o, (list, tuple)):
        return [ser(x, depth + 1) for x in o]
    ann = getattr(o, "__annotations__", None)
    if isinstance(ann, dict):
        return {k: ser(getattr(o, k, None), depth + 1) for k in ann}
    d = getattr(o, "__dict__", None)
    if isinstance(d, dict):
        return {k: ser(v, depth + 1) for k, v in d.items() if not k.startswith("_")}
    return str(o)


async def fetch(days: int, max_pages: int):
    import aiotieba
    from aiotieba import Client

    cutoff = dt.datetime.now(CST) - dt.timedelta(days=days)
    log(f"截止时间 = {cutoff:%Y-%m-%d %H:%M} (最近 {days} 天)")

    rows, seen = [], set()
    async with Client() as cli:
        # 先打印结构, 确认字段命名
        first = list(await cli.get_threads(FID, pn=1, sort=aiotieba.ThreadSortType.CREATE))
        if first:
            t0 = first[0]
            print("--- Thread_t 字段 ---", flush=True)
            print("  ", list(getattr(type(t0), "__annotations__", {}).keys()), flush=True)
            c0 = getattr(t0, "contents", None)
            print("--- Contents_t 字段 ---", flush=True)
            print("  ", list(getattr(type(c0), "__annotations__", {}).keys()), flush=True)
            u0 = getattr(t0, "user", None)
            print("--- UserInfo_t 字段 (前20) ---", flush=True)
            print("  ", list(getattr(type(u0), "__annotations__", {}).keys())[:20], flush=True)
            print("--- contents 序列化样例 ---", flush=True)
            print("  ", json.dumps(ser(c0), ensure_ascii=False)[:600], flush=True)

        for pn in range(1, max_pages + 1):
            try:
                ths = list(await asyncio.wait_for(
                    cli.get_threads(FID, pn=pn, sort=aiotieba.ThreadSortType.CREATE), timeout=30))
            except Exception as e:
                log(f"  pn={pn} ERR {type(e).__name__}"); continue
            if not ths:
                log(f"  pn={pn} 空, 停止"); break
            newest = max(t.create_time for t in ths)
            oldest = min(t.create_time for t in ths)
            for t in ths:
                if t.tid in seen:
                    continue
                ct = dt.datetime.fromtimestamp(t.create_time, CST)
                if ct < cutoff:
                    continue
                seen.add(t.tid)
                u = getattr(t, "user", None)
                rows.append({
                    "tid": int(t.tid), "title": str(getattr(t, "title", "") or ""),
                    "text": str(getattr(getattr(t, "contents", None), "text", "") or ""),
                    "contents_json": json.dumps(ser(getattr(t, "contents", None)), ensure_ascii=False),
                    "user_json": json.dumps(ser(u), ensure_ascii=False),
                    "author_id": int(getattr(t, "author_id", 0) or (getattr(u, "user_id", 0) if u else 0) or 0),
                    "author_level": getattr(u, "level_id", None) if u else None,
                    "create_time": ct, "last_time": dt.datetime.fromtimestamp(
                        int(getattr(t, "last_time", 0) or 0), CST) if getattr(t, "last_time", 0) else None,
                    "view_num": int(getattr(t, "view_num", 0) or 0),
                    "reply_num": int(getattr(t, "reply_num", 0) or 0),
                    "agree": int(getattr(t, "agree", 0) or 0),
                    "disagree": int(getattr(t, "disagree", 0) or 0),
                    "share_num": int(getattr(t, "share_num", 0) or 0),
                    "is_good": bool(getattr(t, "is_good", False)),
                    "is_top": bool(getattr(t, "is_top", False)),
                    "is_share": bool(getattr(t, "is_share", False)),
                    "is_hide": bool(getattr(t, "is_hide", False)),
                })
            dnew = dt.datetime.fromtimestamp(newest, CST); dold = dt.datetime.fromtimestamp(oldest, CST)
            n_inwin = sum(1 for t in ths
                          if dt.datetime.fromtimestamp(t.create_time, CST) >= cutoff)
            log(f"  pn={pn:>3} 页内 {len(ths)} 帖 ({dold:%m-%d %H:%M} ~ {dnew:%m-%d %H:%M}) "
                f"窗口内 {n_inwin}  累计命中 {len(rows)}")
            # 停止条件: 整页**没有任何**窗口内帖子。单个旧帖(置顶等)不应终止扫描。
            if n_inwin == 0:
                log("  整页均早于截止, 停止"); break
            await asyncio.sleep(0.05)

    df = pd.DataFrame(rows).sort_values("create_time").reset_index(drop=True)
    out = ROOT / "data" / "reward" / "recent_threads.parquet"
    df.to_parquet(out, index=False)
    log(f"WROTE {out}  ({len(df):,} 帖)")
    print()
    print("=== 概况 ===")
    if len(df):
        print(f"  时间跨度 {df['create_time'].min():%Y-%m-%d %H:%M} ~ {df['create_time'].max():%Y-%m-%d %H:%M}")
        print(f"  唯一作者 {df['author_id'].nunique():,}")
        print(f"  置顶 {int(df['is_top'].sum())}  精华 {int(df['is_good'].sum())}  隐藏 {int(df['is_hide'].sum())}")
        print(f"  view_num  均值={df['view_num'].mean():.1f} 中位={df['view_num'].median():.0f}")
        print(f"  reply_num 均值={df['reply_num'].mean():.1f} 中位={df['reply_num'].median():.0f}")
        print(f"  agree     均值={df['agree'].mean():.1f} 中位={df['agree'].median():.0f}")
        print("  \n各日发帖量:")
        for d, n in df.groupby(df["create_time"].dt.date).size().items():
            print(f"    {d}  {n:>4} 帖")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--max-pages", type=int, default=40)
    a = ap.parse_args()
    raise SystemExit(asyncio.run(fetch(a.days, a.max_pages)))
