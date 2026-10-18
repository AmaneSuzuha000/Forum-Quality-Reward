# -*- coding: utf-8 -*-
"""用 aiotieba 抓取帖子互动指标 + 官方质量标注 → analysis.thread_engagement

接口说明
--------
贴吧接口需要客户端 protobuf 签名，aiotieba 实现了完整协议栈，故直接使用。

实测可得字段 (Client.get_threads, **无需登录**):
    view_num     浏览量
    agree        点赞数
    disagree     踩数
    share_num    分享数
    reply_num    回复数
    is_good      ★ 精华帖 —— 吧务人工加精, 即"官方质量标注"
    is_top       置顶
    is_share     转发
    create_time / last_time

范围限制
--------
    楼层级接口 (Client.get_posts / Client.get_user_threads) 本环境下不可用，
    抓取粒度止于主题帖级；楼层级点赞与用户主页数据不在覆盖范围内。

策略
----
get_threads(fid, pn=N) 深分页可翻到 pn≈3000 且能回到 2012 年,
但分页会重叠/漏帖, 所以本脚本以**多排序模式全量扫描**并累积唯一 tid,
覆盖每一条能拿到的帖子, 而不是赌单一分页序列。

用法
----
  # 探测 (不发批量请求)
  python scripts/reward/scrape_engagement.py --probe --tid 10284430088

  # 扫描 200 页 (约 3 分钟)
  python scripts/reward/scrape_engagement.py --pages 200

  # 全量多模式扫描 (CREATE + REPLY + HOT, 各 3000 页)
  python scripts/reward/scrape_engagement.py --full

  # 续跑 (自动跳过已扫过的页)
  python scripts/reward/scrape_engagement.py --pages 3000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resrank.runtime import connect, forum_id, log, setup_stdout

setup_stdout()

import argparse
import asyncio
import time

FID = forum_id()
PARSER_VERSION = "aiotieba-getthreads"
SORT_MODES = {"CREATE": 1, "REPLY": 6, "HOT": 3}


def to_dt(ts: int):
    import datetime as dt
    if not ts:
        return None
    try:
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone(dt.timedelta(hours=8)))
    except Exception:
        return None


# ------------------------------------------------------------------ 抓取
async def scan(pages: int, sort_names: list[str], sleep: float, resume: bool,
               max_minutes: float | None = None) -> int:
    import aiotieba
    from aiotieba import Client

    conn = connect()
    cur = conn.cursor()

    done_pages: dict[str, set[int]] = {s: set() for s in sort_names}
    if resume:
        cur.execute("""SELECT sort_mode, page FROM analysis.engagement_scan_progress
                       WHERE fid=%s""", (FID,))
        for m, p in cur.fetchall():
            done_pages.setdefault(m, set()).add(p)

    total_new = 0
    t_start = time.time()

    async with Client() as cli:
        for sort_name in sort_names:
            sort_enum = aiotieba.ThreadSortType(SORT_MODES[sort_name])
            got = skipped = 0
            for pn in range(1, pages + 1):
                if pn in done_pages.get(sort_name, set()):
                    skipped += 1
                    continue
                if max_minutes and (time.time() - t_start) / 60 > max_minutes:
                    log(f"  达到时间上限 {max_minutes}min, 停止")
                    break
                try:
                    ths = await asyncio.wait_for(
                        cli.get_threads(FID, pn=pn, sort=sort_enum), timeout=30)
                    ths = list(ths)
                except Exception:
                    continue
                if not ths:
                    log(f"  [{sort_name}] pn={pn} 空, 该模式结束")
                    break

                rows = []
                for t in ths:
                    rows.append((
                        int(t.tid), FID,
                        int(getattr(t, "view_num", 0) or 0),
                        int(getattr(t, "agree", 0) or 0),
                        int(getattr(t, "disagree", 0) or 0),
                        int(getattr(t, "share_num", 0) or 0),
                        int(getattr(t, "reply_num", 0) or 0),
                        bool(getattr(t, "is_good", False)),
                        bool(getattr(t, "is_top", False)),
                        bool(getattr(t, "is_share", False)),
                        int(getattr(getattr(t, "user", None), "user_id", 0) or 0) or None,
                        to_dt(int(getattr(t, "create_time", 0) or 0)),
                        to_dt(int(getattr(t, "last_time", 0) or 0)),
                        f"aiotieba/{sort_name}", PARSER_VERSION,
                    ))
                cur.executemany("""
                    INSERT INTO analysis.thread_engagement
                      (tid, fid, view_num, agree_num, disagree_num, share_num,
                       reply_num_live, is_good, is_top, is_share, author_id,
                       create_time, last_time, source, parser_version)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (tid) DO UPDATE SET
                      view_num   = GREATEST(EXCLUDED.view_num,  analysis.thread_engagement.view_num),
                      agree_num  = GREATEST(EXCLUDED.agree_num, analysis.thread_engagement.agree_num),
                      share_num  = GREATEST(EXCLUDED.share_num, analysis.thread_engagement.share_num),
                      reply_num_live = GREATEST(EXCLUDED.reply_num_live, analysis.thread_engagement.reply_num_live),
                      is_good    = EXCLUDED.is_good OR analysis.thread_engagement.is_good,
                      is_top     = EXCLUDED.is_top  OR analysis.thread_engagement.is_top,
                      is_share   = EXCLUDED.is_share OR analysis.thread_engagement.is_share,
                      author_id  = COALESCE(EXCLUDED.author_id, analysis.thread_engagement.author_id),
                      create_time= COALESCE(EXCLUDED.create_time, analysis.thread_engagement.create_time),
                      last_time  = COALESCE(EXCLUDED.last_time, analysis.thread_engagement.last_time),
                      scraped_at = now(), source = EXCLUDED.source,
                      parser_version = EXCLUDED.parser_version
                """, rows)
                cur.execute("""INSERT INTO analysis.engagement_scan_progress
                                 (fid, sort_mode, page, n_threads)
                               VALUES (%s,%s,%s,%s)
                               ON CONFLICT (fid,sort_mode,page) DO UPDATE
                               SET n_threads=EXCLUDED.n_threads, scanned_at=now()""",
                            (FID, sort_name, pn, len(rows)))
                conn.commit()
                got += len(rows); total_new += len(rows)
                if pn % 25 == 0 or pn == 1:
                    el = time.time() - t_start
                    log(f"  [{sort_name}] pn={pn:>4} 累计 {got:>6,} 行, "
                        f"{el/max(1,pn):.2f}s/页")
                if sleep:
                    await asyncio.sleep(sleep)
            log(f"  [{sort_name}] 完成: {got:,} 行 (跳过已扫 {skipped} 页)")

    conn.close()
    return total_new


# ------------------------------------------------------------------ 主
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=200)
    ap.add_argument("--modes", type=str, default="CREATE,REPLY,HOT")
    ap.add_argument("--full", action="store_true", help="三模式各 3000 页")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--tid", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0)
    a = ap.parse_args()

    if a.probe:
        return probe(a.tid or 10284430088)

    pages = 3000 if a.full else a.pages
    modes = [m.strip().upper() for m in a.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in SORT_MODES]
    if bad:
        print(f"未知排序模式: {bad}; 可用: {list(SORT_MODES)}"); return 1

    log(f"扫描 fid={FID} pages={pages} modes={modes} resume={not a.no_resume}")
    try:
        n = asyncio.run(asyncio.wait_for(
            scan(pages, modes, a.sleep, not a.no_resume,
                 a.max_minutes or None), timeout=7 * 3600))
    except TimeoutError:
        log("整体超时 (7h)"); n = 0
    log(f"本次写入/更新 {n:,} 行")

    conn = connect(); cur = conn.cursor()
    cur.execute("SELECT * FROM analysis.v_engagement_coverage")
    cols = [d[0] for d in cur.description]
    print("\n=== 覆盖率 ===")
    for k, v in zip(cols, cur.fetchone(), strict=False):
        print(f"  {k:20} {v}")
    conn.close()
    return 0


def probe(tid: int) -> int:
    """单帖探测: 不写库, 只报告字段可得性。"""
    from aiotieba import Client

    async def _run():
        async with Client() as cli:
            print(f"--- get_threads(fid={FID}) ---")
            try:
                ths = list(await asyncio.wait_for(cli.get_threads(FID, pn=1), timeout=30))
                t = ths[0] if ths else None
                if t:
                    print(f"  OK n={len(ths)}")
                    print(f"    view_num={t.view_num} agree={t.agree} disagree={t.disagree} "
                          f"share_num={t.share_num} reply_num={t.reply_num}")
                    print(f"    is_good={t.is_good} is_top={t.is_top} is_share={t.is_share}")
            except Exception as e:
                print(f"  ERR {type(e).__name__}: {str(e)[:150]}")
            print(f"--- get_posts(tid={tid}) ---")
            try:
                r = await asyncio.wait_for(cli.get_posts(tid), timeout=30)
                th = r.thread
                print(f"  view_num={th.view_num} agree={th.agree} share_num={th.share_num}")
            except Exception as e:
                print(f"  ERR {type(e).__name__}: {str(e)[:150]}")
                print("  (预期失败: tiebac.baidu.com 应用层不可达)")

    asyncio.run(asyncio.wait_for(_run(), timeout=120))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
