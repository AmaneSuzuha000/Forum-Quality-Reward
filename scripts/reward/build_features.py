# -*- coding: utf-8 -*-
"""构建帖子特征宽表。

  - contents 各元素类型的计数与占比 (text/emoji/image/link/at/video/voice)
  - 图片：累计原始字节数、累计显示像素（图文信息量代理）
  - 视频：时长总计、最大播放量
  - media_richness 复合媒体丰度
  - 完整性闸门所需的重复度与低质标记

输出: data/reward/thread_features.parquet

用法:
    python scripts/reward/build_features.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resrank.runtime import ROOT, connect, forum_id, log, setup_stdout

setup_stdout()

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

OUT_DIR = ROOT / "data" / "reward"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FID = forum_id()

TYPES = ["text", "emoji", "image", "link", "at", "video", "voice"]

# ---- 完整性闸门常量 ----
DUP_SAT = 8.0         # 重复次数饱和点：达到即视为完全重复
DUP_W = 0.50          # 重复项权重
SELF_SHARE_W = 0.30   # 楼主自我回复占比权重
LOW_EFFORT_W = 0.30   # 低质标记权重

# 文本中出现即计数的校内实体词
ENTITIES = ["交大", "西南交大", "犀浦", "九里", "峨眉", "校区", "专业", "学院", "考研", "保研",
            "选课", "宿舍", "食堂", "就业", "校招", "实习", "绩点", "考试", "军训", "校庆"]


def fetch(cur, label, sql, args=None) -> pd.DataFrame:
    """执行一条查询并计时。"""
    t0 = time.time()
    cur.execute(sql, args or ())
    df = pd.DataFrame(cur.fetchall())
    log(f"  {label}: {len(df):,} rows ({time.time() - t0:.1f}s)")
    return df


def _as_int(value) -> int:
    """把平台返回的数值字段转成 int；字段缺失或类型异常时取 0。"""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _parse_contents(ct):
    """解析一条帖子的 contents 结构。

    返回 (类型计数 dict, 图片字节, 图片像素, 视频时长, 视频播放量)。
    """
    counts = dict.fromkeys(TYPES, 0)
    img_bytes = img_px = vid_duration = vid_views = 0
    if not isinstance(ct, list):
        return counts, img_bytes, img_px, vid_duration, vid_views
    for item in ct:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind in counts:
            counts[kind] += 1
        if kind == "image":
            img_bytes += _as_int(item.get("origin_size"))
            img_px += _as_int(item.get("show_width")) * _as_int(item.get("show_height"))
        elif kind == "video":
            vid_duration += _as_int(item.get("duration"))
            vid_views = max(vid_views, _as_int(item.get("view_num")))
    return counts, img_bytes, img_px, vid_duration, vid_views


def fetch_base(cur) -> pd.DataFrame:
    """帖子主表，并把 contents 展开成各类型计数。"""
    base = fetch(cur, "thread base", """
        SELECT tid, author_id, create_time, last_time, title, text,
               reply_num, author_level, contents
        FROM public.thread WHERE fid = %s""", (FID,))

    base["len_title"] = base["title"].fillna("").str.len()
    base["len_text"] = base["text"].fillna("").str.len()
    created = pd.to_datetime(base["create_time"], utc=True).dt.tz_convert("Asia/Shanghai")
    base["month"] = created.dt.to_period("M").astype(str)
    base["year"] = created.dt.year

    log("parsing contents (full vocabulary) ...")
    parsed = base["contents"].apply(_parse_contents)
    for t in TYPES:
        # 默认参数把 t 绑死在当前迭代，避免闭包晚绑定
        base[f"n_{t}"] = parsed.apply(lambda r, k=t: r[0][k]).astype(np.int32)
    base["img_bytes"] = parsed.apply(lambda r: r[1]).astype(np.int64)
    base["img_px"] = parsed.apply(lambda r: r[2]).astype(np.int64)
    base["vid_duration"] = parsed.apply(lambda r: r[3]).astype(np.int32)
    base["vid_views"] = parsed.apply(lambda r: r[4]).astype(np.int32)
    return base.drop(columns=["contents"])


def fetch_sources(cur) -> dict:
    """各条辅助线的按帖聚合表。"""
    sources = {
        "breadth": fetch(cur, "post breadth", """
            SELECT tid, count(*) AS n_posts, count(DISTINCT author_id) AS n_distinct_authors,
                   max(floor) AS max_floor, min(create_time) AS first_post_at
            FROM public.post WHERE fid = %s GROUP BY tid""", (FID,)),
        "opc": fetch(cur, "op self", """
            SELECT p.tid, count(*) AS n_op_posts FROM public.post p
            JOIN public.thread t ON t.tid = p.tid
            WHERE p.fid = %s AND p.author_id = t.author_id GROUP BY p.tid""", (FID,)),
        "comments": fetch(cur, "comments", """
            SELECT tid, count(*) AS n_comments, count(DISTINCT author_id) AS n_comment_authors,
                   count(DISTINCT pid) AS n_commented_posts
            FROM public.comment WHERE fid = %s GROUP BY tid""", (FID,)),
        "sentiment": fetch(cur, "sentiment", """
            SELECT tid, count(*) AS n_senti, avg(senti) AS senti_mean,
                   stddev_samp(senti) AS senti_std, sum((senti < -0.2)::int) AS n_neg,
                   sum((senti > 0.2)::int) AS n_pos
            FROM analysis.message_sentiment WHERE fid = %s AND tid IS NOT NULL GROUP BY tid""", (FID,)),
        "topic": fetch(cur, "topic",
                       "SELECT tid, topic_id, prob AS topic_prob "
                       "FROM analysis.thread_topic WHERE fid = %s", (FID,)),
        "multimodal": fetch(cur, "multimodal", """
            SELECT tid, avg(pred_confidence) AS mm_conf, avg(conflict_prob) AS mm_conflict,
                   avg(alignment) AS mm_align, max(pred_label) AS mm_label,
                   max(image_count) AS mm_images
            FROM analysis.thread_multimodal_pred WHERE fid = %s AND tid IS NOT NULL GROUP BY tid""", (FID,)),
        "irony": fetch(cur, "irony",
                       "SELECT tid, count(*) AS n_irony FROM analysis.irony_candidate "
                       "WHERE fid = %s AND tid IS NOT NULL GROUP BY tid", (FID,)),
        "clip": fetch(cur, "clip align",
                      "SELECT * FROM analysis.thread_clip_alignment WHERE fid = %s", (FID,)),
    }
    sources["sentiment"]["neg_share"] = (
        sources["sentiment"]["n_neg"] / sources["sentiment"]["n_senti"].replace(0, np.nan))
    return sources


def merge_sources(base: pd.DataFrame, sources: dict) -> pd.DataFrame:
    """把各条辅助线并到主表上。"""
    df = (base.merge(sources["breadth"], on="tid", how="left")
              .merge(sources["opc"], on="tid", how="left")
              .merge(sources["comments"], on="tid", how="left")
              .merge(sources["sentiment"], on="tid", how="left")
              .merge(sources["topic"], on="tid", how="left")
              .merge(sources["multimodal"], on="tid", how="left")
              .merge(sources["irony"], on="tid", how="left"))
    clip = sources["clip"]
    if "tid" in clip.columns and len(clip):
        df = df.merge(clip[[c for c in clip.columns if c != "fid"]],
                      on="tid", how="left", suffixes=("", "_clip"))
    log(f"merged: {len(df):,} x {len(df.columns)}")
    return df


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """互动结构、媒体构成与时间派生特征。"""
    for c in ["n_posts", "n_op_posts", "n_comments", "n_comment_authors", "n_irony"]:
        df[c] = df[c].fillna(0)
    df["op_self_share"] = df["n_op_posts"] / df["n_posts"].replace(0, np.nan)
    df["breadth_ratio"] = df["n_distinct_authors"] / df["n_posts"].replace(0, np.nan)
    df["depth_ratio"] = df["n_commented_posts"] / df["n_posts"].replace(0, np.nan)
    df["irony_rate"] = df["n_irony"] / df["n_posts"].replace(0, np.nan)

    segments = df[[f"n_{t}" for t in TYPES]].sum(axis=1)
    df["n_seg"] = segments
    for t in TYPES:
        df[f"{t}_ratio"] = df[f"n_{t}"] / segments.replace(0, np.nan)
    df["has_image"] = (df["n_image"] > 0).astype(int)
    df["has_video"] = (df["n_video"] > 0).astype(int)
    df["has_link"] = (df["n_link"] > 0).astype(int)
    df["img_bytes_log"] = np.log1p(df["img_bytes"].clip(lower=0))
    df["img_px_log"] = np.log1p(df["img_px"].clip(lower=0))
    # 媒体丰度：图片/视频/链接/emoji 的加权存在性
    df["media_richness"] = (df["n_image"] * 2 + df["n_video"] * 3
                            + df["n_link"] * 1.5 + df["n_emoji"] * 0.2)

    created = pd.to_datetime(df["create_time"], utc=True)
    df["first_reply_min"] = ((pd.to_datetime(df["first_post_at"], utc=True) - created)
                             .dt.total_seconds() / 60.0)
    df["first_reply_min_valid"] = (
        (df["first_reply_min"] >= 0) & df["first_reply_min"].notna()).astype(int)
    df["survival_days"] = ((pd.to_datetime(df["last_time"], utc=True) - created)
                           .dt.total_seconds().div(86400).clip(lower=0))
    return df


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """标题与正文的文本派生特征。"""

    def _density(s: str) -> float:
        """去空白后的字符去重率。复读机式的文本密度低。"""
        if not isinstance(s, str) or not s:
            return 0.0
        ch = [c for c in s if not c.isspace()]
        return len(set(ch)) / len(ch) if ch else 0.0

    df["info_density"] = df["text"].fillna("").apply(_density)
    both = df["title"].fillna("") + " " + df["text"].fillna("")
    df["n_digits"] = both.str.count(r"\d")
    df["has_bracket_prefix"] = both.str.contains(r"^[【\[]", regex=True).astype(int)
    df["is_question"] = both.str.contains(r"[?？]").astype(int)
    df["n_entities"] = both.apply(lambda s: sum(1 for e in ENTITIES if e in s))
    return df


def add_quality_marks(df: pd.DataFrame) -> pd.DataFrame:
    """低质标记与事实效用分。"""
    # low_effort 是模型特征之一，保持原定义不变，以保证模型行为可复现。
    df["low_effort"] = ((df["len_text"] <= 5) | (df["len_title"] <= 3)).astype(int)

    # 闸门专用：更严格的低质判定 —— 短正文 AND 短标题 AND 无任何媒体，三者同时成立。
    # 若沿用上面的 OR 口径，会把「正文只当引子、内容写在回复里」的连载帖误判为低质：
    # 实测曾因此误拦 6 条官方精华帖（含一条 1,396 回复的经验长贴，正文仅 "rt"）。
    # 改成 AND 后漏掉的精华从 6 条降到 1 条，而刷屏帖（重复 >= 100）依然 100% 被拦
    # —— 它们由「重复项」而非「低质项」拦下，防刷能力无损。
    media_any = (df["n_image"] > 0) | (df["n_link"] > 0) | (df["n_video"] > 0)
    df["low_effort_gate"] = (
        (df["len_text"] <= 5) & (df["len_title"] <= 3) & ~media_any).astype(int)

    df["y_reply"] = np.log1p(df["reply_num"].clip(lower=0))
    df["y_comment"] = np.log1p(df["n_comments"])
    df["y_breadth"] = np.log1p(df["n_distinct_authors"].fillna(0))
    df["y_survival"] = np.log1p(df["survival_days"].clip(lower=0))
    df["utility"] = (df["y_reply"] + 0.5 * df["y_comment"]
                     + 0.5 * df["y_breadth"] + 0.3 * df["y_survival"])
    return df


def add_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """重复检测与完整性闸门。

    闸门把「能不能发」和「该发多少」分开：integrity 只回答够不够格进发放池。
    integrity 不是模型特征，只用于结算，因此改动这里不需要重训模型。
    """
    log("duplicate detection ...")
    txt = df["text"].fillna("").str.strip()
    df["_h"] = txt.where(txt.str.len() > 0, other=pd.NA)
    df["dup_count_author"] = (df.groupby(["author_id", "_h"], dropna=True)["tid"]
                              .transform("size").fillna(1).astype(int))
    df["dup_count_global"] = (df.groupby("_h", dropna=True)["tid"]
                              .transform("size").fillna(1).astype(int))
    df = df.drop(columns=["_h"])

    dup = df["dup_count_global"].clip(lower=1).astype(float)
    dup_term = np.where(dup > 1, np.minimum(1.0, np.log1p(dup) / np.log1p(DUP_SAT)), 0.0)
    self_share = df["op_self_share"].fillna(0).clip(0, 1)
    low = df["low_effort_gate"].astype(float)
    df["dup_term"] = dup_term
    df["integrity"] = (1.0 - (DUP_W * dup_term
                              + SELF_SHARE_W * self_share
                              + LOW_EFFORT_W * low)).clip(0, 1)
    return df


def build() -> pd.DataFrame:
    """跑完全部取数与派生，返回特征宽表。"""
    conn = connect()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        base = fetch_base(cur)
        sources = fetch_sources(cur)
    finally:
        conn.close()
    df = merge_sources(base, sources)
    df = add_structure_features(df)
    df = add_text_features(df)
    df = add_quality_marks(df)
    return add_integrity(df)


def main() -> int:
    df = build()
    out = OUT_DIR / "thread_features.parquet"
    df.to_parquet(out, index=False)
    log(f"WROTE {out} ({len(df):,} x {len(df.columns)}, {out.stat().st_size/1e6:.1f} MB)")

    print("\n=== 固有结构特征 ===")
    for c in ["n_text", "n_emoji", "n_image", "n_link", "n_at", "n_video", "n_voice",
              "img_bytes_log", "img_px_log", "media_richness", "has_image", "has_video", "has_link"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            print(f"  {c:16} mean={s.mean():>10.3f} p50={s.median():>8.1f} "
                  f"p90={s.quantile(.9):>9.1f} max={s.max():>10.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
