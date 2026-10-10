# -*- coding: utf-8 -*-
"""内容特征 —— 发布时即可得，不需要任何外部产物。

这是「零外部依赖」的一半：文本特征只看文本本身，媒体特征只看平台自带的计数。
没有任何一列需要额外跑一个模型或服务才能算出来。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u2b00-\u2bff]")


def _density(s: str) -> float:
    """去重字符占比 —— 「哈哈哈哈」密度低，「线性代数期末」密度高。"""
    ch = [c for c in str(s) if not c.isspace()]
    return len(set(ch)) / len(ch) if ch else 0.0


def text_features(df: pd.DataFrame, title_col: str = "title",
                  body_col: str = "text") -> pd.DataFrame:
    """通用文本特征。low_effort 是 OR 口径（模型特征），不是闸门口径。"""
    t = df[title_col].fillna("").astype(str) if title_col in df.columns else pd.Series([""] * len(df), index=df.index)
    b = df[body_col].fillna("").astype(str) if body_col in df.columns else pd.Series([""] * len(df), index=df.index)
    both = t + " " + b
    return pd.DataFrame({
        "len_title": t.str.len(),
        "len_text": b.str.len(),
        "info_density": b.map(_density),
        "n_digits": both.str.count(r"\d"),
        "n_emoji": both.map(lambda s: len(_EMOJI.findall(s))),
        "is_question": both.str.contains(r"[?？]").astype(int),
        "has_bracket_prefix": both.str.contains(r"^[【\[]", regex=True).astype(int),
        "low_effort": ((b.str.len() <= 5) | (t.str.len() <= 3)).astype(int),
    }, index=df.index)


def media_features(df: pd.DataFrame, image="n_image", video="n_video",
                   link="n_link", at="n_at", img_bytes="img_bytes",
                   img_px="img_px", emoji="n_emoji") -> pd.DataFrame:
    """媒体特征。列缺失时按 0 处理 —— 平台没有这项能力就不该编一个出来。"""
    def col(name):
        if name and name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0).astype(float)
        return pd.Series(0.0, index=df.index)

    ni, nv, nl, na, ne = col(image), col(video), col(link), col(at), col(emoji)
    ib, ip = col(img_bytes), col(img_px)
    n_media = ni + nv + nl + na
    return pd.DataFrame({
        "n_image": ni, "n_video": nv, "n_link": nl, "n_at": na,
        "img_bytes_log": np.log1p(ib.clip(lower=0)),
        "img_px_log": np.log1p(ip.clip(lower=0)),
        "has_image": (ni > 0).astype(int),
        "has_video": (nv > 0).astype(int),
        "has_link": (nl > 0).astype(int),
        "emoji_ratio": ne / n_media.replace(0, np.nan),
        "image_ratio": ni / n_media.replace(0, np.nan),
        # 媒体丰度：图片/视频/链接/emoji 的加权存在性
        "media_richness": ni * 2 + nv * 3 + nl * 1.5 + ne * 0.2,
    }, index=df.index).fillna(0.0)
