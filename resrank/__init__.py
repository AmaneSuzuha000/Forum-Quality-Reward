# -*- coding: utf-8 -*-
"""ResRank —— 去混杂残差排序 (Deconfounded Residual Ranking)。

一个通用的「观测数据 → 内容侧打分器 → 配额分配」算法，四个步骤：

    L1 目标合成   u = Σ wₖ·log1p(yₖ)          多路交互 → 单标量目标
    L2 去混杂     r = u − E[u | 上下文]        剥掉「谁发的 / 何时发的 / 什么主题」
    L3 蒸馏       ŝ = f(内容) ≈ r              学一个冷启动可用的打分器
    L4 配额分配   队列内分位 → 档位 → 积分      预算自动跟随规模

为什么是这四步
--------------
观测到的互动量 = 真实质量 + 暴露偏差。大 V 的帖天然被更多人看到，
周末发的帖天然被更多人刷到。直接按互动量排序，奖励的是「已经有的流量」，
不是「写得好」—— 强者愈强，新人永远排不上。

L2 把偏差剥掉，得到的残差 r 回答「这篇比同等条件下的平均水平好多少」。
但 r 要等窗口关闭才算得出来，而奖励要在窗口内发 —— L3 把 r 压进内容特征，
得到一个**只看内容**的打分器，冷启动即可用。

L4 用队列内相对分位而不是绝对阈值：奖励总额自动跟随队列规模，不需要每年重调预算。

用法
----
    from resrank import Ranker

    r = Ranker(
        objective={"reply_7d": 0.60, "comment_7d": 0.30, "self_bump": -0.60},
        context=["author_prior", "month_activity", "topic_te"],
        content=["len_text", "n_image", "author_level"],
        text=("title", "text"),
        time_col="create_time", train_end="2022-12-31",
    )
    r.fit(df)
    r.residual(df)          # L1+L2：结算量（窗口关闭后）
    r.score(df)             # L1+L2+L3：预测分（窗口内）
    r.allocate(df)          # L4：档位 + 积分
    r.save("model.pkl")     # 单文件，含全部拟合产物

依赖：numpy / pandas / scipy / scikit-learn。无预训练模型，无 GPU，无网络。
"""
from .allocate import allocate, summarize
from .artifact import manifest_for, verify_manifest, write_manifest
from .config import SCHEMES, Scheme, pick_scheme
from .distill import BODY_CHARS, TITLE_CHARS, Distiller, build_doc, build_docs, clean
from .ranker import Ranker
from .signal import Debiaser, compose

__version__ = "1.0.0"
__all__ = [
    "BODY_CHARS",
    "SCHEMES",
    "TITLE_CHARS",
    "Debiaser",
    "Distiller",
    "Ranker",
    "Scheme",
    "allocate",
    "build_doc",
    "build_docs",
    "clean",
    "compose",
    "manifest_for",
    "pick_scheme",
    "summarize",
    "verify_manifest",
    "write_manifest",
]
