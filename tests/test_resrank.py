# -*- coding: utf-8 -*-
"""ResRank 自检 —— 不依赖任何外部数据，几分钟内证明算法成立。

跑法：  pytest tests/test_resrank.py -q
     （也可直接 python tests/test_resrank.py 看逐条输出）

核心那个测试（test_debias_recovers_quality）是这套算法的存在理由：
构造一批**真实质量与作者无关**的对象，但让互动量被作者曝光度污染。
朴素排序（直接按互动量排）会排成「谁有名谁第一」；
ResRank 应该把它纠正回「谁写得好谁第一」。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resrank import (
    SCHEMES,
    Distiller,
    Ranker,
    allocate,
    build_doc,
    clean,
    compose,
    pick_scheme,
    summarize,
    verify_manifest,
)
from resrank.distill import BODY_CHARS, TITLE_CHARS

_OK, _FAIL = [], []


def check(name, cond, detail=""):
    """记录并打印一条断言。不成立时抛 AssertionError，好让 pytest 收集到失败。"""
    (_OK if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        raise AssertionError(name + (("  " + detail) if detail else ""))


def _synthetic(n=6000, seed=0):
    """造数据：真实质量 q 与作者无关，但互动量被作者名气污染。"""
    rng = np.random.default_rng(seed)
    n_author, n_vocab = 20, 300
    author = rng.integers(0, n_author, n)
    # 名气差异必须**压过**质量差异，否则测不出「朴素排序被名气带跑」
    author_skill = rng.normal(0.0, 1.5, n_author)
    skill = author_skill[author]                                   # 名气 -> 曝光
    q = rng.normal(0.0, 1.0, n)                                    # 真实质量
    year = rng.choice([2020, 2021, 2022, 2023, 2024], n, p=[.15, .2, .25, .2, .2])

    # 观测互动 ≈ exp(质量 + 名气 + 噪声)
    obs = np.exp(q + skill + rng.normal(0, 0.4, n))
    # 文本：词分布被 q 平移 —— 好内容用词不一样
    direction = rng.normal(0, 1, n_vocab)
    logits = q[:, None] * direction[None, :] * 0.9
    logits -= logits.max(1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(1, keepdims=True)
    words = []
    for i in range(n):
        k = int(rng.integers(12, 28))
        idx = rng.choice(n_vocab, k, p=prob[i])
        words.append(" ".join(f"w{j}" for j in idx))

    return pd.DataFrame({
        "tid": np.arange(n),
        "create_time": pd.to_datetime([f"{y}-01-01" for y in year], utc=True),
        "author_id": author,
        "title": [f"标题 {w[:20]}" for w in words],
        "text": words,
        "reply": obs,
        "author_prior": skill,
        "author_level": skill * 10,
        "q_true": q, "skill_true": skill,
    })


# --------------------------------------------------------------------- 测试
def test_text_contract():
    print("\n[1] 文本契约")
    check("clean 折叠空白", clean("  a\n\tb  ", 99) == "a b")
    check("clean 截断", clean("x" * 500, 10) == "x" * 10)
    check("clean 处理 None", clean(None, 10) == "")
    d = build_doc("t" * 500, "b" * 900)
    check("标题截到 160", ("t" * TITLE_CHARS) in d and ("t" * (TITLE_CHARS + 1)) not in d)
    check("正文截到 400", ("b" * BODY_CHARS) in d and ("b" * (BODY_CHARS + 1)) not in d)
    check("有正文才加分隔", build_doc("t", "  ") == "标题：t")
    check("正文为空只留标题", build_doc("t", "") == "标题：t")


def test_debias_recovers_quality():
    """算法的存在理由：把「谁有名」纠正成「谁写得好」。"""
    print("\n[2] 去混杂 —— 从名气回到质量")
    d = _synthetic()
    tr = (d["create_time"].dt.year <= 2022).to_numpy()

    r = Ranker(objective={"reply": 1.0}, context=["author_prior", "author_level"],
               content=["author_level"], text=("title", "text"),
               time_col="create_time", train_end="2022-12-31", verbose=False)
    r.fit(d)
    te = ~tr
    q = d["q_true"].to_numpy()

    u = compose(d, r.objective, r.log1p)   # L1 目标（朴素排序的依据）
    res = r.residual(d)
    sc = r.score(d)
    naive = spearmanr(u[te], q[te]).statistic
    resid = spearmanr(res[te], q[te]).statistic
    model = spearmanr(sc[te], q[te]).statistic
    print(f"       朴素(直接按互动量)  rho(互动, 真质量) = {naive:+.4f}")
    print(f"       L1+L2 残差          rho(残差, 真质量) = {resid:+.4f}")
    print(f"       L1+L2+L3 模型分      rho(模型, 真质量) = {model:+.4f}")
    check("去混杂显著优于朴素", resid > naive + 0.10, f"{naive:+.4f} -> {resid:+.4f}")
    check("模型分显著为正", model > 0.25, f"{model:+.4f}")

    # 模型分不该重新捡回作者名气
    skill = d["skill_true"].to_numpy()
    rho_a = abs(spearmanr(sc[te], skill[te]).statistic)
    rho_q = abs(spearmanr(sc[te], q[te]).statistic)
    print(f"       rho(模型, 作者名气) = {rho_a:+.4f}   rho(模型, 真质量) = {rho_q:+.4f}")
    check("模型更贴质量而非名气", rho_q > rho_a)


def test_distiller_standalone():
    print("\n[3] 蒸馏器单独可用")
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 800)
    docs = [f"标题：{'好' if v > 0 else '差'}内容 {i}" for i, v in enumerate(y)]
    X = np.c_[y * 2 + rng.normal(0, .1, 800), rng.normal(0, 1, 800)]
    m = Distiller(alpha=1.0).fit(docs[:600], X[:600], y[:600])
    pred = m.predict(docs[600:], X[600:])
    rho = spearmanr(pred, y[600:]).statistic
    check("能从文本+特征恢复目标", rho > 0.9, f"rho={rho:+.4f}")
    check("词表非空", m.vocab_size > 0, f"{m.vocab_size} 词")
    m2 = Distiller(alpha=1.0).fit(None, X[:600], y[:600])
    check("只给特征也能跑", len(m2.predict(None, X[600:])) == 200)
    m3 = Distiller(alpha=1.0).fit(docs[:600], None, y[:600])
    check("只给文本也能跑", len(m3.predict(docs[600:], None)) == 200)
    try:
        Distiller().fit(None, None, y)
        check("两者都空应报错", False)
    except ValueError:
        check("两者都空应报错", True)


def test_allocate():
    print("\n[4] 配额分配")
    s = np.arange(1000, dtype=float)          # 1000 个对象，分数递增
    plan = allocate(s, scheme=SCHEMES[5])
    t = plan["tier"].to_numpy()
    check("最高分进 T4", t[-1] == 4)
    check("最低分进 T0", t[0] == 0)
    check("获奖比例 ≈ 40%", abs((t > 0).mean() - 0.40) < 0.01, f"{(t>0).mean():.3f}")
    check("积分单调不减", np.all(np.diff(plan['points'].to_numpy()) >= 0))
    check("T0 零分", plan.loc[t == 0, "points"].sum() == 0)

    blk = np.zeros(1000, bool); blk[:100] = True
    p2 = allocate(s, scheme=SCHEMES[5], blocked=blk)
    check("被拦下得 0 分", p2.loc[blk, "points"].sum() == 0)
    check("被拦下 tier=-1", (p2.loc[blk, "tier"] == -1).all())
    check("被拦下不占分位", p2.loc[blk, "pct"].isna().all())

    s2 = np.r_[np.zeros(500), np.ones(500)]     # 大量并列
    p3 = allocate(s2, scheme=SCHEMES[4])
    check("并列同分给同一分位",
          p3.loc[:499, "pct"].nunique() == 1 and p3.loc[500:, "pct"].nunique() == 1)
    check("无并列时分位覆盖 (0,1]", abs(allocate(np.arange(100.)).loc[99, "pct"] - 1.0) < 1e-9)

    chk = np.zeros(10, bool); chk[0] = True
    p4 = allocate(np.arange(10.), blocked=chk, rankable=np.ones(10, bool))
    check("blocked 优先于 rankable", p4.loc[0, "tier"] == -1)
    check("summarize 结构完整", set(summarize(p4)) ==
          {"n", "blocked", "ranked", "paid", "points_total", "tiers"})


def test_scheme_selection():
    print("\n[5] 分档方案选择")
    check("小队列 3 档", pick_scheme(50).n == 3)
    check("中队列 4 档", pick_scheme(150).n == 4)
    check("大队列 5 档", pick_scheme(500).n == 5)
    check("边界 100 -> 4 档", pick_scheme(100).n == 4)
    check("边界 250 -> 5 档", pick_scheme(250).n == 5)
    for k, sc in SCHEMES.items():
        ok = (sc.edges[0] == 0.0 and sc.edges[-1] == 1.0
              and all(a < b for a, b in zip(sc.edges, sc.edges[1:], strict=False))
              and len(sc.points) == sc.n and sc.points[0] == 0)
        check(f"{k} 档方案自洽 (获奖 {sc.award:.0%})", ok)


def test_persistence():
    print("\n[6] 单文件持久化")
    d = _synthetic(n=1200, seed=3)
    r = Ranker(objective={"reply": 1.0}, context=["author_prior"],
               content=["author_level"], text=("title", "text"),
               time_col="create_time", train_end="2022-12-31", verbose=False)
    r.fit(d)
    with tempfile.TemporaryDirectory() as tmp:
        p = str(Path(tmp) / "m.pkl")
        r.save(p)
        mb = Path(p).stat().st_size / 1e6
        r2 = Ranker.load(p)
        a, b = r.score(d), r2.score(d)
        check("载入后打分完全一致", np.allclose(a, b))
        check("残差完全一致", np.allclose(r.residual(d), r2.residual(d)))
        check("manifest 哈希可验证", verify_manifest(p)["ok"])
        check("工件 < 5 MB", mb < 5.0, f"{mb:.3f} MB")
    check("explain 返回词与系数", len(r.explain(d, top=5)) == 5)


def test_ranker_edges():
    print("\n[7] 边界情况")
    d = _synthetic(n=800, seed=4)
    r = Ranker(objective={"reply": 1.0}, text=("title", "text"), verbose=False)
    r.fit(d)                                   # 无 context、无 content、无时间列
    check("无上下文时退化为不去混杂", r.residual(d).shape == (800,))
    check("无上下文仍可打分", r.score(d).shape == (800,))

    d2 = d.copy(); d2["reply"] = np.nan
    r2 = Ranker(objective={"reply": 1.0}, context=["author_prior"],
                content=["author_level"], text=("title", "text"), verbose=False)
    r2.fit(d2)
    check("目标全 NaN 不崩", np.isfinite(r2.residual(d2)).all())

    r3 = Ranker(objective={"reply": 1.0}, content=["author_level"], verbose=False)
    r3.fit(d)
    check("只有数值特征也能跑", r3.score(d).shape == (800,))

    d3 = d.copy()
    for c in ("title", "text"):
        d3[c] = ""
    r4 = Ranker(objective={"reply": 1.0}, content=["author_level"], verbose=False)
    r4.fit(d3)
    check("文本全空不崩", np.isfinite(r4.score(d3)).all())

    # 默认宽松模式保留通用包兼容性；Tieba profile 使用 strict=True。
    r5 = Ranker(objective={"reply": 1.0, "不存在的列": 5.0},
                context=["author_prior", "不存在的列"],
                content=["author_level"], text=("title", "text"), verbose=False)
    r5.fit(d)
    check("缺失的列被跳过", np.isfinite(r5.residual(d)).all())


def test_safety_contract():
    print("\n[8] 时间安全与严格契约")
    d = _synthetic(n=240, seed=8)
    d["create_time"] = pd.to_datetime(
        ["2022-12-31 12:00:00+00:00"] * 120
        + ["2023-01-01 00:00:00+00:00"] * 120,
        utc=True,
    )
    r = Ranker(objective={"reply": 1.0}, context=["author_prior"],
               content=["author_level"], text=("title", "text"),
               time_col="create_time", train_end="2022-12-31",
               strict=True, verbose=False)
    r.fit(d)
    check("日期截止含整天", r.diag_["n_train"] == 120, str(r.diag_))
    rejected = False
    try:
        r.fit(d.drop(columns=["author_prior"]))
    except ValueError:
        rejected = True
    check("严格训练契约拒绝缺列", rejected)
    preview = d[["title", "text", "author_level"]]
    check("预排名不需要结算字段", len(r.score(preview)) == len(preview))
    try:
        r.score(d[["title", "text"]])
        check("严格预测契约拒绝缺列", False)
    except ValueError:
        check("严格预测契约拒绝缺列", True)


def test_speed():
    print("\n[9] 速度")
    import time
    d = _synthetic(n=20000, seed=5)
    r = Ranker(objective={"reply": 1.0}, context=["author_prior"],
               content=["author_level"], text=("title", "text"), verbose=False)
    r.fit(d)
    t = time.time(); r.score(d); dt = time.time() - t
    print(f"       打分 {len(d)/dt:,.0f} 条/秒")
    check("打分 > 2000 条/秒", len(d) / dt > 2000, f"{len(d)/dt:,.0f}/s")


TESTS = (
    test_text_contract,
    test_debias_recovers_quality,
    test_distiller_standalone,
    test_allocate,
    test_scheme_selection,
    test_persistence,
    test_ranker_edges,
    test_safety_contract,
    test_speed,
)


def main() -> int:
    print("=" * 68)
    print("ResRank 自检")
    print("=" * 68)
    for fn in TESTS:
        try:
            fn()
        except AssertionError:
            pass  # check() 已记录失败，继续跑完其余用例
    print("\n" + "=" * 68)
    print(f"通过 {len(_OK)} / {len(_OK) + len(_FAIL)}")
    if _FAIL:
        print("失败:")
        for f in _FAIL:
            print("  -", f)
    print("=" * 68)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
