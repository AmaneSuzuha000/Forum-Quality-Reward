# -*- coding: utf-8 -*-
"""周榜生成 —— 读取内容特征表，计算质量分，过完整性闸门，按队列内分位分档发放。

输入:
    data/reward/board_creation.parquet   由数据构建步骤产出的按帖特征表
输出:
    data/reward/leaderboard.parquet
    data/reward/leaderboard.html

用法:
    python scripts/reward/make_leaderboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resrank.runtime import ROOT, setup_stdout

setup_stdout()

import html as _html

import numpy as np
import pandas as pd

from resrank.config import pick_scheme

SRC = ROOT / "data" / "reward" / "board_creation.parquet"
OUT_PARQUET = ROOT / "data" / "reward" / "leaderboard.parquet"
OUT_HTML = ROOT / "data" / "reward" / "leaderboard.html"

# 媒体项权重：在实测曲线上取的拐点，用于回报「以图叙事」的记录帖
W_IMG = 0.3
# 完整性闸门，与 apps/tieba_reward.py 保持一致
INTEGRITY_GATE = 0.60
# 顶帖刷屏判定：顶帖占比过高且几乎没有原创长文
BUMP_RATIO_MAX = 0.8
LONG_UNIQ_MIN = 2

TIER_COLOR = {3: ('#5f4415', '#ffd166'), 2: ('#3d2a5f', '#c39bff'), 1: ('#1e3a5f', '#7cc4ff')}

CSS = (
    '*{box-sizing:border-box}'
    'body{margin:0;background:#0f1116;color:#e6e9ef;'
    'font:14px/1.55 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}'
    '.w{max-width:1200px;margin:0 auto;padding:28px 20px 60px}'
    'h1{font-size:20px;margin:0 0 6px}'
    '.s{color:#8b93a7;font-size:13px;margin-bottom:16px}'
    'table{width:100%;border-collapse:collapse;background:#171a21;'
    'border:1px solid #262b36;border-radius:10px;overflow:hidden}'
    'thead th{position:sticky;top:0;background:#1b1f28;color:#8b93a7;font-size:12px;'
    'text-align:left;padding:10px;border-bottom:1px solid #262b36}'
    'tbody td{padding:8px 10px;border-bottom:1px solid #1e232d}'
    'tbody tr:hover{background:#1b2029}'
    'td.c{text-align:center;white-space:nowrap}'
    'td.n{font-variant-numeric:tabular-nums;color:#c3cad8}'
    'td.n.p{color:#7ee787;font-weight:650}'
    '.b{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:650}'
    '.tt{font-weight:600}'
)


def compute(d: pd.DataFrame):
    d = d.copy()
    d['n_image'] = pd.to_numeric(d.get('n_image', 0), errors='coerce').fillna(0.0)
    d['integrity'] = pd.to_numeric(d['integrity'], errors='coerce').fillna(0.0)

    if 'flood' in d.columns:
        d['flood'] = pd.to_numeric(d['flood'], errors='coerce').fillna(0).astype(int)
    else:
        bump_ratio = pd.to_numeric(d.get('bump_ratio', 0), errors='coerce').fillna(0.0)
        long_col = 'self_long_uniq' if 'self_long_uniq' in d.columns else 'op_long_uniq'
        long_uniq = pd.to_numeric(d.get(long_col, 0), errors='coerce').fillna(0.0)
        d['flood'] = ((bump_ratio > BUMP_RATIO_MAX) & (long_uniq <= LONG_UNIQ_MIN)).astype(int)

    d['score'] = (pd.to_numeric(d['score_new'], errors='coerce')
                  + W_IMG * np.log1p(d['n_image'].clip(lower=0)))
    d = d[d['score'].notna()].copy()
    d['rankable'] = (d['flood'] == 0) & (d['integrity'] >= INTEGRITY_GATE)
    d['rank'] = d['score'].rank(ascending=False, method='first').astype(int)

    # 百分位只在可排名子集内计算：被闸门拦截的帖子不稀释获奖比例
    d['pct'] = np.nan
    d.loc[d['rankable'], 'pct'] = d.loc[d['rankable'], 'score'].rank(pct=True)

    scheme = pick_scheme(int(d['rankable'].sum()))
    tier = np.zeros(len(d), dtype=int)
    tier[d['rankable'].to_numpy()] = scheme.tier(d.loc[d['rankable'], 'pct'].to_numpy())
    d['tier'] = np.where(d['rankable'], tier, -1)
    d['points'] = np.where(d['rankable'], np.asarray(scheme.points)[np.clip(tier, 0, None)], 0).astype(int)
    return d.sort_values('rank'), scheme


def write_html(out: pd.DataFrame, scheme) -> None:
    head = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>内容质量周榜</title><style>' + CSS + '</style></head><body>'
    )
    paid = out[out['points'] > 0]
    summary = (
        '<div class="w"><h1>内容质量周榜</h1>'
        '<div class="s">队列 ' + str(len(out)) + ' 篇 · 可排名 ' + str(int(out['rankable'].sum()))
        + ' 篇 · 获奖 ' + str(len(paid)) + ' 篇 · 合计 ' + str(int(out['points'].sum())) + ' 分 · '
        + str(scheme.n) + ' 档（边界 ' + str(tuple(scheme.edges)) + '，积分 ' + str(tuple(scheme.points)) + '）</div>'
    )
    rows = []
    for _, x in out.iterrows():
        t = int(x['tier'])
        if t in TIER_COLOR:
            bg, fg = TIER_COLOR[t]
            badge = '<span class="b" style="background:' + bg + ';color:' + fg + '">T' + str(t) + '</span>'
        elif t == -1:
            badge = '<span class="b" style="background:#3a2626;color:#ff7b72">拦截</span>'
        else:
            badge = '<span class="b" style="background:#333846;color:#aab2c4">T0</span>'
        cls = 'n p' if x['points'] > 0 else 'n'
        rows.append(
            '<tr><td class="c n">' + str(int(x['rank'])) + '</td>'
            + '<td class="c">' + badge + '</td>'
            + '<td class="c ' + cls + '">' + str(int(x['points'])) + '</td>'
            + '<td class="c n">' + format(float(x['score']), '.2f') + '</td>'
            + '<td class="tt">' + _html.escape(str(x['title']), quote=True) + '</td></tr>'
        )
    table = (
        '<table><thead><tr><th class="c">名次</th><th class="c">档位</th><th class="c">积分</th>'
        '<th class="c">分值</th><th>标题</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
    )
    OUT_HTML.write_text(head + summary + table + '</div></body></html>', encoding='utf-8')


def main() -> int:
    if not SRC.exists():
        print('缺少输入文件 ' + str(SRC))
        print('请先运行数据构建步骤产出 board_creation.parquet。')
        return 2
    d, scheme = compute(pd.read_parquet(SRC))
    out = d[['tid', 'title', 'rank', 'tier', 'points', 'score', 'pct', 'integrity', 'n_image']]
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    write_html(d, scheme)

    paid = out[out['points'] > 0]
    print('队列 ' + str(len(out)) + ' 篇 | 可排名 ' + str(int(d['rankable'].sum()))
          + ' | 获奖 ' + str(len(paid)) + ' 篇 | 合计 ' + str(int(out['points'].sum())) + ' 分')
    print('分档方案 ' + str(scheme.n) + ' 档，边界 ' + str(tuple(scheme.edges)) + '，积分 ' + str(tuple(scheme.points)))
    for t in range(1, scheme.n):
        print('  T' + str(t) + '  ' + str(scheme.points[t]) + ' 分 × ' + str(int((out['tier'] == t).sum())) + ' 篇')
    print()
    for _, x in paid.head(15).iterrows():
        print('  #' + str(int(x['rank'])).rjust(3) + '  T' + str(int(x['tier'])) + '  '
              + str(int(x['points'])).rjust(3) + '分  ' + str(x['title'])[:38])
    print()
    print('WROTE ' + OUT_PARQUET.name + ' + ' + OUT_HTML.name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
