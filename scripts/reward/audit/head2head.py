# -*- coding: utf-8 -*-
"""决定性对照: 同一评测集、同一指标、同一特征。"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.sparse import vstack as svstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

R = Path(r'str(Path(__file__).resolve().parents[3])'); D = R / 'data' / 'reward'
sys.path.insert(0, str(R))
from resrank import Ranker
from resrank.distill import build_docs

d = pd.read_parquet(D / 'thread_residual_weekly.parquet').reset_index(drop=True)
pr = pd.read_parquet(D / 'pairs_train_weekly.parquet').reset_index(drop=True)
yrv = pd.to_datetime(d['create_time'], utc=True).dt.year.to_numpy()
YM = dict(zip(d['tid'].astype(int), yrv, strict=False))
row = {int(t): j for j, t in enumerate(d['tid'])}
p = pr[pr['tid_a'].isin(row) & pr['tid_b'].isin(row)].reset_index(drop=True)
ya = p['tid_a'].map(YM).to_numpy(int); yb = p['tid_b'].map(YM).to_numpy(int)
k = ya == yb; p = p[k].reset_index(drop=True); pyr = ya[k]
ia = p['tid_a'].map(row).to_numpy(int); ib = p['tid_b'].map(row).to_numpy(int)
def acc(s, Y=None):
    m = np.ones(len(pyr), bool) if Y is None else (pyr == Y)
    return float(np.mean(s[ia[m]] > s[ib[m]]))
def line(nm, s, mb):
    print('  %-26s val %.4f | 2024 %.4f | 2025 %.4f | 2026 %.4f | 2024+ %.4f | %.3f MB' % (
          nm, acc(s, 2023), acc(s, 2024), acc(s, 2025), acc(s, 2026),
          float(np.mean(s[ia[pyr >= 2024]] > s[ib[pyr >= 2024]])), mb))
    return float(np.mean(s[ia[pyr >= 2024]] > s[ib[pyr >= 2024]]))
print('评测集: %d 对 (val %d / 2024 %d / 2025 %d / 2026 %d)' % (
      len(pyr), (pyr==2023).sum(), (pyr==2024).sum(), (pyr==2025).sum(), (pyr==2026).sum()))
print('')
print('=== [1] resrank (四层, 稀疏 TF-IDF + Ridge) ===')
r = Ranker.load(str(R / 'models' / 'reward' / 'resrank.pkl'))
a1 = line('resrank', r.score(d), (R / 'models' / 'reward' / 'resrank.pkl').stat().st_size / 1048576)
print('')
print('=== [2] 旧方案的成对目标 (同特征, 同评测集) ===')
FC = r.content
docs = build_docs(d['title'].fillna('').tolist(), d['text'].fillna('').tolist())
M = d[[c for c in FC if c in d.columns]].apply(pd.to_numeric, errors='coerce').astype(float).fillna(0).to_numpy(np.float32)
tr = yrv <= 2022
tf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,3), min_df=10, max_features=8000, sublinear_tf=True, dtype=np.float32)
tf.fit([docs[i] for i in np.flatnonzero(tr)])
X = hstack([tf.transform(docs), csr_matrix((M-M[tr].mean(0))/np.where(M[tr].std(0)==0,1,M[tr].std(0)))]).tocsr()
A = X[ia] - X[ib]; Atr = A[pyr <= 2022]
Aa = svstack([Atr, -Atr]).tocsr(); yy = np.r_[np.ones(Atr.shape[0]), np.zeros(Atr.shape[0])]
best = None
for C in (0.01, 0.03, 0.1, 0.3, 1.0):
    lr = LogisticRegression(C=C, solver='liblinear', max_iter=3000).fit(Aa, yy)
    s = np.asarray(X @ lr.coef_.ravel()).ravel()
    v = acc(s, 2023)
    if best is None or v > best[0]: best = (v, C, s, lr)
print('  (C=%.2f, 训练对 %d)' % (best[1], Atr.shape[0]))
a2 = line('成对 Bradley-Terry', best[2], len(__import__('pickle').dumps(best[3])) / 1048576 + 0.248)
print('')
print('=== [3] 结论 ===')
print('  resrank 比成对目标: %+.4f' % (a1 - a2))
print('  注意: 成对训练只覆盖 %d/%d = %.1f%% 的训练帖' % (
      len(set(ia[pyr<=2022]) | set(ib[pyr<=2022])), tr.sum(),
      100.0 * len(set(ia[pyr<=2022]) | set(ib[pyr<=2022])) / tr.sum()))
