# Content-Quality Reward Algorithm

[![tests](https://github.com/AmaneSuzuha000/forum-quality-reward/actions/workflows/tests.yml/badge.svg)](https://github.com/AmaneSuzuha000/forum-quality-reward/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**forum-quality-reward** — content-quality scoring and tiered reward allocation for long-horizon forums.

*Read this in [中文](README.zh-CN.md).*

Across twenty years of forum history, a one-character "bump" and a ten-thousand-word essay
look identical in the database. This project separates them, and answers one question:
**which content deserves to be rewarded?**

> **Before you start**
>
> - **This repository contains no data.** It ships the method and a runnable implementation only.
>   Running the full pipeline requires your own archived PostgreSQL database.
> - **You do not need data to verify the algorithm:** [`examples/demo_reward_round.py`](examples/demo_reward_round.py)
>   generates its own synthetic posts and finishes in seconds.
> - **Requirements:** Python **3.11+**, PostgreSQL **14+**.
> - **Validated on a single forum only.** Cross-forum generality is unproven; the weights and
>   thresholds are not universal constants. See [§7 Known limitations](#7-known-limitations).
> - You are responsible for ensuring data collection complies with the platform's terms of
>   service and applicable law.

## Table of contents

- [1. The problem](#1-the-problem)
- [2. Method](#2-method)
- [3. Layout](#3-layout)
- [4. Quick start](#4-quick-start)
- [5. Data requirements](#5-data-requirements)
- [6. Validation](#6-validation)
- [7. Known limitations](#7-known-limitations)
- [8. License](#8-license)

---

## 1. The problem

"Reward people by the numbers" almost always slides into the same failure mode:
**reward whoever gets the most replies, whoever is most active, whoever arrived first.**
That produces three specific errors:

| Error | Symptom |
| --- | --- |
| Rewards help-seeking | Requests for help attract replies naturally and get counted as quality |
| Rewards flooding | Very short self-replies (bumps) inflate engagement counters |
| Rewards identity | Long-time users have more history; newcomers can never outrank them |

Every design decision here exists to avoid those three things.

---

## 2. Method

### 2.1 Three signals

Only the **original poster's own replies** are examined — other people's engagement is not mixed in:

| Signal | Criterion |
| --- | --- |
| Bump | ≤ 5 characters after stripping whitespace — treated as flooding |
| Creation tier | ≥ 15 characters, counted with `nunique()` per thread — how many **distinct** long-form pieces |
| Depth | ≥ 30 characters, likewise `nunique()` |

Using `nunique()` rather than `count()` is the crux: posting the same paragraph repeatedly is not creation.

Bumps account for **36.5%** of OP self-replies but carry a discriminative power of only **AUC 0.5304**
— essentially a coin flip. Modelling them separately with a **negative** weight raises content-quality
discrimination from 0.7822 to **0.8041**, while the influence of author identity falls from 0.2727 to **0.2131**.

### 2.2 Utility score

A weighted linear combination of engagement and creation intensity. **The creation tier outweighs every
engagement term**, and raw length carries weight 0 (long ≠ good). All count terms are passed through
`log1p` before entering, to suppress head outliers.

### 2.3 Integrity gate

The gate answers "is this eligible for payment at all", **separately** from the quality score —
no multiplicative mixing:

```
integrity = 1 − ( 0.50·duplication + 0.30·self-reply share + 0.30·low-effort )
```

Below `INTEGRITY_GATE = 0.60` a post does not enter the payout pool. A separate **bump-flooding
hard block** catches posts whose bump share exceeds 0.8 with at most 2 distinct long-form pieces;
in testing it fired on 21 posts with **zero false positives**.

The low-effort test uses an **AND** criterion (short body **and** short title **and** no media) —
switching it to OR wipes out large numbers of legitimate posts that merely have a short title.

### 2.4 Tiered allocation

Tier boundaries are **within-queue percentiles**, not absolute scores. The total payout therefore tracks
queue size automatically: no periodic budget re-tuning, and no overspend when posting volume spikes.
Granularity is chosen by queue size:

| Queue size | Boundaries | Points |
| --- | --- | --- |
| n ≥ 250 | 0.80 / 0.90 / 0.96 / 0.99 | 10 / 30 / 60 / 100 |
| 100 ≤ n < 250 | 0.70 / 0.85 / 0.95 | 5 / 20 / 60 |
| 30 ≤ n < 100 | 0.70 / 0.85 | 10 / 30 |

Percentiles are computed **only over the rankable subset**; blocked posts are excluded, otherwise the
award rate would be diluted by invalid data.

> **Measured trade-off on cut positions.** Raising the cuts multiplies "effective creation captured per
> thousand points", but yields **almost nothing at an equal total payout** (best 2.372 vs current 2.183).
> Coverage therefore stays wide, and discrimination is carried by the point gradient rather than the threshold.

Full weight tables, threshold derivations and ablations are in [`docs/reward_system.md`](docs/reward_system.md).

---

## 3. Layout

```
forum-quality-reward/
├── apps/
│   └── tieba_reward.py          Production entry: design matrix, gate, settlement
├── resrank/                     Scoring and tiering package
│   ├── config.py                    Tier schemes (adaptive to queue size)
│   ├── features.py                  Text and media features
│   ├── signal.py                    Signal construction
│   ├── allocate.py                  Tiering and point allocation
│   ├── ranker.py                    Ranking model
│   ├── artifact.py                  Artifact manifest and verification
│   ├── distill.py                   Distillation
│   └── runtime.py                   Config, connection, stdout encoding
├── scripts/reward/              Data pipeline and evaluation
│   ├── apply_sql.py                 Create analysis views
│   ├── build_features.py            Build engagement features
│   ├── build_self_content.py        Build OP self-reply features (three signals)
│   ├── build_window_features.py     7-day observation-window features
│   ├── build_residual.py            Deconfounded residual target (L1+L2)
│   ├── integrate_engagement.py      Merge engagement metrics and official labels
│   ├── score_posts.py               Scoring
│   ├── make_leaderboard.py          Weekly board and tiered payout
│   ├── weekly_eval.py / weekly_tier_check.py
│   ├── fetch_recent.py / fetch_batch_posts.py / scrape_engagement.py
│   ├── audit/                       Evaluation: blind labels, head-to-head, policy audit
│   └── sql/thread_engagement.sql
├── examples/
│   └── demo_reward_round.py     End-to-end demo (no data required)
├── tests/
│   └── test_resrank.py          Algorithm self-test (no external data, runs in seconds)
├── docs/
│   ├── reward_system.md         Weight and threshold derivations
│   └── pipeline.md              Pipeline and file contracts
├── .github/workflows/tests.yml  CI
└── config/config.example.toml
```

---

## 4. Quick start

### Run the demo first (no data needed, ~30 seconds)

```bash
pip install -r requirements.txt
python examples/demo_reward_round.py
```

The script generates posts whose **true quality is independent of author fame, while engagement is
contaminated by it**, runs all four steps, and prints the week's award list. Measured output over 900
synthetic posts:

| | Naive (rank by engagement) | This algorithm |
| --- | --- | --- |
| Mean true quality of awardees | +0.480 | **+0.861** |
| Mean author fame of awardees | +0.522 | **−0.424** |
| Overlap of the two lists | — | 224 / 361 |

### Running on real data

Requires a read-only PostgreSQL archive. The full chain — what each step reads and writes — is in
[`docs/pipeline.md`](docs/pipeline.md).

### Environment

Python **3.11+**, PostgreSQL **14+** (holding archived post data).

```bash
git clone https://github.com/AmaneSuzuha000/forum-quality-reward.git
cd forum-quality-reward

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

```bash
cp config/config.example.toml config/config.toml
cp .env.example .env
```

In production, keep credentials in `.env` only and leave the password blank in `config.toml`.

### Running

The pipeline is ordered; each step produces the next step's input. Main chain (`fid` comes from the
config file — no forum ID is hard-coded in the SQL):

```bash
python scripts/reward/apply_sql.py scripts/reward/sql/thread_engagement.sql  # analysis views
python scripts/reward/build_features.py         # engagement features
python scripts/reward/build_self_content.py     # OP self-reply signals
python scripts/reward/build_window_features.py  # 7-day window metrics
python scripts/reward/build_residual.py --weekly  # deconfounded residual (L1+L2)
python apps/tieba_reward.py                     # train scorer + emit award plan
```

The complete input/output contract is in [`docs/pipeline.md`](docs/pipeline.md). It documents two
**intermediate files with no producer** (`reward_scores.parquet`, `board_creation.parquet`): they are
emitted by other parts upstream and this repository does not contain the code that writes them.
`score_posts.py` and `make_leaderboard.py` depend on them, as do the evaluation scripts.
The main chain is unaffected.

### Self-test

```bash
pytest -q     # 14 tests, no database required
```

The core case is `test_debias_recovers_quality`: it constructs items whose **true quality is
independent of the author**, while engagement is contaminated by author exposure, then verifies the
algorithm corrects the ranking from "whoever is famous ranks first" back to "whoever writes well ranks
first". That is the minimal proof the method works.

---

## 5. Data requirements

**This repository ships no data.** You need your own archived PostgreSQL database containing at least:

- a **post-floor table** (thread id, floor id, author id, body, timestamp)
- a **thread table** (thread id, title, timestamp)
- an **engagement table** (reply count, comment count, participant count, active days)

The authoritative field list lives in `scripts/reward/build_features.py` and
`scripts/reward/sql/thread_engagement.sql` — the code is the single source of truth, and this file
deliberately does not enumerate fields to avoid drifting out of sync.

You are responsible for ensuring data collection complies with the platform's terms of service and
applicable law, and for respecting user privacy.

---

## 6. Validation

| Metric | Result | Note |
| --- | --- | --- |
| Out-of-sample content-quality discrimination AUC | **0.9595** | Time-based split, not random |
| Shuffled-label negative control | 0.5207 | Returns to chance, ruling out pipeline leakage |
| Untrained floor | 0.4739 | The model must sit clearly above this |
| Author-identity ablation | Spearman 0.99783, top 10 10/10 | Ranking barely moves when author features are removed → that layer contributes nothing to the final order |

For the residual construction (L2) re-measured on this repository's own code, and for the difference
between the two "author identity only" figures, see [`docs/reward_system.md` §6](docs/reward_system.md).

Three disciplines govern the evaluation protocol: **split by time** (random splits cause optimistic
bias), **always run a shuffled-label negative control**, and **always report the untrained floor**.

> **On reproducibility.** The AUC and ablation figures above come from the original project's historical
> evaluation records. This repository retains the scripts that produced them (`scripts/reward/audit/`),
> but they were **not re-run on this published copy**, because evaluation needs intermediate files such
> as `pairs_train_weekly.parquet` (see [`docs/pipeline.md`](docs/pipeline.md)). Only the residual
> construction group was re-measured. Passing off unverified numbers as verified would be dishonest,
> so it is stated plainly here.

---

## 7. Known limitations

1. **Correlation is not causation.** This system measures "what kind of post is more likely to be
   recognised as high quality"; it does **not** establish that "posting in a certain way causes" that
   recognition. Any causal reading of the score exceeds the available evidence.
2. **Validated on a single forum.** Every weight and threshold comes from one forum's data;
   cross-forum generality is unproven.
3. **Signals favour visible behaviour.** Only counters and text the platform exposes are visible;
   content that is well written but attracts no replies cannot be adequately recognised.
4. **Depends on existing data.** Without an archive the pipeline cannot run end to end. This repository
   provides a method and an implementation, not a ready-to-use dataset.

---

## 8. License

[MIT](LICENSE)
