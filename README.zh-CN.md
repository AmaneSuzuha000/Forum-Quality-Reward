# 内容质量激励算法

[![tests](https://github.com/AmaneSuzuha000/forum-quality-reward/actions/workflows/tests.yml/badge.svg)](https://github.com/AmaneSuzuha000/forum-quality-reward/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**forum-quality-reward** —— 面向长周期论坛的内容质量评分与分档发放。

*[English](README.md) · 中文*

一段二十年的论坛历史里，一句「顶」和一篇万字长文，在数据库里长得一模一样。这个项目做的事情就是把它们分开，并且回答一个问题：**哪些内容值得被奖励。**

> **开始之前**
>
> - **本仓库不含任何数据**，只提供方法与可运行的实现。跑通管线需要自备一份已归档的 PostgreSQL 库。
> - **不需要数据也能验证算法**：[`examples/demo_reward_round.py`](examples/demo_reward_round.py) 自己造合成数据，几秒钟跑完。
> - **环境**：Python **3.11+**、PostgreSQL **14+**。
> - **只在单一论坛上验证过**，跨吧通用性未经证明；权重与阈值不是普适常数。详见 [§7 已知限制](#7-已知限制)。
> - 数据获取需自行确保符合平台服务条款与当地法律。

## 目录

- [1. 要解决的问题](#1-要解决的问题)
- [2. 方法](#2-方法)
- [3. 结构](#3-结构)
- [4. 快速开始](#4-快速开始)
- [5. 数据要求](#5-数据要求)
- [6. 验证](#6-验证)
- [7. 已知限制](#7-已知限制)
- [8. 许可](#8-许可)

---

## 1. 要解决的问题

「按数据发奖」几乎总是滑向同一个失败模式：**谁回复多、谁活跃、谁来得早就奖谁。** 这会带来三个具体的错误：

| 错误 | 表现 |
| --- | --- |
| 奖励求助 | 求助帖天然回复多，被当成优质内容 |
| 奖励灌水 | 极短自我回复（顶帖）刷高互动计数 |
| 奖励身份 | 老用户历史数据多，新人永远排不过 |

这个项目的全部设计都围绕绕开这三件事。

---

## 2. 方法

### 2.1 三个信号

只看**楼主自己的回复**，不掺入他人互动：

| 信号 | 判据 |
| --- | --- |
| 顶帖 | 去空白后 ≤ 5 字符 —— 判定为灌水 |
| 创作层 | ≥ 15 字符，按主题取 `nunique()` —— 贡献了几段**不同的**长文 |
| 深度 | ≥ 30 字符，同样取 `nunique()` |

取 `nunique()` 而不是 `count()` 是关键：同一段话反复发不算创作。

顶帖占楼主自我回复的 **36.5%**，但判别力只有 **AUC 0.5304**（接近抛硬币）。把它单独建模并作**负向**权重后，内容质量判别力从 0.7822 升到 **0.8041**，同时作者身份对结果的影响从 0.2727 降到 **0.2131**。

### 2.2 事实效用分

互动与创作强度的加权线性组合，**创作层的权重高于任何互动项**，纯长度权重为 0（长 ≠ 好）。所有计数项进入前取 `log1p` 压制极值。

### 2.3 完整性闸门

闸门回答「够不够格被发放」，与质量分**分开**，不做乘法混合：

```
integrity = 1 − ( 0.50·重复项 + 0.30·自我回复占比 + 0.30·低努力 )
```

低于 `INTEGRITY_GATE = 0.60` 不进发放池。另有独立的**顶帖刷屏硬拦截**（顶帖占比 > 0.8 且原创长文段数 ≤ 2），实测命中 21 例、**0 例误伤**。

低努力门采用 **AND 口径**（短正文 **且** 短标题 **且** 无媒体）—— 改成 OR 会一次性误杀大量标题短但正文完整的正常帖。

### 2.4 分档发放

边界是**队列内分位点**，不是绝对分数 —— 奖励总额自动跟随队列规模，不需要周期性重调预算，也不会因发帖量暴涨而超支。按队列规模自动选粒度：

| 队列规模 | 边界 | 积分 |
| --- | --- | --- |
| n ≥ 250 | 0.80 / 0.90 / 0.96 / 0.99 | 10 / 30 / 60 / 100 |
| 100 ≤ n < 250 | 0.70 / 0.85 / 0.95 | 5 / 20 / 60 |
| 30 ≤ n < 100 | 0.70 / 0.85 | 10 / 30 |

百分位**只在可排名子集内计算**，被拦截的帖子不参与，否则获奖比例会被无效数据稀释。

> 切点位置的实测取舍：把切点上移可把「每千分捕获的有效创作数」提高数倍，但**在同等发放总额下几乎无收益**（最优 2.372 对现况 2.183）。因此保持较宽覆盖，把区分度交给分值梯度而非门槛。

完整的权重表、阈值推导与消融结果见 [`docs/reward_system.md`](docs/reward_system.md)。

---

## 3. 结构

```
forum-quality-reward/
├── apps/
│   └── tieba_reward.py          生产入口：设计矩阵、闸门、结算
├── resrank/                     评分与分档模块
│   ├── config.py                    分档方案（按队列规模自适应）
│   ├── features.py                  文本与媒体特征
│   ├── signal.py                    信号构造
│   ├── allocate.py                  分档与积分分配
│   ├── ranker.py                    排序模型
│   ├── artifact.py                  产物清单与校验
│   ├── distill.py                   蒸馏
│   └── runtime.py                   配置、连接、输出编码
├── scripts/reward/              数据管线与评测
│   ├── apply_sql.py                 建立分析视图
│   ├── build_features.py            构建互动特征
│   ├── build_self_content.py        构建楼主自回复特征（三信号）
│   ├── build_window_features.py     7 天观察窗口特征
│   ├── build_residual.py            去混杂残差目标（L1+L2）
│   ├── integrate_engagement.py      并入互动指标与官方标注
│   ├── score_posts.py               打分
│   ├── make_leaderboard.py          周榜与分档发放
│   ├── weekly_eval.py / weekly_tier_check.py
│   ├── fetch_recent.py / fetch_batch_posts.py / scrape_engagement.py
│   ├── audit/                       评测：盲标、独立标注、头对头、政策审计
│   └── sql/thread_engagement.sql
├── examples/
│   └── demo_reward_round.py     端到端演示（不依赖任何数据）
├── tests/
│   └── test_resrank.py          算法自检（不依赖外部数据，几秒跑完）
├── docs/
│   ├── reward_system.md         评分体系的权重与阈值推导
│   └── pipeline.md              数据管道与文件契约
├── .github/workflows/tests.yml  CI
└── config/config.example.toml
```

---

## 4. 快速开始

### 先跑示例（不需要数据，约 30 秒）

```bash
pip install -r requirements.txt
python examples/demo_reward_round.py
```

脚本自己造一批**真实质量与作者名气无关、但互动量被名气污染**的帖子，跑完整四步并打印获奖名单。
在 900 篇合成帖上的实测输出：

| | 朴素（按互动量排） | 本算法 |
| --- | --- | --- |
| 获奖者平均真质量 | +0.480 | **+0.861** |
| 获奖者平均作者名气 | +0.522 | **−0.424** |
| 两份名单重合 | — | 224 / 361 |

### 在真实数据上跑

需要一个只读的 PostgreSQL 归档库。完整链路（每步读什么、写什么）见 [`docs/pipeline.md`](docs/pipeline.md)。

### 环境

Python **3.11+**，PostgreSQL **14+**（存已归档的帖子数据）。

```bash
git clone https://github.com/AmaneSuzuha000/forum-quality-reward.git
cd forum-quality-reward

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 配置

```bash
cp config/config.example.toml config/config.toml
cp .env.example .env
```

生产环境建议只把凭据放在 `.env`，`config.toml` 留空密码。

### 运行

管线是有序的，逐级产出下一级的输入。主链路（`fid` 由配置文件决定，SQL 里没有硬编码吧号）：

```bash
python scripts/reward/apply_sql.py scripts/reward/sql/thread_engagement.sql  # 建分析视图
python scripts/reward/build_features.py         # 互动特征
python scripts/reward/build_self_content.py     # 楼主自回复三信号
python scripts/reward/build_window_features.py  # 7 天窗口指标
python scripts/reward/build_residual.py --weekly  # 去混杂残差目标（L1+L2）
python apps/tieba_reward.py                     # 训练打分器 + 出获奖方案
```

完整的输入输出契约见 [`docs/pipeline.md`](docs/pipeline.md)。其中说明了两个**没有生产者的中间文件**
（`reward_scores.parquet`、`board_creation.parquet`）——它们由上游其他部分产出，本仓库不含写出它们的代码；
`score_posts.py` 与 `make_leaderboard.py` 依赖它们，评估类脚本同理。主链路不受影响。

### 自检

```bash
pytest -q     # 14 个测试，不依赖数据库
```

自检的核心用例是 `test_debias_recovers_quality`：构造一批**真实质量与作者无关**、但互动量被作者曝光度污染的对象，验证算法能把排序从「谁有名谁第一」纠正回「谁写得好谁第一」。这是整套方法成立的最小证明。

---

## 5. 数据要求

**本仓库不含数据。** 需要自备一份已归档的 PostgreSQL 库，至少包含：

- 一张**楼层表**（帖子 id、楼层 id、作者 id、正文、发布时间）
- 一张**主题表**（帖子 id、标题、发布时间）
- 一张**互动表**（回复数、评论数、参与人数、参与天数）

字段清单以 `scripts/reward/build_features.py` 与 `scripts/reward/sql/thread_engagement.sql` 为准 —— 代码是唯一权威，本文件不重复枚举以免漂移。

数据获取需自行确保符合平台服务条款与当地法律，并尊重用户隐私。

---

## 6. 验证

| 指标 | 结果 | 说明 |
| --- | --- | --- |
| 样本外内容质量判别 AUC | **0.9595** | 按时间切分，非随机切分 |
| 随机标签负对照 | 0.5207 | 打乱标签后回到随机水平，排除流程泄漏 |
| 未训练下界 | 0.4739 | 模型必须显著高于此线 |
| 作者身份消融 | Spearman 0.99783，前 10 名 10/10 | 移除作者特征后排序几乎不变 → 该层对最终排序无贡献 |

残差构造（L2）在本仓库代码上的重新实测、以及两种「仅作者身份」口径的差异，见
[`docs/reward_system.md` §6](docs/reward_system.md)。

评测协议三条纪律：**按时间切分**（随机切分会造成乐观偏差）、**必设随机标签负对照**、**必报未训练下界**。

> **关于可复现性。** 上表的 AUC 与消融数字来自原始工程的历史评测记录，本仓库保留了
> 产生它们的脚本（`scripts/reward/audit/`），但**发布副本上未重新跑过** —— 因为评测需要
> `pairs_train_weekly.parquet` 等中间文件（见 [`docs/pipeline.md`](docs/pipeline.md)）。
> 只有残差构造那一组是我们重新实测的。以未验证的数字冒充已验证是不诚实的，故在此明说。

---

## 7. 已知限制

1. **相关不是因果。** 本体系度量「什么样的帖子更可能被公认为优质」，**不证明**「发什么样的帖子能导致被公认为优质」。
2. **在单一论坛上验证。** 全部权重与阈值来自一个吧的数据，跨吧通用性未被证明。
3. **信号偏向可见行为。** 只能看到平台暴露的计数与文本，「写得好但无人回复」的内容无法被充分识别。
4. **依赖已有数据。** 没有归档数据时管线无法端到端跑通；本仓库提供方法与实现，不是开箱即用的数据集。

---

## 8. 许可

[MIT](LICENSE)
