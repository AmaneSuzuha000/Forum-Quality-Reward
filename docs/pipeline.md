# 数据管道与文件契约

> **先读这一节。** 本仓库只包含**代码**，不包含任何数据。所有 \`data/reward/*.parquet\`
> 都要你自己产出。下表说明每一步读什么、写什么，以及**哪些输入没有生产者**。

## 完整链路

    PostgreSQL (归档库, 只读)
      │
      ├─ build_features.py ────────► thread_features.parquet      基础特征
      ├─ build_self_content.py ────► self_content.parquet         楼主创作量
      └─ build_window_features.py ─► window7d.parquet             7 天窗口指标
                                          │
      thread_features + window7d + self_content
                                          │
                          build_residual.py --weekly
                                          │
                                          ▼
                            thread_residual_weekly.parquet       去混杂残差目标
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
        apps/tieba_reward.py  (训练打分器)              weekly_eval.py (评估)
                  │
                  ▼
            模型 + 获奖方案

## 每步的输入输出

| 脚本 | 读 | 写 |
| --- | --- | --- |
| \`build_features.py\` | PostgreSQL \`public.thread\` / \`public.post\` 等 | \`thread_features.parquet\` |
| \`build_self_content.py\` | 同上（楼主自己的回复） | \`self_content.parquet\` |
| \`build_window_features.py\` | PostgreSQL（按 \`fid\` 过滤） | \`window7d.parquet\` |
| \`integrate_engagement.py\` | \`thread_features.parquet\` + \`reward_scores.parquet\` | \`thread_features_enriched.parquet\` |
| **\`build_residual.py --weekly\`** | \`thread_features[_enriched].parquet\` + \`window7d.parquet\` + \`self_content.parquet\` | \`thread_residual_weekly.parquet\` |
| \`apps/tieba_reward.py\` | \`thread_residual_weekly.parquet\` | 模型工件 + 获奖方案 |
| \`weekly_eval.py\` / \`weekly_tier_check.py\` | \`thread_residual_weekly.parquet\` + \`reward_scores_weekly.parquet\` | 报表 |
| \`audit/evaluate_resrank.py\` | \`thread_residual_weekly.parquet\` + \`pairs_train_weekly.parquet\` | 指标 |
| \`audit/head2head.py\` | \`thread_residual_weekly.parquet\` + \`pairs_train_weekly.parquet\` | 对比结果 |
| \`audit/policy_audit.py\` | \`--input\` 指定的特征表 | 策略审计报表 |

> 跑起来的最短路径：\`build_features\` → \`build_self_content\` → \`build_window_features\`
> → \`build_residual --weekly\` → \`apps/tieba_reward.py\`。其余脚本都是评估与审计，
> 不参与主链路，可以先不看。

## ⚠️ 两个没有生产者的输入

这两个文件被脚本读取，但**本仓库和原始工程里都没有写出它们的代码**。它们是上游
其他部分产出的。要完整复现评估环节，你需要自己提供，或跳过依赖它们的脚本：

| 文件 | 被谁读 | 需要什么 |
| --- | --- | --- |
| \`reward_scores.parquet\` | \`score_posts.py\`、\`integrate_engagement.py\`、\`weekly_eval.py\`、\`weekly_tier_check.py\` | \`tid\` + 打分类脚本产出的预测列 |
| \`reward_scores_weekly.parquet\` | \`weekly_eval.py\`、\`weekly_tier_check.py\` | 同上，周窗口口径 |
| \`pairs_train_weekly.parquet\` | \`audit/evaluate_resrank.py\`、\`audit/head2head.py\` | 成对偏好训练样本（\`tid_a\`/\`tid_b\`/标签） |
| \`board_creation.parquet\` | \`make_leaderboard.py\` | 按帖特征表，含 \`tid\` + 打分所需列 |

**主链路不受影响**——\`apps/tieba_reward.py\` 直接从 \`thread_residual_weekly.parquet\`
出方案，不需要 \`reward_scores*\`。这两个缺口只影响评估脚本。

## 需要的 PostgreSQL 表

\`scripts/reward/sql/thread_engagement.sql\` 定义了互动指标视图。建库后先跑：

    python scripts/reward/apply_sql.py scripts/reward/sql/thread_engagement.sql

SQL 里**没有硬编码吧号**，\`fid\` 是参数；\`config/config.toml\` 的 \`[report] fid\`
决定处理范围（\`0\` = 全部）。

## 数据规模参考

在验证用的单一论坛上：99,468 篇帖、时间跨度约 20 年、
\`thread_features.parquet\` 约 26 MB，\`thread_residual_weekly.parquet\` 约 30 MB。
窗口未满 7 天的帖子会被剔除（99,468 → 99,392）。
