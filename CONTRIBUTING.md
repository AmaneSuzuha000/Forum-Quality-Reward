# 贡献指南

## 环境

只需要 Python 3.11+。**跑测试和示例不需要数据库、不需要任何外部数据。**

    python -m venv .venv
    # Windows: .venv\Scripts\activate    其他: source .venv/bin/activate
    pip install -r requirements.txt
    pip install pytest ruff

## 跑验证

    pytest -q                              # 14 个测试，全部用合成数据
    python examples/demo_reward_round.py   # 端到端演示

## 改评分逻辑前先读

这套系统的每一条权重、每一个阈值都有实测依据。提交改动时请附上：

1. **改之前 / 改之后的数字**，以及是在什么数据上测的。
2. **若改的是阈值或切点**，说明为什么这个位置更好，而不只是「看起来更合理」。
3. **负对照**。任何声称「指标变好了」的结论，都要同时报打乱标签后的结果
   与未训练下界 —— 否则无法排除流程泄漏。

### 两条硬规矩

- **不许引入目标泄漏。** 目标里的加项**不得**出现在模型的 `CONTENT` 特征里。
  `build_residual.py` 里有断言守着这条，`CONTRIBUTING` 不接受绕过它。
  背景：`len_text` 本身是模型特征，把长度放进目标，测到的就不是模型能力，
  而是目标泄漏。
- **纯长度权重必须保持 0。** 长度只作为分段阈值使用。

## 提交前

    pytest -q
    ruff check .

**不要提交任何数据文件。** `.gitignore` 已排除 `*.parquet`、`data/`、
`config/config.toml` 与 `.env`。请确认没有把数据库口令带进去。
