## 改了什么

<!-- 一句话说清动机 -->

## 为什么

<!-- 现在的做法错在哪，或哪里不够好 -->

## 怎么验证的

<!-- 贴实际命令与输出，不要只写「测试通过」 -->

- [ ] `pytest -q` 全绿
- [ ] `python examples/demo_reward_round.py` 正常跑完
- [ ] 若改了权重 / 阈值 / 切点，已附实测证据（见 CONTRIBUTING.md）

## 注意

评分逻辑的改动会影响**已发放**的奖励结果，属于破坏性变更。若改动
`resrank/config.py` 的 `SCHEMES`、`apps/tieba_reward.py` 的 `OBJECTIVE`
或完整性闸门常量，请在描述里明确标注。
