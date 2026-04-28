# SCI 无真人评分实验设计

## 适用前提

当研究时间、伦理审批或招募条件不足时，可以不做人类参与者评分，但必须把论文设计成“自动评估 + 代理评估 + 稳健性分析”的形式，而不是把代理 judge 写成真人评价。

## 推荐三层结构

### 第一层：全量自动评估

- 100 产品
- 4 受众
- 400 任务
- 5 个主方法
- 4 个消融

这一层证明方法总体有效。

### 第二层：Checklist-based judge

让一个 judge 模型对每条广告做结构化检查，而不是只给一个含糊分数。

检查项：

- 标题是否清晰
- 卖点是否覆盖
- 是否符合目标受众
- CTA 是否有效
- 是否安全
- 文图设定是否一致
- 是否足够新颖

这一层提升可解释性。

### 第三层：Pairwise judge

让 judge 直接比较：

- `Ours` vs `B1`
- `Ours` vs `B2`
- `Ours` vs `B3`
- `Ours` vs `B0`

并做左右交换，统计顺序一致性。

这一层用于增强 judge 结果可信度。

## 创新点怎么写

无真人评分时，创新点应从“人更喜欢它”改写为：

1. 提出探索式广告创意优化框架，而非单轮生成框架；
2. 将受众建模、多目标评分和多样性保持整合到统一优化流程；
3. 设计面向广告创意的 checklist-based proxy evaluation；
4. 通过换位 pairwise judge 显式控制 judge 的位置偏差。

## 论文里要怎么表述

应使用：

- proxy evaluation
- judge-based evaluation
- predicted engagement
- judge-rated click intention

不要使用：

- human evaluation
- user preference
- real click improvement

## 什么时候算实验充分

至少包括：

- 全量自动评估
- 完整消融
- checklist judge
- pairwise judge
- order consistency
- audience-wise breakdown
- 失败案例分析

只有这样，无真人评分的版本才足够像一篇完整 SCI 论文。
