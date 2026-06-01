# Local GPU Experiment Report

## Setup
- Local generator: `Qwen/Qwen2.5-7B-Instruct`
- Evaluation: automatic multi-objective metrics with full benchmark coverage
- Protocol: 5 primary methods, 4 ablations, offline automatic-evaluation design

## Main Findings
- Best primary method: `Ours_EAI_CO` with mean reward `0.8051`.
- `Ours_EAI_CO` mean predicted engagement: `0.8238`.
- `Ours_EAI_CO` mean latency: `11114.23` ms per task.
- `Ours_EAI_CO` median latency: `11358.50` ms per task.

## Audience Breakdown
- family_users: reward `0.8147`, audience fit `0.5125`, predicted engagement `0.8262`.
- price_sensitive_consumers: reward `0.7908`, audience fit `0.4258`, predicted engagement `0.8233`.
- students: reward `0.8098`, audience fit `0.5125`, predicted engagement `0.8237`.
- young_professionals: reward `0.8050`, audience fit `0.4366`, predicted engagement `0.8222`.

## Statistical Comparison vs Ours
- vs `B0_Template`: reward delta `0.1195` (95% CI [0.1086, 0.1301], p=3.72e-23, adjusted p=1.488e-22, Cohen's d=3.3831), engagement delta `0.1668` (p=1.31e-28), audience-fit delta `0.0677` (p=5.057e-05).
- vs `B1_SingleShot_API`: reward delta `0.0557` (95% CI [0.0354, 0.0756], p=5.109e-06, adjusted p=1.022e-05, Cohen's d=0.8353), engagement delta `0.0544` (p=4.167e-07), audience-fit delta `0.0352` (p=0.03561).
- vs `B2_OpenSource_Only`: reward delta `0.0606` (95% CI [0.0428, 0.0787], p=9.472e-08, adjusted p=2.842e-07, Cohen's d=1.0328), engagement delta `0.0592` (p=1.529e-07), audience-fit delta `0.0406` (p=0.02015).
- vs `B3_PromptEngineered_AI`: reward delta `0.0483` (95% CI [0.0274, 0.0689], p=6.706e-05, adjusted p=6.706e-05, Cohen's d=0.7056), engagement delta `0.0389` (p=0.0005262), audience-fit delta `0.0325` (p=0.008709).

## Ablations
- `Ours_EAI_CO`: reward `0.8051`, reward gap vs ours `0.0000`, audience-fit gap `0.0000`.
- `Ours_without_factual_penalty`: reward `0.8006`, reward gap vs ours `-0.0045`, audience-fit gap `-0.0027`.
- `Ours_without_audience_modeling`: reward `0.7833`, reward gap vs ours `-0.0218`, audience-fit gap `-0.0759`.
- `Ours_without_iterative_loop`: reward `0.7760`, reward gap vs ours `-0.0291`, audience-fit gap `-0.0298`.
- `Ours_without_diversity`: reward `0.7349`, reward gap vs ours `-0.0702`, audience-fit gap `-0.0163`.

## Cost Profile
- `Ours_without_factual_penalty`: mean latency `11470.75` ms, median latency `11962.50` ms, mean model calls `8.00`, aggregate latency `0.1275` hours.
- `Ours_EAI_CO`: mean latency `11114.23` ms, median latency `11358.50` ms, mean model calls `8.00`, aggregate latency `0.1235` hours.
- `Ours_without_diversity`: mean latency `10796.67` ms, median latency `11013.50` ms, mean model calls `8.00`, aggregate latency `0.1200` hours.
- `Ours_without_audience_modeling`: mean latency `10465.55` ms, median latency `10712.50` ms, mean model calls `8.00`, aggregate latency `0.1163` hours.
- `Ours_without_iterative_loop`: mean latency `5379.95` ms, median latency `5440.00` ms, mean model calls `4.00`, aggregate latency `0.0598` hours.
- `B1_SingleShot_API`: mean latency `2780.30` ms, median latency `2851.50` ms, mean model calls `4.00`, aggregate latency `0.0309` hours.
- `B3_PromptEngineered_AI`: mean latency `2727.47` ms, median latency `2810.00` ms, mean model calls `4.00`, aggregate latency `0.0303` hours.
- `B2_OpenSource_Only`: mean latency `2707.25` ms, median latency `2731.50` ms, mean model calls `3.00`, aggregate latency `0.0301` hours.
- `B0_Template`: mean latency `1.00` ms, median latency `1.00` ms, mean model calls `0.00`, aggregate latency `0.0000` hours.