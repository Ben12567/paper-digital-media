# Exploratory AI for Social Media Advertising Creatives

This workspace implements the planned SCI paper package for:

**Exploratory Artificial Intelligence for Automated Social Media Advertising Content Creation and Multi-objective Creative Optimization**

## Contents

- `manuscript/sci_paper_draft.md`: structured SCI manuscript draft.
- `protocol/experiment_protocol.md`: detailed experiment and analysis protocol.
- `protocol/human_evaluation_form.md`: human evaluation questionnaire template.
- `configs/experiment_config.json`: default experiment settings.
- `data/sample_products.csv`: small demo product set.
- `src/eai_co/`: runnable EAI-CO prototype with deterministic mock generation/evaluation.
- `scripts/run_demo.py`: demo runner that creates candidate ads and exports results.

## Quick Start

```powershell
python scripts/run_demo.py
```

Outputs are written to:

- `outputs/demo_candidates.jsonl`
- `outputs/demo_summary.csv`

The prototype is dependency-light and uses deterministic rule-based stand-ins for API LLMs, image models, and automatic evaluators. Replace the provider classes in `src/eai_co/` with actual model calls when running the full study.

## Research Claim Boundary

The project is designed around predicted engagement and human-rated click intention. It does not claim real platform CTR improvement unless a future live A/B test is added.

## Real Study Upgrade

Real-study scaffolding is included for replacing the prototype with actual model calls and blind human evaluation:

- `configs/real_experiment_config.json`
- `scripts/run_real_generation.py`
- `scripts/export_human_eval_pack.py`
- `scripts/score_human_eval.py`
- `protocol/real_experiment_runbook.md`

### Current blockers in this workspace

- `HF_TOKEN` is not set for Hugging Face hosted inference.
- `OPENAI_API_KEY` is not set for OpenAI hosted inference.
- No local generative text model is configured.
- No completed human ratings CSV exists yet.

### Intended sequence

```powershell
python scripts/run_real_generation.py
python scripts/export_human_eval_pack.py --input outputs/real_generation/real_candidates.csv
python scripts/score_human_eval.py --items outputs/human_eval_pack/rating_items.csv --ratings path\to\completed_rating_responses.csv --blind-key outputs/human_eval_pack/blind_key.csv
```
