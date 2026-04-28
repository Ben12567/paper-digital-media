# Real Experiment Runbook

## Purpose

This runbook upgrades the local prototype into a real study using:

- real text generation models;
- optional real image generation models;
- blind human evaluation exported as CSV packs;
- statistical analysis from collected ratings.

## Preconditions

### Option A: Hugging Face inference

- Recommended default for pilot runs:
  - text: `Qwen/Qwen2.5-7B-Instruct`
  - image: `black-forest-labs/FLUX.1-schnell`
- Set `HF_TOKEN` if the selected model or provider requires authentication.
- Free Hugging Face credits are suitable for small pilot runs, not full paper-scale experiments.

### Option B: OpenAI text generation

- Set `OPENAI_API_KEY`.
- Recommended: upgrade the `openai` package to a current 1.x release before production use.

### Option C: Local transformer text generation

- Place a compatible text-generation model on disk or ensure it is already cached locally.
- Update `configs/real_experiment_config.json`:
  - `text_provider.kind = "transformers"`
  - `text_provider.model_name_or_path = "<local model path or model id>"`

### Optional image generation

- Enable `image_provider.enabled = true`.
- Provide a local Stable Diffusion model path or a downloadable model id.
- CPU-only image generation is possible but slow.

## Step 1: Generate real creatives

```powershell
python scripts/run_real_generation.py
```

Output:

- `outputs/real_generation/real_candidates.csv`
- `outputs/real_generation/real_summary.csv`
- optional generated images in `outputs/real_generation/images/`

Recommended pilot run before full scale:

```powershell
python scripts/run_real_generation.py --config configs/real_experiment_config_hf_pilot.json --primary-only
```

## Step 2: Export blind human-evaluation pack

```powershell
python scripts/export_human_eval_pack.py --input outputs/real_generation/real_candidates.csv
```

Output:

- `outputs/human_eval_pack/rating_items.csv`
- `outputs/human_eval_pack/pairwise_items.csv`
- `outputs/human_eval_pack/blind_key.csv`
- response templates for raters

## Step 3: Collect ratings

Collect two CSV files:

- one from scalar ratings based on `rating_response_template.csv`
- one from pairwise choices based on `pairwise_response_template.csv`

## Step 4: Score collected ratings

```powershell
python scripts/score_human_eval.py ^
  --items outputs/human_eval_pack/rating_items.csv ^
  --ratings path\\to\\completed_rating_responses.csv ^
  --blind-key outputs/human_eval_pack/blind_key.csv
```

Output:

- `outputs/human_eval_scored/human_method_summary.csv`
- `outputs/human_eval_scored/human_significance_vs_ours.csv`
- `outputs/human_eval_scored/human_global_statistics.csv`

## Current hard blockers in this workspace

- `HF_TOKEN` is not set for Hugging Face hosted inference.
- `OPENAI_API_KEY` is not set for OpenAI hosted inference.
- No local generative text model is currently cached for the transformers path.
- No real human ratings file is present.

Until at least one real text-generation route is configured and rating data exists, this workspace can prepare and verify the pipeline, but it cannot produce true real-model and real-human results.
