# EAI-CO Reproducibility Package

This repository contains the reproducible local benchmark for EAI-CO, an exploratory multi-objective optimization framework for AI-assisted social-media advertising creative generation.

## Scientific Scope

The checked-in experiment is an offline automatic-evaluation benchmark. It compares creative-generation workflows under the same task set, search budget, and heuristic evaluator. The results do not establish real click-through-rate improvement, conversion lift, legal compliance, or human preference.

## Repository Layout

- `configs/local_qwen7b.json`: pinned local Qwen2.5-7B-Instruct benchmark configuration.
- `data/sample_products.csv`: product records used to construct the controlled benchmark.
- `src/eai_co/`: generation, evaluation, and optimization implementation.
- `scripts/run_real_generation.py`: local or API-backed benchmark runner.
- `scripts/analyze_real_experiment.py`: analysis pipeline for raw candidate records.
- `results/local_qwen7b_10p/`: checked-in local benchmark records and derived analysis tables.
- `manuscript/applsci_mdpi_template_exact.tex`: English manuscript working draft.
- `manuscript/figures/`: manuscript-ready PDF figures.
- `manuscript/figure_sources/`: editable source for the introduction figure.

## Environment

The local benchmark was executed on an NVIDIA RTX 3090 GPU. Install Python dependencies in a clean environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce the Local Benchmark

The model configuration uses the public Hugging Face model ID and a fixed repository revision.

```powershell
python scripts/run_real_generation.py `
  --config configs/local_qwen7b.json `
  --output-dir outputs/local_qwen7b

python scripts/analyze_real_experiment.py `
  --input outputs/local_qwen7b/real_candidates.csv `
  --output-dir outputs/local_qwen7b_analysis `
  --model-label Qwen/Qwen2.5-7B-Instruct
```

The full generation run is computationally expensive. To validate the analysis layer against the checked-in raw records:

```powershell
python scripts/analyze_real_experiment.py `
  --input results/local_qwen7b_10p/real_candidates.csv `
  --output-dir outputs/reproduced_analysis `
  --model-label Qwen/Qwen2.5-7B-Instruct
```

Hosted provider implementations remain available for future extension, but no API-backed result is included as validated evidence in this repository.
