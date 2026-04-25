# FITO Reproducibility Package

This repository contains the code, generated artifacts, and manuscript for:

`Formula-Inspired Team-Based Evolutionary Optimization with Reactive Redeployment for Dynamic Multi-Objective Tracking`

The package is organized so that all reported tables, figures, and the manuscript PDF can be regenerated from the repository root.

## Repository layout

- `experiments/run_dynamic_asoc_suite.py`: expanded DF1--DF9 dynamic benchmark suite over three protocols, with ablations and a fixed-budget auxiliary study.
- `experiments/run_dynamic_pitstop_budget_probe.py`: focused fixed-budget comparison between FITO and `FITO-noPS`.
- `experiments/run_dynamic_sensitivity_asoc.py`: multi-problem FITO parameter sensitivity audit over DF1, DF4, DF7, and DF9.
- `experiments/rebuild_dynamic_asoc_reports.py`: rebuilds the final dynamic summary/report files from the generated raw artifacts.
- `experiments/run_benchmarks.py`: static ZDT sanity check and appendix tables.
- `experiments/run_portfolio_asoc_suite.py`: two-universe walk-forward portfolio suite with decision-rule and cost sensitivity.
- `experiments/validate_portfolio_results.py`: verifies that `asoc_portfolio_raw_metrics.csv` covers all seven ASOC algorithms, both universes, both budget families, and all seeds.
- `experiments/results/`: generated CSV, JSON, PNG, and LaTeX table artifacts.
- `manuscript/asoc_fito.tex`: main manuscript source.
- `manuscript/build_graphical_abstract.py`: generates `graphical_abstract.png` and `graphical_abstract.pdf` from current result files.
- `manuscript/highlights.txt`: submission highlights.
- `manuscript/generative_ai_declaration.txt`: submission declaration text.
- `reproduce_all.ps1`: one-command PowerShell entry point for regenerating all artifacts.
- `requirements.txt`: pinned Python package versions used for the reported runs.

## Tested environment

The reported artifacts were regenerated in the following local environment:

- OS: Windows
- Python: `3.14.3`
- `pdflatex`: MiKTeX `25.4`

The Python package versions used for the reported runs are pinned in `requirements.txt`.

## Quick start

From the repository root:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To regenerate everything with one command:

```powershell
.\reproduce_all.ps1
```

This runs the three experiment scripts in sequence and compiles the manuscript twice.

## Manual reproduction

If you prefer to run each step manually:

```powershell
Set-Location experiments
python run_benchmarks.py
python run_dynamic_asoc_suite.py
python run_dynamic_pitstop_budget_probe.py
python rebuild_dynamic_asoc_reports.py
python run_dynamic_sensitivity_asoc.py
python run_portfolio_asoc_suite.py
python validate_portfolio_results.py
Set-Location ..\manuscript
python build_graphical_abstract.py
pdflatex -interaction=nonstopmode -halt-on-error asoc_fito.tex
pdflatex -interaction=nonstopmode -halt-on-error asoc_fito.tex
Set-Location ..
```

## Expected outputs

The main regenerated artifacts are:

- `experiments/results/asoc_dynamic_summary.txt`
- `experiments/results/asoc_dynamic_protocol_ranks.tex`
- `experiments/results/asoc_dynamic_budget_ranks.tex`
- `experiments/results/asoc_dynamic_ablation_ranks.tex`
- `experiments/results/asoc_dynamic_main_stats.csv`
- `experiments/results/asoc_dynamic_budget_stats.csv`
- `experiments/results/asoc_dynamic_sensitivity_raw.csv`
- `experiments/results/asoc_dynamic_sensitivity_summary.csv`
- `experiments/results/asoc_dynamic_sensitivity_table.tex`
- `experiments/results/sensitivity_analysis.pdf`
- `experiments/results/hv_table.tex`
- `experiments/results/igd_table.tex`
- `experiments/results/ablation_table.tex`
- `experiments/results/summary.txt`
- `experiments/results/eval_budget_summary.csv`
- `experiments/results/asoc_portfolio_mhv_table.tex`
- `experiments/results/asoc_portfolio_migd_table.tex`
- `experiments/results/asoc_portfolio_deployment_table.tex`
- `experiments/results/asoc_portfolio_summary.txt`
- `experiments/results/asoc_portfolio_stats.csv`
- `experiments/results/asoc_portfolio_algorithm_coverage.csv`
- `experiments/results/asoc_portfolio_algorithm_manifest.json`
- `experiments/results/asoc_portfolio_benchmarks.csv`
- `manuscript/graphical_abstract.png`
- `manuscript/graphical_abstract.pdf`
- `manuscript/asoc_fito.pdf`

## Data notes

- The `tech14` walk-forward study uses the bundled cache files `experiments/data/big_tech_stock_prices.csv`, `experiments/data/big_tech_companies.csv`, and `experiments/data/big_tech_walkforward_envs.pkl`.
- The `market20` walk-forward study uses locally cached `yfinance` downloads stored under `experiments/data/market20_adj_close.csv` and `experiments/data/market20_walkforward_envs.pkl`.
- If the cache files are missing, the portfolio suite will rebuild them; the `market20` rebuild path requires network access.

## Runtime notes

Approximate runtimes observed in the tested environment:

- `python experiments/run_benchmarks.py`: about 29 minutes
- `python experiments/run_dynamic_asoc_suite.py`: about 70-75 minutes
- `python experiments/run_dynamic_pitstop_budget_probe.py`: about 4 minutes
- `python experiments/run_dynamic_sensitivity_asoc.py`: auxiliary multi-problem sensitivity run; runtime depends on CPU and seed count
- `python experiments/run_portfolio_asoc_suite.py`: about 55 minutes
- `python manuscript/build_graphical_abstract.py`: under 1 minute
- manuscript compilation: under 1 minute

Actual runtime depends on CPU, Python build, and BLAS configuration.

## Verification

After regeneration, the manuscript should compile successfully and the summary files should reflect the reported results:

- dynamic average ranks in `experiments/results/asoc_dynamic_summary.txt`
- dynamic pit-stop budget probe in `experiments/results/asoc_dynamic_pitstop_budget_probe.txt`
- multi-problem sensitivity audit in `experiments/results/asoc_dynamic_sensitivity_summary.csv` and `experiments/results/sensitivity_analysis.pdf`
- static average ranks in `experiments/results/summary.txt`
- static objective-evaluation budgets in `experiments/results/eval_budget_summary.csv`
- walk-forward portfolio coverage in `experiments/results/asoc_portfolio_algorithm_coverage.csv`
- walk-forward portfolio summary in `experiments/results/asoc_portfolio_summary.txt`
- walk-forward deterministic benchmarks in `experiments/results/asoc_portfolio_benchmarks.csv`

The LaTeX build may emit `overfull` or `underfull hbox` warnings; these are typographic warnings and do not indicate a failed reproduction.

## Predictive-baseline debug patch

The patched package includes an audited implementation of the MDDM-DMOEA and
PPS-DMOEA baselines in `experiments/predictive_baselines.py`.  The patch fixes
three baseline-validity issues: effective environment-clock tracking,
forced objective re-evaluation after direct `X` edits, and explicit response
counters for predictive/density-based branches.

Recommended verification order:

```bash
python experiments/debug_predictive_baselines.py
python experiments/run_dynamic_asoc_suite.py
python experiments/rebuild_dynamic_asoc_reports.py
```

The smoke script writes
`experiments/results/predictive_baseline_debug_smoke.csv` and should show
non-zero `change_response_count`, `prediction_count`, and `replaced_count` for
MDDM-DMOEA and PPS-DMOEA.

## Final baseline/audit fairness patch

The current package includes an additional predictive-baseline audit/fairness fix:

- `experiments/predictive_baselines.py` hooks both `_advance(...)` and `_step()` so that current `pymoo` versions trigger dynamic-response logic correctly.
- MDDM-DMOEA and PPS-DMOEA refresh objective values after direct population edits, but response-refresh evaluations are reported separately as `response_evaluation_count` instead of inflating nominal generation-matched `n_evals`.
- `experiments/debug_predictive_baselines.py` now fails fast if MDDM-DMOEA/PPS-DMOEA remain identical to NSGA-II, fail to activate, or inflate nominal `n_evals`.
- `experiments/validate_dynamic_activation_audit.py` validates the full dynamic suite after `run_dynamic_asoc_suite.py` and writes `experiments/results/asoc_dynamic_activation_validation_summary.csv`.
