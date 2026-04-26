# FITO Reproducibility Package

This repository contains the code, generated artifacts, cached data, and manuscript source for:

`Formula-Inspired Team-Based Evolutionary Optimization with Reactive Redeployment for Dynamic Multi-Objective Tracking`

Repository URL for the data-availability statement:

`https://github.com/ErcanErkalkan/Formula_Inspired_Team_Based_Optimization_for_Decision_Making_and_Strategy_in_Dynamic_Systems`

The repository is organized so that the reported dynamic benchmark tables, secondary walk-forward portfolio stress-test tables, figures, graphical abstract, and manuscript PDF can be regenerated from the repository root.

## Repository layout

- `experiments/run_dynamic_asoc_suite.py`: expanded DF1--DF9 dynamic benchmark suite over three protocols, ablations, activation auditing, and the primary approximately fixed-budget dynamic study.
- `experiments/run_dynamic_pitstop_budget_probe.py`: focused fixed-budget comparison between FITO and `FITO-noPS`.
- `experiments/run_dynamic_sensitivity_asoc.py`: multi-problem FITO parameter sensitivity audit over DF1, DF4, DF7, and DF9.
- `experiments/rebuild_dynamic_asoc_reports.py`: rebuilds final dynamic summary/report files from existing raw artifacts.
- `experiments/run_benchmarks.py`: static ZDT sanity check and appendix table generator.
- `experiments/run_portfolio_asoc_suite.py`: two-universe walk-forward portfolio stress-test suite with decision-rule and transaction-cost sensitivity.
- `experiments/rebuild_portfolio_reports.py`: rebuilds portfolio stress-test report tables from existing raw metrics.
- `experiments/validate_portfolio_results.py`: verifies that `asoc_portfolio_raw_metrics.csv` covers all seven ASOC algorithms, both universes, both budget families, and all seeds.
- `experiments/smoke_test.py`: quick import, artifact, fixed-budget, and portfolio-coverage smoke test.
- `experiments/results/`: generated CSV, JSON, PNG, PDF, and LaTeX table artifacts.
- `manuscript/asoc_fito.tex`: main manuscript source.
- `manuscript/asoc_fito.pdf`: compiled manuscript PDF corresponding to this snapshot.
- `manuscript/build_graphical_abstract.py`: generates `graphical_abstract.png` and `graphical_abstract.pdf` from current result files.
- `manuscript/highlights.txt`: submission highlights.
- `manuscript/generative_ai_declaration.txt`: submission declaration text.
- `reproduce_all.ps1`: Windows PowerShell entry point.
- `reproduce_all.sh`: Linux/macOS Bash entry point.
- `Makefile`: convenience targets for setup, smoke testing, table rebuilds, full reproduction, and checksum verification.
- `requirements.txt`: pinned Python package versions used for the reported runs.
- `environment.yml`: Conda environment wrapper that installs the pinned pip requirements.
- `pyproject.toml`: lightweight project metadata for the reproducibility artifact.
- `scripts/generate_checksums.py`: generates the SHA-256 manifest.
- `scripts/verify_checksums.py`: verifies the SHA-256 manifest.
- `SHA256SUMS.txt`: checksum manifest for the repository snapshot.

## Tested environment

The reported artifacts were regenerated in the following local environment:

- OS: Windows
- Python: `3.14.3`
- `pdflatex`: MiKTeX `25.4`

The dependency pins are listed in `requirements.txt`. For Linux/macOS and reviewer environments, use `reproduce_all.sh`, `Makefile`, or `environment.yml`. Full runtimes may vary with CPU, Python build, and BLAS configuration.

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux/macOS Bash

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Conda alternative

```bash
conda env create -f environment.yml
conda activate fito-reproducibility
```

## Reproduction levels

### 1. Quick smoke test

This command checks imports, required artifacts, primary fixed-budget ranks, realized-budget files, and portfolio coverage. It does not rerun the full experiments.

Windows:

```powershell
.\reproduce_all.ps1 -QuickSmoke
```

Linux/macOS:

```bash
bash reproduce_all.sh --quick-smoke
```

Makefile:

```bash
make smoke
```

### 2. Table-only rebuild from existing raw artifacts

This is the recommended reviewer command when the generated raw metrics are already present. It rebuilds dynamic tables, portfolio tables, the graphical abstract, and the manuscript PDF without rerunning all expensive optimization experiments. Static ZDT appendix tables are shipped as precomputed artifacts in this snapshot and are regenerated only by running `experiments/run_benchmarks.py` or the full pipeline.

Windows:

```powershell
.\reproduce_all.ps1 -TablesOnly
```

Linux/macOS:

```bash
bash reproduce_all.sh --tables-only
```

Makefile:

```bash
make tables
```

### 3. Full reproduction

This reruns the static benchmarks, dynamic DF benchmark suite, fixed-budget dynamic study, activation audit, pit-stop probe, sensitivity audit, portfolio stress test, graphical abstract generation, and manuscript build.

Windows:

```powershell
.\reproduce_all.ps1
```

Linux/macOS:

```bash
bash reproduce_all.sh
```

Makefile:

```bash
make full
```

Selective execution is also supported:

```bash
bash reproduce_all.sh --skip-static --skip-portfolio
bash reproduce_all.sh --skip-manuscript
make dynamic
make portfolio
make manuscript
```

## Expected outputs present in this package

The main regenerated dynamic benchmark artifacts are:

- `experiments/results/asoc_dynamic_summary.txt`
- `experiments/results/asoc_dynamic_main_summary.csv`
- `experiments/results/asoc_dynamic_main_stats.csv`
- `experiments/results/asoc_dynamic_main_ranks.csv`
- `experiments/results/asoc_dynamic_protocol_ranks.tex`
- `experiments/results/asoc_dynamic_eval_budget.csv` (nominal, response-refresh, and total-call audit source)
- `experiments/results/asoc_dynamic_eval_budget_table.tex` (generation-matched nominal/response/total evaluation audit)
- `experiments/results/asoc_dynamic_budget_summary.csv` (primary fixed-budget performance summary, not the evaluation-budget audit)
- `experiments/results/asoc_dynamic_budget_stats.csv`
- `experiments/results/asoc_dynamic_budget_ranks.csv`
- `experiments/results/asoc_dynamic_budget_ranks.tex`
- `experiments/results/asoc_dynamic_fixed_budget_table.tex` (primary fixed-budget nominal/response/total evaluation audit)
- `experiments/results/asoc_dynamic_ablation_summary.csv`
- `experiments/results/asoc_dynamic_ablation_ranks.csv`
- `experiments/results/asoc_dynamic_ablation_ranks.tex`
- `experiments/results/asoc_dynamic_pitstop_budget_probe.csv`
- `experiments/results/asoc_dynamic_pitstop_budget_probe.txt`
- `experiments/results/asoc_dynamic_pitstop_budget_probe_ranks.csv`
- `experiments/results/asoc_dynamic_pitstop_budget_probe_summary.csv`
- `experiments/results/asoc_dynamic_sensitivity_raw.csv`
- `experiments/results/asoc_dynamic_sensitivity_summary.csv`
- `experiments/results/asoc_dynamic_sensitivity_table.tex`
- `experiments/results/asoc_dynamic_activation_audit_events.csv`
- `experiments/results/asoc_dynamic_activation_audit_summary.csv`
- `experiments/results/asoc_dynamic_activation_validation_summary.csv`
- `experiments/results/asoc_dynamic_fast_t5_n10_table.tex`
- `experiments/results/asoc_dynamic_moderate_t10_n10_table.tex`
- `experiments/results/asoc_dynamic_severe_t10_n20_table.tex`
- `experiments/results/sensitivity_analysis.pdf`
- `experiments/results/sensitivity_analysis.png`

The static sanity-check and appendix artifacts are precomputed in this snapshot; they are regenerated by `experiments/run_benchmarks.py` or the full pipeline:

- `experiments/results/hv_table.tex`
- `experiments/results/igd_table.tex`
- `experiments/results/ablation_table.tex`

The secondary walk-forward portfolio stress-test artifacts are:

- `experiments/results/asoc_portfolio_summary.txt`
- `experiments/results/asoc_portfolio_raw_metrics.csv`
- `experiments/results/asoc_portfolio_stats.csv`
- `experiments/results/asoc_portfolio_algorithm_coverage.csv`
- `experiments/results/asoc_portfolio_algorithm_manifest.json`
- `experiments/results/asoc_portfolio_benchmarks.csv`
- `experiments/results/asoc_portfolio_benchmark_table.tex`
- `experiments/results/asoc_portfolio_budget_stats.csv`
- `experiments/results/asoc_portfolio_deployment_summary.csv`
- `experiments/results/asoc_portfolio_deployment_table.tex`
- `experiments/results/asoc_portfolio_mhv_summary.csv`
- `experiments/results/asoc_portfolio_mhv_table.tex`
- `experiments/results/asoc_portfolio_migd_summary.csv`
- `experiments/results/asoc_portfolio_migd_table.tex`
- `experiments/results/asoc_portfolio_sensitivity_counts.csv`
- `experiments/results/asoc_portfolio_sensitivity_full.csv`
- `experiments/results/asoc_portfolio_activation_audit_events.csv`
- `experiments/results/asoc_portfolio_activation_audit_summary.csv`
- `experiments/results/asoc_portfolio_scope_table.tex`

The manuscript and graphical-abstract artifacts are:

- `manuscript/graphical_abstract.png`
- `manuscript/graphical_abstract.pdf`
- `manuscript/asoc_fito.tex`
- `manuscript/asoc_fito.pdf`

The obsolete legacy names `experiments/results/summary.txt` and `experiments/results/eval_budget_summary.csv` are intentionally not listed for the current ASOC dynamic-report pipeline. Legacy or static helper scripts may still create similarly named files in isolated runs; the current ASOC dynamic manuscript artifacts are `experiments/results/asoc_dynamic_summary.txt` and `experiments/results/asoc_dynamic_eval_budget.csv`.

## Data notes

- The `tech14` walk-forward study uses the bundled cache files `experiments/data/big_tech_stock_prices.csv`, `experiments/data/big_tech_companies.csv`, and `experiments/data/big_tech_walkforward_envs.pkl`.
- The `market20` walk-forward study uses locally cached `yfinance` downloads stored under `experiments/data/market20_adj_close.csv` and `experiments/data/market20_walkforward_envs.pkl`.
- If the cache files are missing, the portfolio suite will rebuild them; the `market20` rebuild path requires network access.
- The cached portfolio price files cover a broader 2012--2022 interval; the generated 38 walk-forward holdout windows run from 2013-05-23 to 2022-11-22.
- The portfolio study is a secondary application stress test, not a real-world finance validation study.

## Checksums and snapshot verification

Generate the checksum manifest:

```bash
python scripts/generate_checksums.py
# or
make checksums
```

Verify the snapshot:

```bash
python scripts/verify_checksums.py
# or
make verify-checksums
```

`SHA256SUMS.txt` is intended to support repository archival, reviewer verification, and data-availability reproducibility.

## Runtime notes

Approximate runtimes observed in the tested environment:

- `python experiments/run_benchmarks.py`: about 29 minutes
- `python experiments/run_dynamic_asoc_suite.py`: about 70--75 minutes
- `python experiments/run_dynamic_pitstop_budget_probe.py`: about 4 minutes
- `python experiments/run_dynamic_sensitivity_asoc.py`: auxiliary multi-problem sensitivity run; runtime depends on CPU and seed count
- `python experiments/run_portfolio_asoc_suite.py`: about 55 minutes
- `python manuscript/build_graphical_abstract.py`: under 1 minute
- manuscript compilation: under 1 minute

Actual runtime depends on CPU, Python build, and BLAS configuration.

## Verification checklist

After regeneration, the manuscript should compile successfully and the summary files should reflect the reported results:

- primary fixed-budget dynamic ranks in `experiments/results/asoc_dynamic_budget_ranks.tex` and `experiments/results/asoc_dynamic_budget_ranks.csv`
- primary fixed-budget performance summaries in `experiments/results/asoc_dynamic_budget_summary.csv` and `experiments/results/asoc_dynamic_budget_stats.csv`, plus the nominal/response/total evaluation audit in `experiments/results/asoc_dynamic_eval_budget.csv` and `experiments/results/asoc_dynamic_fixed_budget_table.tex`
- auxiliary generation-matched dynamic average ranks in `experiments/results/asoc_dynamic_protocol_ranks.tex`
- auxiliary generation-matched nominal/response/total objective-evaluation audit in `experiments/results/asoc_dynamic_eval_budget.csv` and `experiments/results/asoc_dynamic_eval_budget_table.tex`
- dynamic pit-stop budget probe in `experiments/results/asoc_dynamic_pitstop_budget_probe.txt`
- multi-problem sensitivity audit in `experiments/results/asoc_dynamic_sensitivity_summary.csv` and `experiments/results/sensitivity_analysis.pdf`
- static appendix sanity-check tables in `experiments/results/hv_table.tex`, `experiments/results/igd_table.tex`, and `experiments/results/ablation_table.tex`
- walk-forward portfolio coverage in `experiments/results/asoc_portfolio_algorithm_coverage.csv`
- walk-forward portfolio stress-test summary in `experiments/results/asoc_portfolio_summary.txt`
- walk-forward deterministic benchmarks in `experiments/results/asoc_portfolio_benchmarks.csv` and `experiments/results/asoc_portfolio_benchmark_table.tex`

The LaTeX build may emit `overfull` or `underfull hbox` warnings; these are typographic warnings and do not indicate a failed reproduction.

## Predictive-baseline debug patch

The package includes an audited implementation of the MDDM-DMOEA and PPS-DMOEA baselines in `experiments/predictive_baselines.py`. The patch fixes effective environment-clock tracking, forced objective re-evaluation after direct `X` edits, and explicit response counters for predictive/density-based branches.

Recommended verification order:

```bash
python experiments/debug_predictive_baselines.py
python experiments/run_dynamic_asoc_suite.py
python experiments/rebuild_dynamic_asoc_reports.py
```

The debug script writes `experiments/results/predictive_baseline_debug_smoke.csv` and should show non-zero `change_response_count`, `prediction_count`, and `replaced_count` for MDDM-DMOEA and PPS-DMOEA.

## Final baseline/audit fairness patch

The current package includes an additional predictive-baseline audit/fairness fix:

- `experiments/predictive_baselines.py` hooks both `_advance(...)` and `_step()` so that current `pymoo` versions trigger dynamic-response logic correctly.
- MDDM-DMOEA and PPS-DMOEA refresh objective values after direct population edits, but response-refresh evaluations are reported separately as `response_evaluation_count` instead of inflating nominal `n_evals`; report tables therefore distinguish nominal evaluations, response-refresh evaluations, and total objective calls.
- `experiments/debug_predictive_baselines.py` fails fast if MDDM-DMOEA/PPS-DMOEA remain identical to NSGA-II, fail to activate, or inflate nominal `n_evals`.
- `experiments/validate_dynamic_activation_audit.py` validates the full dynamic suite after `run_dynamic_asoc_suite.py` and writes `experiments/results/asoc_dynamic_activation_validation_summary.csv`.
