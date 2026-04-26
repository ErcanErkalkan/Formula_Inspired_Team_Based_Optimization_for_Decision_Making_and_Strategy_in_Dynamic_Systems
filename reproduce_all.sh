#!/usr/bin/env bash
# Cross-platform reproduction entry point for Linux/macOS.
# Usage examples:
#   bash reproduce_all.sh --quick-smoke
#   bash reproduce_all.sh --tables-only
#   bash reproduce_all.sh --skip-static --skip-portfolio
#   bash reproduce_all.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$ROOT_DIR/experiments"
MANUSCRIPT_DIR="$ROOT_DIR/manuscript"

SKIP_STATIC=0
SKIP_DYNAMIC=0
SKIP_PORTFOLIO=0
SKIP_MANUSCRIPT=0
QUICK_SMOKE=0
TABLES_ONLY=0

usage() {
  cat <<'EOF'
FITO reproduction script for Linux/macOS

Options:
  --quick-smoke      Run a lightweight import/artifact/activation smoke test only.
  --tables-only      Rebuild report tables from existing raw result artifacts only.
  --skip-static      Skip static ZDT sanity benchmarks.
  --skip-dynamic     Skip dynamic DF benchmark suite.
  --skip-portfolio   Skip walk-forward portfolio suite.
  --skip-manuscript  Skip graphical abstract and LaTeX manuscript build.
  -h, --help         Show this help message.

Default behavior runs the full reproduction pipeline.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick-smoke) QUICK_SMOKE=1 ;;
    --tables-only) TABLES_ONLY=1 ;;
    --skip-static) SKIP_STATIC=1 ;;
    --skip-dynamic) SKIP_DYNAMIC=1 ;;
    --skip-portfolio) SKIP_PORTFOLIO=1 ;;
    --skip-manuscript) SKIP_MANUSCRIPT=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

cd "$ROOT_DIR"

if [[ "$QUICK_SMOKE" -eq 1 ]]; then
  echo "[smoke] Running lightweight reproducibility smoke test..."
  python experiments/smoke_test.py
  exit 0
fi

if [[ "$TABLES_ONLY" -eq 1 ]]; then
  echo "[tables] Rebuilding dynamic and portfolio report tables from existing artifacts..."
  (cd "$EXPERIMENTS_DIR" && python rebuild_dynamic_asoc_reports.py && python rebuild_portfolio_reports.py)
  if [[ "$SKIP_MANUSCRIPT" -eq 0 ]]; then
    echo "[tables] Rebuilding graphical abstract and manuscript PDF..."
    (cd "$MANUSCRIPT_DIR" && python build_graphical_abstract.py && pdflatex -interaction=nonstopmode -halt-on-error asoc_fito.tex && pdflatex -interaction=nonstopmode -halt-on-error asoc_fito.tex)
  fi
  exit 0
fi

if [[ "$SKIP_STATIC" -eq 0 ]]; then
  echo "[1/4] Running static ZDT sanity benchmarks..."
  (cd "$EXPERIMENTS_DIR" && python run_benchmarks.py)
fi

if [[ "$SKIP_DYNAMIC" -eq 0 ]]; then
  echo "[2/4] Running expanded dynamic DF benchmarks..."
  (cd "$EXPERIMENTS_DIR" && \
    python debug_predictive_baselines.py && \
    python run_dynamic_asoc_suite.py && \
    python validate_dynamic_activation_audit.py && \
    python run_dynamic_pitstop_budget_probe.py && \
    python rebuild_dynamic_asoc_reports.py && \
    python run_dynamic_sensitivity_asoc.py)
fi

if [[ "$SKIP_PORTFOLIO" -eq 0 ]]; then
  echo "[3/4] Running expanded walk-forward portfolio suite..."
  (cd "$EXPERIMENTS_DIR" && \
    python run_portfolio_asoc_suite.py && \
    python validate_portfolio_results.py && \
    python rebuild_portfolio_reports.py)
fi

if [[ "$SKIP_MANUSCRIPT" -eq 0 ]]; then
  echo "[4/4] Building graphical abstract and manuscript PDF..."
  (cd "$MANUSCRIPT_DIR" && \
    python build_graphical_abstract.py && \
    pdflatex -interaction=nonstopmode -halt-on-error asoc_fito.tex && \
    pdflatex -interaction=nonstopmode -halt-on-error asoc_fito.tex)
fi
