param(
    [switch]$SkipStatic,
    [switch]$SkipDynamic,
    [switch]$SkipPortfolio,
    [switch]$SkipManuscript
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$experiments = Join-Path $root "experiments"
$manuscript = Join-Path $root "manuscript"

Push-Location $root
try {
    if (-not $SkipStatic) {
        Write-Host "[1/4] Running static ZDT sanity benchmarks..."
        Push-Location $experiments
        try {
            python run_benchmarks.py
        }
        finally {
            Pop-Location
        }
    }

    if (-not $SkipDynamic) {
        Write-Host "[2/4] Running expanded dynamic DF benchmarks..."
        Push-Location $experiments
        try {
            python debug_predictive_baselines.py
            python run_dynamic_asoc_suite.py
            python validate_dynamic_activation_audit.py
            python run_dynamic_pitstop_budget_probe.py
            python rebuild_dynamic_asoc_reports.py
            python run_dynamic_sensitivity_asoc.py
        }
        finally {
            Pop-Location
        }
    }

    if (-not $SkipPortfolio) {
        Write-Host "[3/4] Running expanded walk-forward portfolio suite..."
        Push-Location $experiments
        try {
            python run_portfolio_asoc_suite.py
            python validate_portfolio_results.py
            python rebuild_portfolio_reports.py
        }
        finally {
            Pop-Location
        }
    }

    if (-not $SkipManuscript) {
        Write-Host "[4/4] Building graphical abstract and manuscript PDF..."
        Push-Location $manuscript
        try {
            python build_graphical_abstract.py
            pdflatex -interaction=nonstopmode -halt-on-error asoc_fito.tex
            pdflatex -interaction=nonstopmode -halt-on-error asoc_fito.tex
        }
        finally {
            Pop-Location
        }
    }
}
finally {
    Pop-Location
}
