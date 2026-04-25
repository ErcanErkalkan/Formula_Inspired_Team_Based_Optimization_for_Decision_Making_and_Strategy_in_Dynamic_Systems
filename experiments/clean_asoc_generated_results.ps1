$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$results = Join-Path $root "results"

Remove-Item (Join-Path $results "predictive_baseline_debug_smoke.csv")
Remove-Item (Join-Path $results "asoc_dynamic_*.csv")
Remove-Item (Join-Path $results "asoc_dynamic_*.tex")
Remove-Item (Join-Path $results "asoc_dynamic_*.txt")
Remove-Item (Join-Path $results "asoc_dynamic_*.json")
Remove-Item (Join-Path $results "sensitivity_analysis.*")
Remove-Item (Join-Path $results "asoc_portfolio_*.csv")
Remove-Item (Join-Path $results "asoc_portfolio_*.tex")
Remove-Item (Join-Path $results "asoc_portfolio_*.txt")
Remove-Item (Join-Path $results "asoc_portfolio_*.json")
Write-Host "Cleaned ASOC-generated result files."
