$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

Set-Location "C:\Users\imwri\OneDrive\Desktop\AlgoverseAssumptionTypes"
$Host.UI.RawUI.WindowTitle = "Azure completion-budget gate: 8000 tokens"

Write-Host "Running one causal four-arm quality gate at 8,000 completion tokens..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe scripts\run_pilot.py `
  --client azure `
  --data outputs\validation\scenarios_scope_live6.json `
  --scenario vendor_001 `
  --variants causal `
  --scope-ladder `
  --topologies fully_connected_mesh `
  --n-seeds 1 `
  --skip-baselines `
  --skip-independent `
  --rounds 4 `
  --temperature none `
  --max-completion-tokens 8000 `
  --verbose `
  --tag budget_smoke8000

if ($LASTEXITCODE -ne 0) {
  Write-Host "Quality-gate run failed with exit code $LASTEXITCODE." -ForegroundColor Red
  return
}

$run = Get-ChildItem -LiteralPath outputs\raw -Directory |
  Where-Object { $_.Name -like "*_azure_budget_smoke8000" } |
  Sort-Object LastWriteTime |
  Select-Object -Last 1
$records = Get-Content -LiteralPath (Join-Path $run.FullName "records.jsonl") |
  ForEach-Object { $_ | ConvertFrom-Json }
$parseFailures = @($records | Where-Object { $_.parse_failed }).Count

Write-Host "`nQuality gate complete." -ForegroundColor Cyan
Write-Host "  Records: $($records.Count)/64"
Write-Host "  Parse failures: $parseFailures"
Write-Host "  Run directory: $($run.FullName)"
if ($parseFailures -eq 0 -and $records.Count -eq 64) {
  Write-Host "  RESULT: PASS" -ForegroundColor Green
} else {
  Write-Host "  RESULT: FAIL" -ForegroundColor Red
}
