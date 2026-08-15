$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

Set-Location -LiteralPath (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Host.UI.RawUI.WindowTitle = "Azure evidence-scope pilot: 6 scenarios"

Write-Host "Azure evidence-scope pilot" -ForegroundColor Cyan
Write-Host "  Scenarios: 6"
Write-Host "  Types: factual + causal (framing excluded)"
Write-Host "  Arms: none / testimony / verbatim relay / direct access"
Write-Host "  Design: 1 seed, 4 rounds, fully connected mesh"
Write-Host "  Expected measured conversations: 48"
Write-Host "  Expected logical model generations: approximately 540"
Write-Host ""

& .\.venv\Scripts\python.exe scripts\run_pilot.py `
  --client azure `
  --data outputs\validation\scenarios_scope_live6.json `
  --variants factual causal `
  --scope-ladder `
  --topologies fully_connected_mesh `
  --n-seeds 1 `
  --skip-baselines `
  --skip-independent `
  --rounds 4 `
  --temperature none `
  --max-completion-tokens 8000 `
  --verbose `
  --tag scope_live6_fc_v4_clean8000

if ($LASTEXITCODE -ne 0) {
  Write-Host "`nPilot failed with exit code $LASTEXITCODE." -ForegroundColor Red
  return
}

$run = Get-ChildItem -LiteralPath outputs\raw -Directory |
  Where-Object { $_.Name -like "*_azure_scope_live6_fc_v4_clean8000" } |
  Sort-Object LastWriteTime |
  Select-Object -Last 1

if ($null -eq $run) {
  Write-Host "Run completed, but its output directory could not be located." -ForegroundColor Red
  return
}

$records = Get-Content -LiteralPath (Join-Path $run.FullName "records.jsonl") |
  ForEach-Object { $_ | ConvertFrom-Json }
$parseFailures = @($records | Where-Object { $_.parse_failed }).Count
Write-Host "`nResponse-quality gate:" -ForegroundColor Cyan
Write-Host "  Agent-round records: $($records.Count)"
Write-Host "  Empty or unparsable responses: $parseFailures"
if ($parseFailures -ne 0) {
  Write-Host "Quality gate FAILED; refusing to analyze this run as valid data." -ForegroundColor Red
  return
}
Write-Host "Quality gate PASSED." -ForegroundColor Green

Write-Host "`nRunning conversation and statistical analysis..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe scripts\analyze_run.py $run.FullName
if ($LASTEXITCODE -ne 0) {
  Write-Host "Conversation analysis failed with exit code $LASTEXITCODE." -ForegroundColor Red
  return
}

& .\.venv\Scripts\python.exe scripts\analyze_statistics.py $run.FullName
if ($LASTEXITCODE -ne 0) {
  Write-Host "Statistical analysis failed with exit code $LASTEXITCODE." -ForegroundColor Red
  return
}

Write-Host "`nFinished successfully." -ForegroundColor Green
Write-Host "Run directory: $($run.FullName)"
