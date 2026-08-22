[CmdletBinding()]
param(
    [switch]$IncludeControls,
    [switch]$ShowRationales
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Repository virtual environment not found at $python. Run 'uv sync' first."
}

Set-Location -LiteralPath $repoRoot
$controlLabel = if ($IncludeControls) { "primary_plus_controls" } else { "primary_only" }
$tag = "tool_v2_mock1_$controlLabel"
$expectedArms = if ($IncludeControls) { 6 } else { 4 }
$expectedRecords = $expectedArms * 4 * 4

Write-Host "Protocol-v2 mock smoke" -ForegroundColor Cyan
Write-Host "  Scenario: vendor_001"
Write-Host "  Semantic type: factual"
Write-Host "  Primary arms: none / testimony / exact relay / direct forced tool"
Write-Host "  Optional controls: $(if ($IncludeControls) { 'included (prompt receipt + tool placebo)' } else { 'not included' })"
Write-Host "  Design: 1 frozen seed, 4 rounds, fully connected mesh"
Write-Host "  Expected measured records: $expectedRecords"
Write-Host ""

$pilotArgs = @(
    "scripts\run_pilot.py",
    "--client", "mock",
    "--scenario", "vendor_001",
    "--variants", "factual",
    "--scope-ladder",
    "--scope-protocol", "tool_v2",
    "--topologies", "fully_connected_mesh",
    "--n-seeds", "1",
    "--skip-baselines",
    "--skip-independent",
    "--rounds", "4",
    "--verbose",
    "--tag", $tag
)
if ($IncludeControls) {
    $pilotArgs += "--scope-controls"
}
if ($ShowRationales) {
    $pilotArgs += "--show-rationales"
}

& $python @pilotArgs
if ($LASTEXITCODE -ne 0) {
    throw "Mock smoke failed with exit code $LASTEXITCODE."
}

$run = Get-ChildItem -LiteralPath "outputs\raw" -Directory |
    Where-Object { $_.Name -like "*_mock_$tag" } |
    Sort-Object LastWriteTime |
    Select-Object -Last 1
if ($null -eq $run) {
    throw "Run completed, but the output directory for tag '$tag' was not found."
}

$manifest = Get-Content -Raw -LiteralPath (Join-Path $run.FullName "run_manifest.json") |
    ConvertFrom-Json
$records = @(Get-Content -LiteralPath (Join-Path $run.FullName "records.jsonl") |
    ForEach-Object { $_ | ConvertFrom-Json })
$parseFailures = @($records | Where-Object { $_.parse_failed }).Count
$executedCalls = @($records | Where-Object { $_.tool_call_executed -eq $true }).Count
$replayedTraces = @($records | Where-Object { $_.tool_trace_replayed -eq $true }).Count
$conditionCount = @($manifest.protocol.injection_conditions).Count

$passed = (
    $manifest.status.state -eq "complete" -and
    $manifest.protocol.scope_protocol -eq "tool_v2" -and
    $conditionCount -eq $expectedArms -and
    $records.Count -eq $expectedRecords -and
    $parseFailures -eq 0 -and
    $executedCalls -gt 0 -and
    $replayedTraces -gt 0
)

Write-Host "`nSmoke integrity summary" -ForegroundColor Cyan
Write-Host "  Manifest state: $($manifest.status.state)"
Write-Host "  Protocol: $($manifest.protocol.scope_protocol)"
Write-Host "  Conditions: $($manifest.protocol.injection_conditions -join ', ')"
Write-Host "  Agent-round records: $($records.Count)/$expectedRecords"
Write-Host "  Parse failures: $parseFailures"
Write-Host "  Measured cells with executed tool calls: $executedCalls"
Write-Host "  Measured cells with replayed anchor traces: $replayedTraces"
Write-Host "  Run directory: $($run.FullName)"

if (-not $passed) {
    throw "Protocol-v2 mock smoke FAILED its integrity gate."
}
Write-Host "  RESULT: PASS" -ForegroundColor Green

Write-Host "`nRunning no-cost analysis..." -ForegroundColor Cyan
& $python scripts\analyze_run.py $run.FullName
if ($LASTEXITCODE -ne 0) {
    throw "Conversation analysis failed with exit code $LASTEXITCODE."
}
& $python scripts\analyze_statistics.py $run.FullName
if ($LASTEXITCODE -ne 0) {
    throw "Statistical analysis failed with exit code $LASTEXITCODE."
}

Write-Host "`nMock smoke and analysis finished successfully." -ForegroundColor Green
Write-Host "Run directory: $($run.FullName)"
