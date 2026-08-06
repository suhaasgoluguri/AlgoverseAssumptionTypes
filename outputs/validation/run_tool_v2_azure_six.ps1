[CmdletBinding()]
param(
    [switch]$IncludeControls,
    [switch]$ShowRationales
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$scenarioSubset = Join-Path $repoRoot "outputs\validation\scenarios_scope_live6.json"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Repository virtual environment not found at $python. Run 'uv sync' first."
}
if (-not (Test-Path -LiteralPath $scenarioSubset -PathType Leaf)) {
    throw "Six-scenario subset not found at $scenarioSubset."
}

Set-Location -LiteralPath $repoRoot
$decodedSubset = Get-Content -Raw -LiteralPath $scenarioSubset | ConvertFrom-Json
$subsetRows = @($decodedSubset)
if ($subsetRows.Count -ne 6) {
    throw "Expected exactly six scenarios in $scenarioSubset; found $($subsetRows.Count)."
}

$controlLabel = if ($IncludeControls) { "primary_plus_controls" } else { "primary_only" }
$tag = "tool_v2_azure6_$controlLabel"
$expectedArms = if ($IncludeControls) { 6 } else { 4 }
$expectedConversations = 6 * 2 * $expectedArms
$expectedRecords = $expectedConversations * 4 * 4

Write-Host "Azure protocol-v2 evidence-scope run" -ForegroundColor Cyan
Write-Host "  Scenarios (6): $($subsetRows.scenario_id -join ', ')"
Write-Host "  Primary semantic types: factual + causal (framing excluded)"
Write-Host "  Primary arms: none / testimony / exact relay / direct forced tool"
Write-Host "  Optional controls: $(if ($IncludeControls) { 'included (prompt receipt + tool placebo)' } else { 'not included' })"
Write-Host "  Design: 1 frozen seed per variant, 4 rounds, fully connected mesh"
Write-Host "  Expected measured conversations: $expectedConversations"
Write-Host "  Expected measured agent-round records: $expectedRecords"
Write-Host ""
Write-Host "The runner is unbuffered; arm results will appear in this terminal as calls finish." -ForegroundColor DarkCyan
Write-Host ""

$pilotArgs = @(
    "scripts\run_pilot.py",
    "--client", "azure",
    "--data", "outputs\validation\scenarios_scope_live6.json",
    "--variants", "factual", "causal",
    "--scope-ladder",
    "--scope-protocol", "tool_v2",
    "--topologies", "fully_connected_mesh",
    "--n-seeds", "1",
    "--skip-baselines",
    "--skip-independent",
    "--rounds", "4",
    "--temperature", "none",
    "--max-completion-tokens", "1000",
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
    throw "Azure tool-v2 run failed with exit code $LASTEXITCODE."
}

$run = Get-ChildItem -LiteralPath "outputs\raw" -Directory |
    Where-Object { $_.Name -like "*_azure_$tag" } |
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
$generationFailures = @($records | Where-Object { $_.generation_failed }).Count
$executedCalls = @($records | Where-Object { $_.tool_call_executed -eq $true }).Count
$replayedTraces = @($records | Where-Object { $_.tool_trace_replayed -eq $true }).Count
$toolErrors = @($records | Where-Object {
    $null -ne $_.tool_trace -and $_.tool_trace.success -eq $false
}).Count
$conditionCount = @($manifest.protocol.injection_conditions).Count

$passed = (
    $manifest.status.state -eq "complete" -and
    $manifest.protocol.scope_protocol -eq "tool_v2" -and
    $conditionCount -eq $expectedArms -and
    $records.Count -eq $expectedRecords -and
    $parseFailures -eq 0 -and
    $generationFailures -eq 0 -and
    $toolErrors -eq 0 -and
    $executedCalls -gt 0 -and
    $replayedTraces -gt 0
)

Write-Host "`nResponse and tool-integrity gate" -ForegroundColor Cyan
Write-Host "  Manifest state: $($manifest.status.state)"
Write-Host "  Protocol: $($manifest.protocol.scope_protocol)"
Write-Host "  Conditions: $($manifest.protocol.injection_conditions -join ', ')"
Write-Host "  Agent-round records: $($records.Count)/$expectedRecords"
Write-Host "  Parse failures: $parseFailures"
Write-Host "  Generation failures: $generationFailures"
Write-Host "  Failed tool traces: $toolErrors"
Write-Host "  Measured cells with executed tool calls: $executedCalls"
Write-Host "  Measured cells with replayed anchor traces: $replayedTraces"
Write-Host "  Run directory: $($run.FullName)"

if (-not $passed) {
    throw "Azure protocol-v2 run FAILED its integrity gate; do not analyze it as valid data."
}
Write-Host "  RESULT: PASS" -ForegroundColor Green

Write-Host "`nRunning conversation and statistical analysis..." -ForegroundColor Cyan
& $python scripts\analyze_run.py $run.FullName
if ($LASTEXITCODE -ne 0) {
    throw "Conversation analysis failed with exit code $LASTEXITCODE."
}
& $python scripts\analyze_statistics.py $run.FullName
if ($LASTEXITCODE -ne 0) {
    throw "Statistical analysis failed with exit code $LASTEXITCODE."
}

Write-Host "`nAzure protocol-v2 run and analysis finished successfully." -ForegroundColor Green
Write-Host "Run directory: $($run.FullName)"
