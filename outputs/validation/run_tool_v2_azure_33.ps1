[CmdletBinding()]
param(
    [switch]$ShowRationales
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$scenarioData = Join-Path $repoRoot "data\scenarios_hidden_profile_candidates.json"
$preregistration = Join-Path $repoRoot "docs\PREREGISTRATION.md"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Repository virtual environment not found at $python. Run 'uv sync' first."
}
if (-not (Test-Path -LiteralPath $scenarioData -PathType Leaf)) {
    throw "Confirmatory scenario corpus not found at $scenarioData."
}

Set-Location -LiteralPath $repoRoot

$trackedChanges = @(git status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0) {
    throw "Could not verify Git status."
}
if ($trackedChanges.Count -gt 0) {
    throw (
        "Tracked worktree changes remain. Commit the frozen preregistration and " +
        "protocol before starting the confirmatory run.`n" +
        ($trackedChanges -join "`n")
    )
}

$preregText = Get-Content -Raw -Encoding UTF8 -LiteralPath $preregistration
if ($preregText -match '\[\s*(FILL|RECOMPUTE)' -or $preregText -match '\*\*Status:\*\* DRAFT') {
    throw "Preregistration still contains an unresolved placeholder or DRAFT status."
}

$deployment = (& $python -c (
    "from dotenv import load_dotenv; import os; " +
    "load_dotenv(); print(os.environ.get('AZURE_OPENAI_DEPLOYMENT', ''))"
)).Trim()
if ($LASTEXITCODE -ne 0 -or $deployment -ne "gpt-5-mini") {
    throw (
        "Confirmatory deployment must be exactly 'gpt-5-mini'; " +
        "configured value is '$deployment'."
    )
}

$decodedScenarios = Get-Content -Raw -Encoding UTF8 -LiteralPath $scenarioData |
    ConvertFrom-Json
$scenarios = @($decodedScenarios)
if ($scenarios.Count -ne 33) {
    throw "Expected exactly 33 scenarios; found $($scenarios.Count)."
}
$primaryVariants = @(
    $scenarios.assumption_variants |
        Where-Object { $_.semantic_type -in @("factual", "causal") }
)
if ($primaryVariants.Count -ne 66) {
    throw "Expected exactly 66 factual/causal variants; found $($primaryVariants.Count)."
}

$nSeeds = 3
$nArms = 6
$baselineReps = 1
$expectedSeeds = $primaryVariants.Count * $nSeeds
$expectedInjectionConversations = $primaryVariants.Count * $nSeeds * $nArms
$expectedBaselineConversations = $primaryVariants.Count * $baselineReps
$expectedConversations = $expectedInjectionConversations + $expectedBaselineConversations
$expectedRecords = $expectedConversations * 4 * 4
$tag = "confirmatory_scope33_gpt5mini"

Write-Host "Confirmatory Azure tool-v2 evidence-delivery run" -ForegroundColor Cyan
Write-Host "  Deployment: azure:$deployment"
Write-Host "  Scenarios: 33"
Write-Host "  Primary semantic types: factual + causal (66 variants)"
Write-Host "  Primary arms: none / testimony / exact relay / direct recipient delivery"
Write-Host "  Mechanism controls: prompt receipt + non-diagnostic tool placebo"
Write-Host "  Reference: 1 uncorrupted conversation per scenario/type"
Write-Host "  Design: 3 frozen seeds per variant, 4 rounds, fully connected mesh"
Write-Host "  Expected seeds: $expectedSeeds"
Write-Host "  Expected injection conversations: $expectedInjectionConversations"
Write-Host "  Expected reference conversations: $expectedBaselineConversations"
Write-Host "  Expected total conversations: $expectedConversations"
Write-Host "  Expected agent-round records: $expectedRecords"
Write-Host ""
Write-Host "Output is unbuffered; results will appear here as calls finish." -ForegroundColor DarkCyan
Write-Host ""

$pilotArgs = @(
    "scripts\run_pilot.py",
    "--client", "azure",
    "--data", "data\scenarios_hidden_profile_candidates.json",
    "--variants", "factual", "causal",
    "--scope-ladder",
    "--scope-protocol", "tool_v2",
    "--scope-controls",
    "--topologies", "fully_connected_mesh",
    "--n-seeds", "$nSeeds",
    "--baseline-reps", "$baselineReps",
    "--skip-independent",
    "--rounds", "4",
    "--temperature", "none",
    "--max-completion-tokens", "8000",
    "--verbose",
    "--tag", $tag
)
if ($ShowRationales) {
    $pilotArgs += "--show-rationales"
}

& $python @pilotArgs
if ($LASTEXITCODE -ne 0) {
    throw "Azure confirmatory run failed with exit code $LASTEXITCODE."
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
$results = @(Get-Content -LiteralPath (Join-Path $run.FullName "results.jsonl"))
$seeds = @(Get-Content -LiteralPath (Join-Path $run.FullName "seeds.jsonl"))
$parseFailures = @($records | Where-Object { $_.parse_failed }).Count
$generationFailures = @($records | Where-Object { $_.generation_failed }).Count
$toolErrors = @($records | Where-Object {
    $null -ne $_.tool_trace -and $_.tool_trace.success -eq $false
}).Count
$conditionCount = @($manifest.protocol.injection_conditions).Count

$passed = (
    $manifest.status.state -eq "complete" -and
    $manifest.client.model_identifier -eq "azure:gpt-5-mini" -and
    $manifest.client.request_timeout_seconds -eq 120 -and
    $manifest.client.sdk_max_retries -eq 0 -and
    $manifest.protocol.scope_protocol -eq "tool_v2" -and
    $manifest.protocol.scope_controls -eq $true -and
    $conditionCount -eq $nArms -and
    $seeds.Count -eq $expectedSeeds -and
    $results.Count -eq $expectedConversations -and
    $records.Count -eq $expectedRecords -and
    $parseFailures -eq 0 -and
    $generationFailures -eq 0 -and
    $toolErrors -eq 0
)

Write-Host "`nConfirmatory response and tool-integrity gate" -ForegroundColor Cyan
Write-Host "  Run directory: $($run.FullName)"
Write-Host "  Manifest state: $($manifest.status.state)"
Write-Host "  Model: $($manifest.client.model_identifier)"
Write-Host "  Conditions: $($manifest.protocol.injection_conditions -join ', ')"
Write-Host "  Seeds: $($seeds.Count)/$expectedSeeds"
Write-Host "  Conversations: $($results.Count)/$expectedConversations"
Write-Host "  Agent-round records: $($records.Count)/$expectedRecords"
Write-Host "  Parse failures: $parseFailures"
Write-Host "  Generation failures: $generationFailures"
Write-Host "  Failed tool traces: $toolErrors"
if (-not $passed) {
    throw "Confirmatory run FAILED its integrity gate; do not report it as valid data."
}
Write-Host "  RESULT: PASS" -ForegroundColor Green

Write-Host "`nRunning frozen conversation and statistical analysis..." -ForegroundColor Cyan
& $python scripts\analyze_run.py $run.FullName
if ($LASTEXITCODE -ne 0) {
    throw "Conversation analysis failed with exit code $LASTEXITCODE."
}
$statisticsPath = Join-Path $run.FullName "analysis\statistics_confirmatory.json"
& $python scripts\analyze_statistics.py $run.FullName `
    --by-semantic-type --json $statisticsPath
if ($LASTEXITCODE -ne 0) {
    throw "Statistical analysis failed with exit code $LASTEXITCODE."
}

Write-Host "`nConfirmatory run and frozen analysis finished successfully." -ForegroundColor Green
Write-Host "Run directory: $($run.FullName)"
Write-Host "Statistics: $statisticsPath"
