# No-authority ablation. Same protocol as run_tool_v2_azure_33.ps1 on the same
# 33 scenarios, but against the corpus whose pooling rule drops the sentence
# making an assigned reviewer's report authoritative and sufficient. Mesh only:
# the ablation asks whether the authority clause carries the effect, which is a
# within-mesh question, so the topology factor is held fixed rather than crossed.
#
# Unlike the confirmatory launcher this does NOT pin a deployment. Both models
# reported in the ablation were run through this same protocol, so the
# deployment is a parameter and is recorded in the run manifest.
#
# Results from this ablation are reported separately and never pooled with the
# confirmatory run: it is a different corpus, with a different hash.

[CmdletBinding()]
param(
    [switch]$ShowRationales,
    [int]$NSeeds = 3
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$scenarioData = Join-Path $repoRoot "data\scenarios_no_authority.json"
# The gate. Hashes the parsed corpus, so it is identical on every platform.
$expectedScenarioCanonicalSha256 = "9100e196c28916bc96329f51a33cd3deb8113fecb5058fa19110839acd3f8ffc"
# Recorded provenance, not a gate: byte hash, so a CRLF checkout changes it
# without changing a single scenario.
$expectedScenarioSha256 = "af6ce27f6b1ef7900eea72ab23aeffe37d0e8b5b7449351c7d0c504e413f6903"
$attribution = "both"
$attributionModes = @("anonymous", "assigned")
$nArms = 6
$nTopologies = 1
$nAttributionModes = $attributionModes.Count

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Repository virtual environment not found at $python. Run 'uv sync' first."
}
if (-not (Test-Path -LiteralPath $scenarioData -PathType Leaf)) {
    throw "Ablation scenario corpus not found at $scenarioData."
}

Set-Location -LiteralPath $repoRoot

$trackedChanges = @(git status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0) {
    throw "Could not verify Git status."
}
if ($trackedChanges.Count -gt 0) {
    throw (
        "Tracked worktree changes remain. Commit the frozen protocol " +
        "before starting the ablation run.`n" + ($trackedChanges -join "`n")
    )
}

$deployment = (& $python -c (
    "from dotenv import load_dotenv; import os; " +
    "load_dotenv(); print(os.environ.get('AZURE_OPENAI_DEPLOYMENT', ''))"
)).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($deployment)) {
    throw "AZURE_OPENAI_DEPLOYMENT is not configured; set it in .env."
}

$scenarioCanonicalSha256 = (& $python -c (
    "import pathlib, sys; " +
    "from assumption_propagation.reproducibility import canonical_json_sha256; " +
    "print(canonical_json_sha256(pathlib.Path(sys.argv[1])))"
) $scenarioData)
if ($LASTEXITCODE -ne 0) {
    throw "Could not compute the canonical scenario corpus hash."
}
$scenarioCanonicalSha256 = $scenarioCanonicalSha256.Trim()
if ($scenarioCanonicalSha256 -ne $expectedScenarioCanonicalSha256) {
    throw (
        "Ablation corpus content mismatch. Expected canonical " +
        "$expectedScenarioCanonicalSha256; found $scenarioCanonicalSha256."
    )
}

$scenarios = @(Get-Content -Raw -LiteralPath $scenarioData | ConvertFrom-Json)
if ($scenarios.Count -ne 33) {
    throw "Expected exactly 33 scenarios; found $($scenarios.Count)."
}
$primaryVariants = @(
    $scenarios | ForEach-Object { $_.assumption_variants } |
        Where-Object { $_.semantic_type -in @("factual", "causal") }
)
if ($primaryVariants.Count -ne 66) {
    throw "Expected exactly 66 factual/causal variants; found $($primaryVariants.Count)."
}

$tag = "ablation_noauthority"

Write-Host "No-authority ablation" -ForegroundColor Cyan
Write-Host "  Deployment: azure:$deployment"
Write-Host "  Scenarios: 33   Primary variants: 66 (factual + causal)"
Write-Host "  Arms: none / testimony / exact relay / direct, plus prompt-receipt and placebo controls"
Write-Host "  Topology: fully_connected_mesh only"
Write-Host "  Speaker attribution: $($attributionModes -join ' + ')"
Write-Host "  Seeds per variant: $NSeeds   Rounds: 4"
Write-Host "  Corpus canonical SHA-256: $scenarioCanonicalSha256"
Write-Host "  Corpus raw SHA-256 (provenance): $expectedScenarioSha256"
Write-Host ""

$pilotArgs = @(
    "scripts\run_pilot.py",
    "--client", "azure",
    "--data", "data\scenarios_no_authority.json",
    "--variants", "factual", "causal",
    "--scope-ladder",
    "--scope-protocol", "tool_v2",
    "--scope-controls",
    "--topologies", "fully_connected_mesh",
    "--attribution", $attribution,
    "--n-seeds", "$NSeeds",
    "--skip-baselines",
    "--skip-independent",
    "--rounds", "4",
    "--temperature", "none",
    "--max-completion-tokens", "1000",
    "--verbose",
    "--tag", $tag
)
if ($ShowRationales) { $pilotArgs += "--show-rationales" }

& $python @pilotArgs
if ($LASTEXITCODE -ne 0) {
    throw "Ablation run failed with exit code $LASTEXITCODE."
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
$seeds = @(Get-Content -LiteralPath (Join-Path $run.FullName "seeds.jsonl") |
    ForEach-Object { $_ | ConvertFrom-Json })
$replayableSeeds = @($seeds | Where-Object {
    -not [string]::IsNullOrWhiteSpace([string]$_.answer)
}).Count
$expectedConversations = $replayableSeeds * $nArms * $nTopologies * $nAttributionModes
$expectedRecords = $expectedConversations * 4 * 4
$parseFailures = @($records | Where-Object { $_.parse_failed }).Count
$generationFailures = @($records | Where-Object { $_.generation_failed }).Count
$toolErrors = @($records | Where-Object {
    $null -ne $_.tool_trace -and $_.tool_trace.success -eq $false
}).Count
$conditionCount = @($manifest.protocol.injection_conditions).Count
$badRecordAttribution = @($records | Where-Object {
    $_.attribution -notin $attributionModes
}).Count

Write-Host "`nAblation integrity summary" -ForegroundColor Cyan
Write-Host "  Manifest state: $($manifest.status.state)"
Write-Host "  Protocol: $($manifest.protocol.scope_protocol)"
Write-Host "  Arms: $conditionCount/$nArms"
Write-Host "  Replayable seeds: $replayableSeeds"
Write-Host "  Agent-round records: $($records.Count)/$expectedRecords"
Write-Host "  Parse failures: $parseFailures   Generation failures: $generationFailures"
Write-Host "  Tool errors: $toolErrors"
Write-Host "  Records with unexpected attribution: $badRecordAttribution"
Write-Host "  Run directory: $($run.FullName)"

$passed = (
    $manifest.status.state -eq "complete" -and
    $manifest.protocol.scope_protocol -eq "tool_v2" -and
    $conditionCount -eq $nArms -and
    $records.Count -eq $expectedRecords -and
    $parseFailures -eq 0 -and
    $generationFailures -eq 0 -and
    $toolErrors -eq 0 -and
    $badRecordAttribution -eq 0
)
if (-not $passed) {
    Write-Host "  RESULT: FAIL" -ForegroundColor Red
    Write-Host ""
    Write-Host "If records were lost to transient API errors, re-collect the affected" -ForegroundColor Yellow
    Write-Host "scenarios and splice them in with scripts/merge_scenarios.py rather than" -ForegroundColor Yellow
    Write-Host "re-running the whole grid. Replacement is by whole scenario because" -ForegroundColor Yellow
    Write-Host "scenario is the bootstrap cluster." -ForegroundColor Yellow
    throw "No-authority ablation FAILED its integrity gate."
}
Write-Host "  RESULT: PASS" -ForegroundColor Green

Write-Host "`nRunning analysis..." -ForegroundColor Cyan
& $python scripts\analyze_run.py $run.FullName
if ($LASTEXITCODE -ne 0) { throw "Conversation analysis failed with exit code $LASTEXITCODE." }
& $python scripts\analyze_statistics.py $run.FullName
if ($LASTEXITCODE -ne 0) { throw "Statistical analysis failed with exit code $LASTEXITCODE." }

Write-Host "`nAblation finished successfully." -ForegroundColor Green
Write-Host "Run directory: $($run.FullName)"
