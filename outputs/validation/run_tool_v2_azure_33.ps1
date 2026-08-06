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
$expectedScenarioSha256 = "7f0f31556b2b9543815eac77272ab00f91804deb3128d33b5c7c12d48ac07fbf"
$attribution = "both"
$attributionModes = @("anonymous", "assigned")
$topologies = @("looped_chain", "neutral_star", "fully_connected_mesh")
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
if ($LASTEXITCODE -ne 0 -or $deployment -ne "gpt-5.4-mini") {
    throw (
        "Confirmatory deployment must be exactly 'gpt-5.4-mini'; " +
        "configured value is '$deployment'."
    )
}

$scenarioSha256 = (Get-FileHash -LiteralPath $scenarioData -Algorithm SHA256).Hash.ToLowerInvariant()
if ($scenarioSha256 -ne $expectedScenarioSha256) {
    throw (
        "Confirmatory scenario corpus hash mismatch. Expected " +
        "$expectedScenarioSha256; found $scenarioSha256."
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
$nTopologies = $topologies.Count
$nAttributionModes = $attributionModes.Count
$baselineReps = 1
$expectedSeeds = $primaryVariants.Count * $nSeeds
$maximumInjectionPerCoordinate = $primaryVariants.Count * $nSeeds * $nArms
$expectedBaselinePerCoordinate = $primaryVariants.Count * $baselineReps
$maximumInjectionConversations = (
    $maximumInjectionPerCoordinate * $nTopologies * $nAttributionModes
)
$expectedBaselineConversations = (
    $expectedBaselinePerCoordinate * $nTopologies * $nAttributionModes
)
$maximumConversations = $maximumInjectionConversations + $expectedBaselineConversations
$maximumRecords = $maximumConversations * 4 * 4
$tag = "confirmatory_scope33_gpt54mini"

Write-Host "Confirmatory Azure tool-v2 evidence-delivery run" -ForegroundColor Cyan
Write-Host "  Deployment: azure:$deployment"
Write-Host "  Scenarios: 33"
Write-Host "  Primary semantic types: factual + causal (66 variants)"
Write-Host "  Primary arms: none / testimony / exact relay / direct recipient delivery"
Write-Host "  Mechanism controls: prompt receipt + non-diagnostic tool placebo"
Write-Host "  Reference: 1 uncorrupted conversation per scenario/type"
Write-Host "  Design: 3 frozen seeds per variant, 4 rounds, full factorial"
Write-Host "  Topologies: $($topologies -join ', ')"
Write-Host "  Speaker attribution: $($attributionModes -join ' + ')"
Write-Host "  Primary cell: fully_connected_mesh + assigned"
Write-Host "  Scenario SHA-256: $scenarioSha256"
Write-Host "  Expected seeds: $expectedSeeds"
Write-Host "  Maximum injection conversations: $maximumInjectionConversations"
Write-Host "  Expected reference conversations: $expectedBaselineConversations"
Write-Host "  Maximum total conversations: $maximumConversations"
Write-Host "  Maximum agent-round records: $maximumRecords"
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
    "--topologies", $topologies[0], $topologies[1], $topologies[2],
    "--attribution", $attribution,
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
$results = @(Get-Content -LiteralPath (Join-Path $run.FullName "results.jsonl") |
    ForEach-Object { $_ | ConvertFrom-Json })
$seeds = @(Get-Content -LiteralPath (Join-Path $run.FullName "seeds.jsonl") |
    ForEach-Object { $_ | ConvertFrom-Json })
$replayableSeeds = @($seeds | Where-Object {
    -not [string]::IsNullOrWhiteSpace([string]$_.answer)
}).Count
$expectedInjectionPerCoordinate = $replayableSeeds * $nArms
$expectedInjectionConversations = (
    $expectedInjectionPerCoordinate * $nTopologies * $nAttributionModes
)
$expectedConversations = $expectedInjectionConversations + $expectedBaselineConversations
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
$badResultAttribution = @($results | Where-Object {
    $_.attribution -notin $attributionModes
}).Count
$badRecordTopology = @($records | Where-Object {
    $_.topology -notin $topologies
}).Count
$badResultTopology = @($results | Where-Object {
    $_.topology -notin $topologies
}).Count
$coordinateMismatches = 0
foreach ($topology in $topologies) {
    foreach ($mode in $attributionModes) {
        $injectionAtCoordinate = @($results | Where-Object {
            $_.topology -eq $topology -and
            $_.attribution -eq $mode -and
            $_.condition -in $manifest.protocol.injection_conditions
        }).Count
        $baselineAtCoordinate = @($results | Where-Object {
            $_.topology -eq $topology -and
            $_.attribution -eq $mode -and
            $_.condition -eq "no_injection_baseline"
        }).Count
        if (
            $injectionAtCoordinate -ne $expectedInjectionPerCoordinate -or
            $baselineAtCoordinate -ne $expectedBaselinePerCoordinate
        ) {
            $coordinateMismatches += 1
        }
    }
}

$passed = (
    $manifest.status.state -eq "complete" -and
    $manifest.client.model_identifier -eq "azure:gpt-5.4-mini" -and
    $manifest.client.request_timeout_seconds -eq 120 -and
    $manifest.client.sdk_max_retries -eq 0 -and
    $manifest.protocol.scope_protocol -eq "tool_v2" -and
    $manifest.protocol.scope_controls -eq $true -and
    $manifest.protocol.attribution -eq $attribution -and
    ($manifest.protocol.topologies -join "|") -eq ($topologies -join "|") -and
    $manifest.protocol.scope_primary_topology -eq "fully_connected_mesh" -and
    $manifest.protocol.scope_primary_attribution -eq "assigned" -and
    $manifest.scenario_data.sha256 -eq $expectedScenarioSha256 -and
    $conditionCount -eq $nArms -and
    $seeds.Count -eq $expectedSeeds -and
    $results.Count -eq $expectedConversations -and
    $records.Count -eq $expectedRecords -and
    $badRecordAttribution -eq 0 -and
    $badResultAttribution -eq 0 -and
    $badRecordTopology -eq 0 -and
    $badResultTopology -eq 0 -and
    $coordinateMismatches -eq 0 -and
    $toolErrors -eq 0
)

Write-Host "`nConfirmatory response and tool-integrity gate" -ForegroundColor Cyan
Write-Host "  Run directory: $($run.FullName)"
Write-Host "  Manifest state: $($manifest.status.state)"
Write-Host "  Model: $($manifest.client.model_identifier)"
Write-Host "  Attribution: $($manifest.protocol.attribution)"
Write-Host "  Topologies: $($manifest.protocol.topologies -join ', ')"
Write-Host "  Scenario SHA-256: $($manifest.scenario_data.sha256)"
Write-Host "  Conditions: $($manifest.protocol.injection_conditions -join ', ')"
Write-Host "  Seeds: $($seeds.Count)/$expectedSeeds"
Write-Host "  Replayable seeds: $replayableSeeds/$expectedSeeds"
Write-Host "  Conversations: $($results.Count)/$expectedConversations"
Write-Host "  Agent-round records: $($records.Count)/$expectedRecords"
Write-Host "  Record attribution mismatches: $badRecordAttribution"
Write-Host "  Result attribution mismatches: $badResultAttribution"
Write-Host "  Record topology mismatches: $badRecordTopology"
Write-Host "  Result topology mismatches: $badResultTopology"
Write-Host "  Incomplete factorial coordinates: $coordinateMismatches"
Write-Host "  Parse failures: $parseFailures"
Write-Host "  Generation failures: $generationFailures"
Write-Host "  Failed tool traces: $toolErrors"
if (-not $passed) {
    throw "Confirmatory run FAILED its integrity gate; do not report it as valid data."
}
Write-Host "  RESULT: PASS" -ForegroundColor Green
if ($parseFailures -gt 0 -or $generationFailures -gt 0) {
    Write-Warning (
        "The run is structurally complete but contains response failures. " +
        "Failed final agent responses receive no vote and are excluded from " +
        "agent-level denominators; report the counts and sensitivity analyses."
    )
}

Write-Host "`nRunning frozen conversation and statistical analysis..." -ForegroundColor Cyan
& $python scripts\analyze_run.py $run.FullName
if ($LASTEXITCODE -ne 0) {
    throw "Conversation analysis failed with exit code $LASTEXITCODE."
}
$statisticsPath = Join-Path $run.FullName "analysis\statistics_confirmatory.json"
& $python scripts\analyze_statistics.py $run.FullName `
    --by-topology --by-attribution --by-semantic-type --factorial `
    --json $statisticsPath
if ($LASTEXITCODE -ne 0) {
    throw "Statistical analysis failed with exit code $LASTEXITCODE."
}

Write-Host "`nConfirmatory run and frozen analysis finished successfully." -ForegroundColor Green
Write-Host "Run directory: $($run.FullName)"
Write-Host "Statistics: $statisticsPath"
