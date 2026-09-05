# Metadata only. Run again in the actual activation turn; this receipt expires
# when its hosted run ends and never authorizes installation or restart.
param([string]$OperationId)
$ErrorActionPreference = 'Stop'
$routerRepo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$routerLead = 'lead-1032891d-b8b5-4bbc-a05e-37284eaef9e3'
$routerRuns = 'C:\Users\Stan\Documents\GitHub\claude-orchestrator\state\runs'
$routerAncestry = @()
$routerProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $PID"
for ($i = 0; $i -lt 12 -and $null -ne $routerProcess; $i++) {
    $routerAncestry += $routerProcess
    if ($routerProcess.ParentProcessId -eq 0) { break }
    $routerProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($routerProcess.ParentProcessId)"
}
$routerMatches = @()
foreach ($routerRun in Get-ChildItem -LiteralPath (Join-Path $routerRuns $routerLead) -Directory) {
    $routerHostFile = Join-Path $routerRun.FullName 'host.json'
    $routerOutcomeFile = Join-Path $routerRun.FullName 'outcome.json'
    if (-not (Test-Path -LiteralPath $routerHostFile) -or (Test-Path -LiteralPath $routerOutcomeFile)) { continue }
    $routerHostRecord = Get-Content -LiteralPath $routerHostFile -Raw | ConvertFrom-Json
    if ($routerHostRecord.leadId -ne $routerLead -or $routerHostRecord.runId -ne $routerRun.Name) { continue }
    $routerHostProcess = $routerAncestry | Where-Object ProcessId -eq $routerHostRecord.hostPid | Select-Object -First 1
    if ($null -eq $routerHostProcess) { continue }
    $routerRecordedStart = [DateTimeOffset]::FromUnixTimeMilliseconds([long]$routerHostRecord.startedAt).UtcDateTime
    if ([Math]::Abs(($routerHostProcess.CreationDate.ToUniversalTime() - $routerRecordedStart).TotalSeconds) -gt 5) { continue }
    $routerMatches += [pscustomobject]@{Run=$routerRun;Record=$routerHostRecord;HostProcess=$routerHostProcess;HostFile=$routerHostFile;OutcomeFile=$routerOutcomeFile}
}
if ($routerMatches.Count -ne 1) { throw 'Cannot uniquely identify this live hosted router run' }
$routerMatch = $routerMatches[0]
$routerEngine = $routerAncestry | Where-Object { $_.Name -in @('codex.exe','claude.exe') } | Select-Object -First 1
if ($null -eq $routerEngine) { throw 'Cannot identify the active engine process' }
$routerLockPath = Join-Path $routerRepo 'docs\credential-error-verification-lock.json'
$routerLock = Get-Content -LiteralPath $routerLockPath -Raw | ConvertFrom-Json
foreach ($entry in $routerLock.files.PSObject.Properties) {
    $actual = (Get-FileHash -LiteralPath (Join-Path $routerRepo $entry.Name) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $entry.Value) { throw 'A frozen verification dependency changed' }
}
function Router-ProcessFields($value) {
    [ordered]@{pid=$value.ProcessId;parent_pid=$value.ParentProcessId;name=$value.Name;executable=$value.ExecutablePath;created_at_utc=$value.CreationDate.ToUniversalTime().ToString('o')}
}
$routerReceipt = [ordered]@{
    observed_at_utc=[DateTime]::UtcNow.ToString('o')
    status='live-run-metadata-only'
    lead_id=$routerLead
    run_id=$routerMatch.Record.runId
    operation_id=$OperationId
    host=(Router-ProcessFields $routerMatch.HostProcess)
    engine=(Router-ProcessFields $routerEngine)
    recorded_child_pid=$routerMatch.Record.childPid
    host_record_path=$routerMatch.HostFile
    host_record_sha256=(Get-FileHash -LiteralPath $routerMatch.HostFile -Algorithm SHA256).Hash.ToLowerInvariant()
    outcome_path=$routerMatch.OutcomeFile
    outcome_exists=(Test-Path -LiteralPath $routerMatch.OutcomeFile)
    verification_lock_sha256=(Get-FileHash -LiteralPath $routerLockPath -Algorithm SHA256).Hash.ToLowerInvariant()
    valid_only_while_this_run_is_active=$true
    restart_survival_observed=$false
    local_handoff_accepted=$true
    activation_authorized=$false
}
if ($routerReceipt.outcome_exists) { throw 'Run ended while capturing its identity' }
$routerOutput = Join-Path $routerRepo 'docs\credential-error-operator-current.json'
$routerJson = ($routerReceipt | ConvertTo-Json -Depth 5) + [Environment]::NewLine
[IO.File]::WriteAllText($routerOutput, $routerJson, [Text.UTF8Encoding]::new($false))
$routerReceipt | ConvertTo-Json -Depth 5
