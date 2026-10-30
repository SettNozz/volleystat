param([switch]$Start)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $projectRoot 'results/vnl_stes_spatial_1248_overnight_20261027_v1'
$ready = Get-Content -LiteralPath (Join-Path $runRoot 'readiness.json') -Raw | ConvertFrom-Json
$plan = Get-Content -LiteralPath (Join-Path $runRoot 'launch_plan.json') -Raw | ConvertFrom-Json
foreach ($item in $plan.pinned_files) {
    if ((Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.sha256) {
        throw "Prepared input changed: $($item.path). Re-prepare before starting."
    }
}
Write-Output "Prepared STES continuation: epoch $($ready.continuation.start_epoch) through $($ready.continuation.final_epoch)."
Write-Output "Resolution 1248x702; 64-frame BiGRU; backbone chunks 16; 128 clips per epoch."
Write-Output "Output: $runRoot"
if (-not $Start) {
    Write-Output 'PREVIEW ONLY. No task registered and no training started. Use -Start when ready.'
    return
}
if (Test-Path -LiteralPath (Join-Path $runRoot 'report.json')) { throw 'This run already has a final report.' }
$taskName = 'Volleystat-STES-1248-20261027'
$worker = Join-Path $PSScriptRoot 'stes_overnight_worker.ps1'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -File "' + $worker + '"') -WorkingDirectory $projectRoot
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 14) -MultipleInstances IgnoreNew
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    if ($existing.Actions.Arguments -ne $action.Arguments) { throw 'Existing task has a different action; refusing to replace it.' }
    if ($existing.State -eq 'Running') { Write-Output 'The overnight task is already running.'; return }
} else {
    Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal -Description 'Manually started Volleystat STES continuation; no timed trigger.' | Out-Null
}
Start-ScheduledTask -TaskName $taskName
Write-Output 'Start requested through Windows Task Scheduler. Verify progress.json before claiming training is running.'
