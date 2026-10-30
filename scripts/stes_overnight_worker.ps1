$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$runRoot = Join-Path $projectRoot 'results/vnl_stes_spatial_1248_overnight_20261027_v1'
$plan = Get-Content -LiteralPath (Join-Path $runRoot 'launch_plan.json') -Raw | ConvertFrom-Json
foreach ($item in $plan.pinned_files) {
    if ((Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.sha256) { throw "Prepared input changed: $($item.path)" }
}
if (Test-Path -LiteralPath (Join-Path $runRoot 'cancel.request')) { throw 'Cancellation requested; refusing to launch.' }
$env:PYTHONPATH = (Join-Path $projectRoot '.runtime/vnl_stes_deps') + ';' + (Join-Path $projectRoot 'volley_env_311/Lib/site-packages')
$env:PYTHONUTF8 = '1'
Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public static class STESAwake { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
$awake = $false
try {
    $awake = [STESAwake]::SetThreadExecutionState([uint32]2147483649) -ne 0
    if (-not $awake) { throw 'Cannot temporarily prevent system sleep for the overnight run.' }
    @{status='launching';started_at=[DateTime]::UtcNow.ToString('o');worker_pid=$PID} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'worker_status.json') -Encoding UTF8
    $trainArgs = @($plan.arguments | ForEach-Object { [string]$_ })
    $ErrorActionPreference = 'Continue'
    & $plan.python @trainArgs > (Join-Path $runRoot 'overnight.log') 2>&1
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    @{status=if($exitCode -eq 0){'completed'}else{'failed'};exit_code=$exitCode;finished_at=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'worker_status.json') -Encoding UTF8
    if ($exitCode -ne 0) { throw "STES exited with code $exitCode. See overnight.log." }
} finally {
    if ($awake) { [STESAwake]::SetThreadExecutionState([uint32]2147483648) | Out-Null }
}
