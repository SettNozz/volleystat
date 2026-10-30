$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = (Resolve-Path 'volley_env_311\Lib\site-packages').Path
$env:PYTHONUTF8 = '1'
$env:YOLO_CONFIG_DIR = (Resolve-Path '.runtime\ultralytics').Path
$env:TEMP = (Resolve-Path '.runtime\player_test_tmp').Path
$env:TMP = $env:TEMP
Write-Host 'Resuming saved extraction, then training 9 comparisons. Keep this terminal open.'
Write-Host 'Logs: results/event_team_pose_offline_20261027_v3/'
& '.\.python\cpython-3.11.14-windows-x86_64-none\python.exe' -u '.runtime/event_team_pose_queue_resume.py'
if ($LASTEXITCODE -ne 0) { throw "Experiment exited with code $LASTEXITCODE. See queue_status.json and logs." }
