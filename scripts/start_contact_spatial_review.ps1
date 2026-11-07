$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot "volley_env_311/Lib/site-packages"
$env:PYTHONUTF8 = "1"
Write-Host "Open http://127.0.0.1:5062 in your browser. Keep this terminal open."
& (Join-Path $projectRoot ".python/cpython-3.11.14-windows-x86_64-none/python.exe") -m action_recognition.annotator.contact_spatial_annotator
