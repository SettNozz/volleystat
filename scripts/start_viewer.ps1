# Start the CourtSense viewer and open the result library in the browser.
# If a viewer already answers on the port, only the browser is opened.
param(
    [int]$Port = 5050,
    [string]$Page = "library"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$url = "http://localhost:$Port/$Page"
$Host.UI.RawUI.WindowTitle = "CourtSense viewer :$Port"

function Test-Viewer {
    # A TCP probe: Invoke-WebRequest in Windows PowerShell spends seconds on
    # proxy detection, and /api/library scans every result directory.
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        return $client.ConnectAsync("127.0.0.1", $Port).Wait(500) -and $client.Connected
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

if (Test-Viewer) {
    Write-Host "Viewer already running on port $Port."
    Start-Process $url
    exit 0
}

# Prefer the venv interpreter; fall back to the standalone Python with the
# venv's packages when the venv launcher is broken.
$venvPython = Join-Path $projectRoot "volley_env_311/Scripts/python.exe"
$standalonePython = Join-Path $projectRoot ".python/cpython-3.11.14-windows-x86_64-none/python.exe"
$env:PYTHONUTF8 = "1"
$python = $venvPython
if (-not (Test-Path -LiteralPath $venvPython)) {
    $python = $standalonePython
    $env:PYTHONPATH = Join-Path $projectRoot "volley_env_311/Lib/site-packages"
}

$viewer = Start-Process -FilePath $python -ArgumentList @("viewer.py", "--port", "$Port") `
    -WorkingDirectory $projectRoot -NoNewWindow -PassThru
Write-Host "Starting viewer on port $Port..."
for ($i = 0; $i -lt 120; $i++) {
    if ($viewer.HasExited) {
        Write-Host "Viewer exited with code $($viewer.ExitCode)." -ForegroundColor Red
        Read-Host "Press Enter to close"
        exit 1
    }
    if (Test-Viewer) { break }
    Start-Sleep -Milliseconds 500
}
Start-Process $url
Write-Host "Viewer: $url"
Write-Host "Close this window to stop the viewer."
$viewer.WaitForExit()
