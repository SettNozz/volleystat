# Create (or refresh) a "CourtSense" shortcut that runs scripts/start_viewer.ps1
# with the CourtSense icon. Defaults to the desktop; -StartMenu adds it to the
# Start menu as well. Re-running overwrites only the shortcut it created.
param(
    [int]$Port = 5050,
    [string]$Name = "CourtSense",
    [switch]$StartMenu
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $projectRoot "scripts\start_viewer.ps1"
$icon = Join-Path $projectRoot "viewer\static\brand\favicon.ico"
foreach ($required in @($launcher, $icon)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing $required" }
}

$folders = @([Environment]::GetFolderPath("Desktop"))
if ($StartMenu) { $folders += [Environment]::GetFolderPath("Programs") }

$shell = New-Object -ComObject WScript.Shell
foreach ($folder in $folders) {
    $path = Join-Path $folder "$Name.lnk"
    # WScript.Shell cannot save to a non-ASCII path (e.g. a localised OneDrive
    # desktop), so the shortcut is written to %TEMP% and moved into place.
    $staged = Join-Path ([IO.Path]::GetTempPath()) "courtsense_$([guid]::NewGuid().ToString('N')).lnk"
    $link = $shell.CreateShortcut($staged)
    $link.TargetPath = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
    $link.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$launcher`" -Port $Port"
    $link.WorkingDirectory = $projectRoot
    $link.IconLocation = "$icon,0"
    $link.Description = "Start the CourtSense volleyball viewer"
    $link.Save()
    Move-Item -LiteralPath $staged -Destination $path -Force
    Write-Host "Shortcut: $path"
}
