param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$python = Join-Path $projectRoot "volley_env_311\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python not found: $python"
}

$runRoot = Join-Path $projectRoot "results\overnight_2026-07-31"
$logRoot = Join-Path $runRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$transcript = Join-Path $logRoot "overnight.log"

Start-Transcript -Path $transcript -Append | Out-Null
try {
    $forceArgs = @()
    if ($Force) {
        $forceArgs = @("--force")
    }

    Write-Output "[1/4] 4K ball-coordinate pre-labels"
    & $python "scripts\prepare_ball_prelabels.py" `
        --manifest "data\labeling_intake\2026-07-31_youtube_4k\intake_manifest.json" `
        --output-root (Join-Path $runRoot "youtube_4k_ball_prelabels") `
        --ball-batch-size 8 `
        @forceArgs
    if ($LASTEXITCODE -ne 0) {
        throw "4K pre-label queue failed with exit code $LASTEXITCODE"
    }

    Write-Output "[2/4] Refresh old derived ball coordinates"
    & $python "scripts\prepare_ball_prelabels.py" `
        --manifest "data\labeling_intake\2026-07-31_old_ball_refresh.json" `
        --output-root (Join-Path $runRoot "old_ball_refresh") `
        --ball-batch-size 8 `
        @forceArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Old ball refresh failed with exit code $LASTEXITCODE"
    }

    Write-Output "[3/4] Rebuild normalized trajectory + pose features"
    $featureRoot = Join-Path $projectRoot "data\event_clips_merged_v5\features_ball_refresh_v2"
    & $python -m action_recognition.event_clips.refresh_features `
        --manifest "data\event_clips_merged_v5\manifest.json" `
        --results-root "results\refresh" `
        --ball-results-root (Join-Path $runRoot "old_ball_refresh") `
        --output-features-dir $featureRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Feature refresh failed with exit code $LASTEXITCODE"
    }

    Write-Output "[4/4] Train normalized dual-stream event model"
    & $python -m action_recognition.training.train_event_sequence `
        --manifest "data\event_clips_merged_v5\manifest.json" `
        --features-dir $featureRoot `
        --video-features-dir "data\event_clips_merged_v5\video_features" `
        --model dual `
        --checkpoint-dir "action_recognition\models\checkpoints_fusion_ball_refresh_v2" `
        --epochs 80 `
        --batch-size 32 `
        --patience 15
    if ($LASTEXITCODE -ne 0) {
        throw "Event model training failed with exit code $LASTEXITCODE"
    }

    Write-Output "Overnight queue completed successfully."
}
finally {
    Stop-Transcript | Out-Null
}
