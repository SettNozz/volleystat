#!/usr/bin/env pwsh

Write-Host "🏐 Volleyball Dataset Curator" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan
Write-Host ""

Write-Host "📊 Checking dataset..." -ForegroundColor Yellow
python check_dataset.py

Write-Host ""
Write-Host "🚀 Starting curator..." -ForegroundColor Green
Write-Host "📱 Open http://localhost:8000 in your browser" -ForegroundColor Blue
Write-Host "🎮 Use keyboard: 'A' to Accept, 'S' to Skip" -ForegroundColor Blue
Write-Host ""
Write-Host "⚠️  DO NOT CLOSE THIS WINDOW! It will stop the server." -ForegroundColor Red
Write-Host ""

python run_curator.py 