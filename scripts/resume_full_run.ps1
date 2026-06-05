# Resume World Cup 2026 from tournament_checkpoint.json (keeps carryover + prior ball logs).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "Stopping prior full-run Python processes (if any)..."
Get-Process python* -ErrorAction SilentlyContinue | Where-Object {
    try {
        $_.Path -and (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue).CommandLine -match "run_world_cup_2026_full"
    } catch { $false }
} | ForEach-Object {
    Write-Host "  kill PID $($_.Id)"
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path "data\persistence\tournament_checkpoint.json")) {
    Write-Host "ERROR: no checkpoint at data\persistence\tournament_checkpoint.json — use restart_full_run.ps1 for a clean start."
    exit 1
}

$env:PYTHONPATH = "."
$env:MATCH_MICRO = "1"
$env:MATCH_COGNITIVE = "1"
$env:MATCH_MICRO_SCORE = "1"
$env:MATCH_SCHEDULED_SHOTS = "0"
$env:MATCH_BALL_LOG = "1"
$env:MATCH_BALL_LOG_MAX = "800"
$env:MATCH_BALL_LOG_TXT = "0"
$env:SAVE_CARRYOVER = "1"
$env:GFS_SEED = "42"
$env:LLM_TIMEOUT_S = "90"
$env:LLM_MAX_RETRIES = "6"
$env:COGNITIVE_LLM_RETRIES = "5"
$env:MATCH_COGNITIVE_COOLDOWN = "150"
$env:MATCH_WORLD_MODEL = "1"
$env:MATCH_WM_PLAN = "1"
$env:MATCH_WM_TRANSITION = "gru"
$env:MATCH_WM_IMAGINATION_STEPS = "3"
$env:MATCH_WM_PLANNER_BLEND = "0.45"
$env:MATCH_WM_SHOT_BLEND = "0.40"
$env:MATCH_WM_RECORD = "0"
$env:MATCH_WM_SNAPSHOT_IN_BALL_LOG = "0"

$py = "C:\Users\K\AppData\Local\Programs\Python\Python313\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "Resuming with --resume (log appends to outputs\full_run_latest.log)..."
& $py run_world_cup_2026_full.py --resume
