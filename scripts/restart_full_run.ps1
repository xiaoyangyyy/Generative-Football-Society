# Restart World Cup 2026 full simulation (physics-first micro + LLM + latent world model).
# Stop any existing run_world_cup python process before starting.

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

Write-Host ""
Write-Host "Cleaning prior run artifacts..."
$cleanPaths = @(
    "data\persistence\squad_carryover.json",
    "data\persistence\tournament_checkpoint.json",
    "outputs\full_run_latest.log",
    "outputs\narrative_debug.jsonl"
)
foreach ($p in $cleanPaths) {
    if (Test-Path $p) {
        Remove-Item $p -Force
        Write-Host "  removed $p"
    }
}
foreach ($dir in @("outputs\ball_log", "data\persistence\cognitive_log")) {
    if (Test-Path $dir) {
        Remove-Item "$dir\*" -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  cleared $dir"
    }
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
# Latent world model (GRU) — pass + shot imagination
$env:MATCH_WORLD_MODEL = "1"
$env:MATCH_WM_PLAN = "1"
$env:MATCH_WM_TRANSITION = "gru"
$env:MATCH_WM_IMAGINATION_STEPS = "1"
$env:MATCH_WM_PLANNER_BLEND = "0.30"
$env:MATCH_WM_SHOT_BLEND = "0.40"
$env:MATCH_WM_RECORD = "0"
$env:MATCH_WM_SNAPSHOT_IN_BALL_LOG = "0"
$env:MATCH_DEBUG_NARRATIVE = "1"
$env:MATCH_MICRO_STRICT = "1"

$py = if (Test-Path -LiteralPath ".venv\Scripts\python.exe") {
    ".venv\Scripts\python.exe"
} else {
    "python"
}

Write-Host ""
Write-Host "Ensuring world model checkpoint (collect+train if missing)..."
& $py scripts/ensure_world_model.py --collect-pairs 32 --epochs 45
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: world model ensure failed."
    exit 1
}

Write-Host ""
Write-Host "Preflight: full-stack micro physics (WM + cognitive path)..."
& $py scripts/verify_micro_stack.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: micro preflight failed — fix before full run."
    exit 1
}

Write-Host ""
Write-Host "Flags: MICRO=1 COGNITIVE=1 physics_first=1 WORLD_MODEL=1 DEBUG_NARRATIVE=1 MICRO_STRICT=1 LLM_TIMEOUT=90"
Write-Host "Log: outputs\full_run_latest.log"
Write-Host ""

& $py run_world_cup_2026_full.py
