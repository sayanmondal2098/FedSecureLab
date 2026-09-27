param([switch]$DashboardOnly)

# Loads .env and launches Flower with the five-client local simulation profile.
$projectRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $projectRoot '.env'
if (-not (Test-Path -LiteralPath $envPath)) { throw "Missing .env. Copy .env.example to .env first." }

Get-Content -LiteralPath $envPath | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith('#')) {
        $key, $value = $line -split '=', 2
        # Update the current PowerShell process so $env: variables are
        # immediately available while assembling Flower's CLI arguments.
        Set-Item -Path "Env:$($key.Trim())" -Value $value.Trim()
    }
}

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
  throw "Missing virtualenv Python at $python. Create .venv and install requirements first."
}

$env:Path = "$projectRoot\.venv\Scripts;$env:Path"
$env:PYTHONUTF8 = if ($env:PYTHON_UTF8) { $env:PYTHON_UTF8 } else { '1' }
$env:FEDSECURELAB_PROJECT_ROOT = $projectRoot
$runStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$flwrHomeBase = if ($env:FLWR_HOME_BASE) { $env:FLWR_HOME_BASE } else { Join-Path $env:LOCALAPPDATA 'FedSecureLab\flwr-runs' }
$env:FLWR_HOME = Join-Path $flwrHomeBase $runStamp
$runtimeTmpDir = if ($env:RUNTIME_TMP_DIR) { $env:RUNTIME_TMP_DIR } else { '.runtime-tmp' }
$env:TEMP = if ([System.IO.Path]::IsPathRooted($runtimeTmpDir)) { $runtimeTmpDir } else { Join-Path $projectRoot $runtimeTmpDir }
$env:TMP = $env:TEMP
$env:RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO = if ($env:RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO) { $env:RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO } else { '0' }
New-Item -ItemType Directory -Force -Path $env:FLWR_HOME, $env:TEMP | Out-Null

# Start the live local dashboard before Flower. It reads the metric fragments
# written by the clients and global evaluator, so the browser updates per round.
$dashboardPort = if ($env:DASHBOARD_PORT) { [int]$env:DASHBOARD_PORT } else { 8000 }
$dashboardUrl = "http://127.0.0.1:$dashboardPort/federation-process.html"
$dashboard = Start-Process -FilePath $python `
  -ArgumentList "scripts\dashboard_server.py --port $dashboardPort --reset" `
  -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
Start-Process $dashboardUrl
Write-Host "Live training dashboard: $dashboardUrl"

if ($DashboardOnly) {
  Write-Host "Dashboard-only mode enabled. Configure poisoning in the dashboard and click 'Run Clean Then Poisoned'."
  Write-Host "Press Ctrl+C in this terminal when finished."
  Push-Location $projectRoot
  try {
    & $python scripts/dashboard_server.py --port $dashboardPort
  } finally {
    Pop-Location
  }
  exit 0
}

$federationConfig = "num-supernodes=$env:FLWR_NUM_SUPERNODES client-resources-num-cpus=$env:FLWR_CLIENT_CPUS client-resources-num-gpus=$env:FLWR_CLIENT_GPUS"
# Keep string-valued research hooks in pyproject.toml for now. Passing them
# through PowerShell to a native executable strips TOML quote characters on
# some Windows hosts. The numeric baseline settings below are safe overrides.
$poisonMode = if ($env:POISON_MODE) { $env:POISON_MODE } else { 'none' }
$poisonClientIds = if ($env:POISON_CLIENT_IDS) { $env:POISON_CLIENT_IDS } else { '' }
$poisonLabelFlipOffset = if ($env:POISON_LABEL_FLIP_OFFSET) { $env:POISON_LABEL_FLIP_OFFSET } else { '1' }
$poisonNoiseStd = if ($env:POISON_NOISE_STD) { $env:POISON_NOISE_STD } else { '0.15' }
$quotedPoisonMode = '\"{0}\"' -f $poisonMode
$quotedPoisonClientIds = '\"{0}\"' -f $poisonClientIds
$runConfigParts = @(
  "num-server-rounds=$env:NUM_SERVER_ROUNDS",
  "local-epochs=$env:LOCAL_EPOCHS",
  "batch-size=$env:BATCH_SIZE",
  "learning-rate=$env:LEARNING_RATE",
  "seed=$env:SEED",
  "poison-mode=$quotedPoisonMode",
  "poison-label-flip-offset=$poisonLabelFlipOffset",
  "poison-noise-std=$poisonNoiseStd"
)
if ($poisonClientIds.Trim()) {
  $runConfigParts += "poison-client-ids=$quotedPoisonClientIds"
}
$runConfig = $runConfigParts -join ' '

$experimentTag = if ($env:EXPERIMENT_TAG) { $env:EXPERIMENT_TAG.Trim() } else { '' }
$resultsRoot = Join-Path $projectRoot 'results'

function Save-ComparisonSnapshot {
  param(
    [string]$Tag,
    [string]$SourceRoot,
    [string]$DestRoot
  )

  if (-not $Tag) {
    return
  }

  $safeTag = ($Tag -replace '[^A-Za-z0-9._-]', '_')
  $targetDir = Join-Path (Join-Path $DestRoot 'comparison') $safeTag
  if (Test-Path -LiteralPath $targetDir) {
    Remove-Item -LiteralPath $targetDir -Recurse -Force
  }
  New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

  $items = @('client_metrics.csv', 'global_metrics.csv', 'metrics.csv', 'config.json', 'client_round_metrics')
  foreach ($name in $items) {
    $src = Join-Path $SourceRoot $name
    if (Test-Path -LiteralPath $src) {
      Copy-Item -LiteralPath $src -Destination (Join-Path $targetDir $name) -Recurse -Force
    }
  }

  Write-Host "Saved comparison snapshot: results/comparison/$safeTag"
}

$runExitCode = 1

try {
  Write-Host "Running immediate experiment from .env values."
  Write-Host "Tip: use '-DashboardOnly' to run clean+poisoned from dashboard controls instead."
  & $python -m flwr.cli.app run $projectRoot --stream `
    --federation-config $federationConfig `
    --run-config $runConfig
  $runExitCode = $LASTEXITCODE
} finally {
  if (-not $dashboard.HasExited) {
    Stop-Process -Id $dashboard.Id -Force
  }
  if ($runExitCode -eq 0) {
    Save-ComparisonSnapshot -Tag $experimentTag -SourceRoot $resultsRoot -DestRoot $resultsRoot
  }
}

exit $runExitCode
