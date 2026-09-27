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
$env:PYTHONUTF8 = '1'
$env:FEDSECURELAB_PROJECT_ROOT = $projectRoot
$runStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$flwrHomeBase = Join-Path $env:LOCALAPPDATA 'FedSecureLab\flwr-runs'
$env:FLWR_HOME = Join-Path $flwrHomeBase $runStamp
$env:TEMP = Join-Path $projectRoot '.runtime-tmp'
$env:TMP = $env:TEMP
$env:RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO = '0'
New-Item -ItemType Directory -Force -Path $env:FLWR_HOME, $env:TEMP | Out-Null

# Start the live local dashboard before Flower. It reads the metric fragments
# written by the clients and global evaluator, so the browser updates per round.
$dashboardPort = 8000
$dashboardUrl = "http://127.0.0.1:$dashboardPort/federation-process.html"
$dashboard = Start-Process -FilePath $python `
  -ArgumentList "scripts\dashboard_server.py --port $dashboardPort --reset" `
  -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
Start-Process $dashboardUrl
Write-Host "Live training dashboard: $dashboardUrl"

$federationConfig = "num-supernodes=$env:FLWR_NUM_SUPERNODES client-resources-num-cpus=$env:FLWR_CLIENT_CPUS client-resources-num-gpus=$env:FLWR_CLIENT_GPUS"
# Keep string-valued research hooks in pyproject.toml for now. Passing them
# through PowerShell to a native executable strips TOML quote characters on
# some Windows hosts. The numeric baseline settings below are safe overrides.
$runConfig = "num-server-rounds=$env:NUM_SERVER_ROUNDS local-epochs=$env:LOCAL_EPOCHS batch-size=$env:BATCH_SIZE learning-rate=$env:LEARNING_RATE seed=$env:SEED"

try {
  & $python -m flwr.cli.app run $projectRoot --stream `
    --federation-config $federationConfig `
    --run-config $runConfig
} finally {
  if (-not $dashboard.HasExited) {
    Stop-Process -Id $dashboard.Id -Force
  }
}
