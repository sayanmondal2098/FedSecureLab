# Serves the live federated-learning dashboard and opens it locally.
param([int]$Port = 8000)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = 'py' }

$url = "http://localhost:$Port/federation-process.html"
Start-Process $url
Write-Host "Serving the visualisation at $url"
Write-Host 'Press Ctrl+C to stop the local server.'
Push-Location $projectRoot
try {
    if ($python -eq 'py') { & py scripts/dashboard_server.py --port $Port } else { & $python scripts/dashboard_server.py --port $Port }
} finally {
    Pop-Location
}
