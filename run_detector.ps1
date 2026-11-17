param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
& $Python (Join-Path $PSScriptRoot 'detector_app.py')
exit $LASTEXITCODE
