param([string]$AudioDir = 'data/raw/imda', [int]$Limit = 0)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
$audioRoot = $AudioDir
$arguments = @('-m', 'switchsafe.cli', '--audio-dir', $audioRoot, '--model', 'tiny.en', '--output', 'data/derived/imda-682.json')
if ($Limit -gt 0) { $arguments += @('--limit', "$Limit") }
& (Join-Path $PSScriptRoot '.venv/Scripts/python.exe') @arguments
exit $LASTEXITCODE
