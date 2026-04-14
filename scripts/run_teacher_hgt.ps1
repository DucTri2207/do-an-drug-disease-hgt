$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
$checkpointPath = Join-Path $projectRoot "checkpoints\teacher_hgt_c_dataset.pt"
$resultPath = Join-Path $projectRoot "results\teacher_hgt_c_dataset.json"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Khong tim thay Python trong venv: $pythonExe"
}

& $pythonExe -m src.main `
    --model hgt `
    --dataset C-dataset `
    --epochs 1000 `
    --learning-rate 0.0005 `
    --weight-decay 0.0001 `
    --hidden-dim 128 `
    --hgt-layers 3 `
    --checkpoint-path $checkpointPath `
    --result-json $resultPath

Write-Host ""
Write-Host "Da chay xong. File ket qua:"
Write-Host "  $resultPath"
Write-Host "File checkpoint:"
Write-Host "  $checkpointPath"
