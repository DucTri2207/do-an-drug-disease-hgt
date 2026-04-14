$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
$checkpointPath = Join-Path $projectRoot "checkpoints\teacher_fusion_hgt_c_dataset.pt"
$resultPath = Join-Path $projectRoot "results\teacher_fusion_hgt_c_dataset.json"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Khong tim thay Python trong venv: $pythonExe"
}

& $pythonExe -m src.main `
    --model fusion_hgt `
    --dataset C-dataset `
    --epochs 100 `
    --learning-rate 0.0005 `
    --weight-decay 0.0001 `
    --hidden-dim 128 `
    --hgt-layers 3 `
    --hgt-heads 4 `
    --similarity-topk 20 `
    --sim-layers 2 `
    --sim-heads 4 `
    --sim-dropout 0.2 `
    --hgt-decoder-mode hybrid `
    --hard-negative-ratio 0.5 `
    --checkpoint-path $checkpointPath `
    --result-json $resultPath

Write-Host ""
Write-Host "Da chay xong AMDGT-lite++. File ket qua:"
Write-Host "  $resultPath"
Write-Host "File checkpoint:"
Write-Host "  $checkpointPath"
