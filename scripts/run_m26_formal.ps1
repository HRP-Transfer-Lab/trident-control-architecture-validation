param(
    [int]$TotalTasks = 5600,
    [int]$ShardSize = 200,
    [int]$Workers = 2,
    [ValidateSet("thread", "process", "serial")]
    [string]$Executor = "thread",
    [string]$CheckpointDir = "C:\trident-runs\M2.6\checkpoints",
    [string]$OutputDir = "reports\generated\formal_synthetic_recovery_v1",
    [string]$Manifest = "manifests\local\formal_synthetic_recovery_v1_manifest.json",
    [string]$Report = "reports\generated\formal_synthetic_recovery_v1.md",
    [int]$ProgressIntervalSeconds = 30,
    [switch]$SkipAggregate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

New-Item -ItemType Directory -Force -Path $CheckpointDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $Manifest -Parent) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $Report -Parent) | Out-Null

Write-Host "M2.6 formal synthetic recovery"
Write-Host "Repo:          $RepoRoot"
Write-Host "CheckpointDir: $CheckpointDir"
Write-Host "Executor:      $Executor"
Write-Host "Workers:       $Workers"
Write-Host "ShardSize:     $ShardSize"
Write-Host "TotalTasks:    $TotalTasks"

for ($start = 0; $start -lt $TotalTasks; $start += $ShardSize) {
    $count = [Math]::Min($ShardSize, $TotalTasks - $start)
    Write-Host ""
    Write-Host "Starting shard $start count $count"

    python -m trident_validation.synthetic.formal_recovery `
        --checkpoint-only `
        --task-start $start `
        --task-count $count `
        --workers $Workers `
        --executor $Executor `
        --checkpoint-dir $CheckpointDir `
        --progress-interval-seconds $ProgressIntervalSeconds

    if ($LASTEXITCODE -ne 0) {
        throw "Stopped after failed shard beginning $start"
    }
}

if (-not $SkipAggregate) {
    Write-Host ""
    Write-Host "Aggregating checkpoints"
    python -m trident_validation.synthetic.formal_recovery `
        --aggregate-checkpoints `
        --checkpoint-dir $CheckpointDir `
        --output-dir $OutputDir `
        --manifest $Manifest `
        --report $Report

    if ($LASTEXITCODE -ne 0) {
        throw "Aggregation failed"
    }
}
