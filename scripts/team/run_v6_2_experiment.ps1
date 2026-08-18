[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(42, 52, 62)]
    [int]$Seed,

    [string]$PcId = $env:COMPUTERNAME,

    [switch]$DryRun,

    [switch]$AllowDirty
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Promptune V6.2 contrastive boundary experiment
# Fixed conditions:
#   Model      : beomi/KcELECTRA-base
#   Epochs     : 3
#   Batch size : 4
#   LR         : 2e-5
#   Max length : 256
#   Primary threshold tuner : V1 (per-label F1 maximization)
#   Threshold search        : 0.01 ~ 0.99, step 0.01
# ---------------------------------------------------------------------------

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
Set-Location $RepoRoot

$Model = "beomi/KcELECTRA-base"
$Epochs = 3
$BatchSize = 4
$LearningRate = "2e-5"
$MaxLength = 256

$TrainRel = "data\v6_2\training_ready\Promptune_v6_2_train_2060_training_v1.csv"
$ValidRel = "data\v6\training_ready\Promptune_v6_valid_400_training_v1.csv"

$TrainPath = Join-Path $RepoRoot $TrainRel
$ValidPath = Join-Path $RepoRoot $ValidRel
$DatasetCheck = Join-Path $RepoRoot "scripts\v6_2\01_validate\check_v6_2_dataset_load.py"

if ([string]::IsNullOrWhiteSpace($PcId)) {
    $PcId = "unknown-pc"
}

$SafePcId = ($PcId.ToLower() -replace '[^a-z0-9_-]', '-').Trim('-')
if ([string]::IsNullOrWhiteSpace($SafePcId)) {
    $SafePcId = "unknown-pc"
}

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunId = "v6-2-e3-seed$Seed-$SafePcId-$Timestamp"
$OutputDir = Join-Path $RepoRoot "artifacts\$RunId"
$LogPath = Join-Path $OutputDir "run.log"
$SubmissionDir = Join-Path $OutputDir "submission"
$SubmissionZip = Join-Path $OutputDir "$RunId-submission.zip"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host $Title
    Write-Host ("=" * 72)
}

function Assert-FileExists {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label not found: $Path"
    }
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    Write-Section $Name
    Write-Host "python $($Arguments -join ' ')"

    $StepStart = Get-Date

    # Windows PowerShell 5.1 can wrap native stderr as NativeCommandError.
    # Hugging Face/Transformers normally writes progress information to stderr,
    # so do not treat stderr itself as a training failure.
    # Stream stdout/stderr to the console + run.log, then judge success only
    # from the native process exit code.
    $PreviousErrorActionPreference = $ErrorActionPreference

    try {
        $ErrorActionPreference = "Continue"

        & python @Arguments 2>&1 |
            Tee-Object -FilePath $LogPath -Append |
            Out-Host

        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    if ($ExitCode -ne 0) {
        throw "$Name failed with exit code $ExitCode"
    }

    # Out-Host above prevents command logs from becoming part of this
    # function's return value. Return only elapsed seconds.
    return [double]((Get-Date) - $StepStart).TotalSeconds
}

Write-Section "Promptune V6.2 experiment preflight"

Write-Host "Repository : $RepoRoot"
Write-Host "PC ID      : $SafePcId"
Write-Host "Seed       : $Seed"
Write-Host "Model      : $Model"
Write-Host "Epochs     : $Epochs"
Write-Host "Batch size : $BatchSize"
Write-Host "LR         : $LearningRate"
Write-Host "Max length : $MaxLength"

Assert-FileExists -Path $TrainPath -Label "V6.2 train file"
Assert-FileExists -Path $ValidPath -Label "V6 valid file"
Assert-FileExists -Path $DatasetCheck -Label "V6 dataset checker"

# Git metadata ---------------------------------------------------------------
$GitCommitFull = (& git rev-parse HEAD 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read Git commit."
}

$GitCommitShort = (& git rev-parse --short HEAD 2>&1 | Out-String).Trim()
$GitBranch = (& git branch --show-current 2>&1 | Out-String).Trim()
if ([string]::IsNullOrWhiteSpace($GitBranch)) {
    $GitBranch = "DETACHED"
}
$GitStatusLines = @(& git status --short)

Write-Host "Git branch  : $GitBranch"
Write-Host "Git commit  : $GitCommitShort"

if ($GitStatusLines.Count -gt 0) {
    Write-Host ""
    Write-Host "Git working tree has changes:"
    $GitStatusLines | ForEach-Object { Write-Host "  $_" }

    if (-not $AllowDirty) {
        throw "Official runs require a clean Git working tree. Commit/stash changes first, or use -AllowDirty only for local validation."
    }
}

# Environment metadata -------------------------------------------------------
$PythonVersion = (& python --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Python is not available."
}

$EnvJson = @'
import json
import sys
import torch
import numpy
import pandas
import sklearn
import transformers
import accelerate

data = {
    "python_runtime": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else None,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "sklearn": sklearn.__version__,
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
}
print(json.dumps(data, ensure_ascii=False))
'@ | python -

if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect Python/ML environment."
}

$EnvInfo = $EnvJson | ConvertFrom-Json

Write-Host "Python      : $($EnvInfo.python_runtime)"
Write-Host "Torch       : $($EnvInfo.torch)"
Write-Host "CUDA        : $($EnvInfo.cuda_available)"
Write-Host "CUDA runtime: $($EnvInfo.torch_cuda)"
Write-Host "GPU         : $($EnvInfo.gpu)"
Write-Host "GPU memory  : $($EnvInfo.gpu_memory_gb) GB"

if (-not $EnvInfo.cuda_available) {
    throw "CUDA is not available. Stop before official GPU training."
}

# Data hashes ----------------------------------------------------------------
$TrainHash = (Get-FileHash -LiteralPath $TrainPath -Algorithm SHA256).Hash
$ValidHash = (Get-FileHash -LiteralPath $ValidPath -Algorithm SHA256).Hash

Write-Host "Train SHA256: $TrainHash"
Write-Host "Valid SHA256: $ValidHash"

if ($DryRun) {
    Write-Section "DRY RUN PASS"
    Write-Host "No model training was executed."
    Write-Host ""
    Write-Host "Actual run command:"
    Write-Host ".\scripts\team\run_v6_2_experiment.ps1 -Seed $Seed -PcId $SafePcId"
    exit 0
}

# Output directory is created only for a real run ----------------------------
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
New-Item -ItemType Directory -Path $SubmissionDir -Force | Out-Null

$RunStart = Get-Date

@"
Promptune V6.2 contrastive boundary run
Run ID      : $RunId
PC ID       : $SafePcId
Seed        : $Seed
Git branch  : $GitBranch
Git commit  : $GitCommitFull
Model       : $Model
Epochs      : $Epochs
Batch size  : $BatchSize
LR          : $LearningRate
Max length  : $MaxLength
Train       : $TrainRel
Valid       : $ValidRel
Train SHA256: $TrainHash
Valid SHA256: $ValidHash
Started at  : $($RunStart.ToString("o"))
"@ | Set-Content -LiteralPath $LogPath -Encoding utf8

# Save initial environment snapshots -----------------------------------------
$GitStatusLines | Set-Content -LiteralPath (Join-Path $OutputDir "git-status.txt") -Encoding utf8

python -m pip freeze |
    Set-Content -LiteralPath (Join-Path $OutputDir "pip-freeze.txt") -Encoding utf8

try {
    nvidia-smi |
        Set-Content -LiteralPath (Join-Path $OutputDir "nvidia-smi.txt") -Encoding utf8
}
catch {
    "nvidia-smi unavailable: $($_.Exception.Message)" |
        Set-Content -LiteralPath (Join-Path $OutputDir "nvidia-smi.txt") -Encoding utf8
}

# 1) Dataset validation -------------------------------------------------------
$DatasetCheckSeconds = Invoke-PythonStep `
    -Name "1/7 V6.2 dataset load check" `
    -Arguments @(
        $DatasetCheck
    )

# 2) Training -----------------------------------------------------------------
$TrainingSeconds = Invoke-PythonStep `
    -Name "2/7 Train V6.2 KcELECTRA" `
    -Arguments @(
        "-m", "ml.train",
        "--model", $Model,
        "--train", $TrainPath,
        "--valid", $ValidPath,
        "--output", $OutputDir,
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--lr", "$LearningRate",
        "--max-length", "$MaxLength",
        "--seed", "$Seed"
    )

# 3) Default threshold evaluation + probability/label export -----------------
$DefaultEvalPath = Join-Path $OutputDir "valid-default05.json"
$ProbabilitiesPath = Join-Path $OutputDir "valid-probabilities.npy"
$LabelsPath = Join-Path $OutputDir "valid-labels.npy"

$DefaultEvalSeconds = Invoke-PythonStep `
    -Name "3/7 Evaluate valid with default threshold" `
    -Arguments @(
        "-m", "ml.evaluate",
        "--model", $OutputDir,
        "--test", $ValidPath,
        "--max-length", "$MaxLength",
        "--output", $DefaultEvalPath,
        "--save-probabilities", $ProbabilitiesPath,
        "--save-labels", $LabelsPath
    )

# 4) Primary threshold tuner: V1 ---------------------------------------------
$ThresholdV1Path = Join-Path $OutputDir "thresholds-v1.json"

$TuneV1Seconds = Invoke-PythonStep `
    -Name "4/7 Tune thresholds V1 (primary)" `
    -Arguments @(
        "-m", "ml.tune_thresholds",
        "--probabilities", $ProbabilitiesPath,
        "--labels", $LabelsPath,
        "--output", $ThresholdV1Path
    )

$TunedV1Path = Join-Path $OutputDir "valid-tuned-v1.json"

$EvalV1Seconds = Invoke-PythonStep `
    -Name "5/7 Evaluate valid with V1 thresholds" `
    -Arguments @(
        "-m", "ml.evaluate",
        "--model", $OutputDir,
        "--test", $ValidPath,
        "--thresholds", $ThresholdV1Path,
        "--max-length", "$MaxLength",
        "--output", $TunedV1Path
    )

# 5) Secondary threshold tuner: V2 -------------------------------------------
$ThresholdV2Path = Join-Path $OutputDir "thresholds-v2.json"
$ThresholdV2ReportPath = Join-Path $OutputDir "thresholds-v2_report.json"

$TuneV2Seconds = Invoke-PythonStep `
    -Name "6/7 Tune thresholds V2 (secondary)" `
    -Arguments @(
        "-m", "ml.tune_thresholds_v2",
        "--probabilities", $ProbabilitiesPath,
        "--labels", $LabelsPath,
        "--output", $ThresholdV2Path
    )

$TunedV2Path = Join-Path $OutputDir "valid-tuned-v2.json"

$EvalV2Seconds = Invoke-PythonStep `
    -Name "7/7 Evaluate valid with V2 thresholds" `
    -Arguments @(
        "-m", "ml.evaluate",
        "--model", $OutputDir,
        "--test", $ValidPath,
        "--thresholds", $ThresholdV2Path,
        "--max-length", "$MaxLength",
        "--output", $TunedV2Path
    )

$RunEnd = Get-Date
$TotalSeconds = ($RunEnd - $RunStart).TotalSeconds

# Load final metrics ----------------------------------------------------------
$DefaultMetrics = Get-Content -LiteralPath $DefaultEvalPath -Raw -Encoding utf8 | ConvertFrom-Json
$V1Metrics = Get-Content -LiteralPath $TunedV1Path -Raw -Encoding utf8 | ConvertFrom-Json
$V2Metrics = Get-Content -LiteralPath $TunedV2Path -Raw -Encoding utf8 | ConvertFrom-Json
$V1Thresholds = Get-Content -LiteralPath $ThresholdV1Path -Raw -Encoding utf8 | ConvertFrom-Json
$V2Thresholds = Get-Content -LiteralPath $ThresholdV2Path -Raw -Encoding utf8 | ConvertFrom-Json

$Metadata = [ordered]@{
    run_id = $RunId
    pc_id = $SafePcId
    seed = $Seed

    started_at = $RunStart.ToString("o")
    finished_at = $RunEnd.ToString("o")

    git = [ordered]@{
        branch = $GitBranch
        commit_short = $GitCommitShort
        commit_full = $GitCommitFull
        working_tree_clean = ($GitStatusLines.Count -eq 0)
    }

    model = $Model

    training = [ordered]@{
        epochs = $Epochs
        batch_size = $BatchSize
        learning_rate = $LearningRate
        max_length = $MaxLength
    }

    threshold_policy = [ordered]@{
        primary = "v1_per_label_f1"
        secondary = "v2_min_precision_f2"
        search_start = 0.01
        search_end = 0.99
        search_step = 0.01
        v2_min_precision = 0.55
    }

    data = [ordered]@{
        train = $TrainRel
        valid = $ValidRel
        train_sha256 = $TrainHash
        valid_sha256 = $ValidHash
    }

    environment = [ordered]@{
        python = $EnvInfo.python_runtime
        torch = $EnvInfo.torch
        torch_cuda = $EnvInfo.torch_cuda
        cuda_available = $EnvInfo.cuda_available
        gpu = $EnvInfo.gpu
        gpu_memory_gb = $EnvInfo.gpu_memory_gb
        numpy = $EnvInfo.numpy
        pandas = $EnvInfo.pandas
        sklearn = $EnvInfo.sklearn
        transformers = $EnvInfo.transformers
        accelerate = $EnvInfo.accelerate
    }

    elapsed_seconds = [ordered]@{
        dataset_check = [math]::Round($DatasetCheckSeconds, 2)
        training = [math]::Round($TrainingSeconds, 2)
        default_eval = [math]::Round($DefaultEvalSeconds, 2)
        tune_v1 = [math]::Round($TuneV1Seconds, 2)
        eval_v1 = [math]::Round($EvalV1Seconds, 2)
        tune_v2 = [math]::Round($TuneV2Seconds, 2)
        eval_v2 = [math]::Round($EvalV2Seconds, 2)
        total = [math]::Round($TotalSeconds, 2)
    }

    metrics = [ordered]@{
        default_05 = $DefaultMetrics
        tuned_v1 = $V1Metrics
        tuned_v2 = $V2Metrics
    }

    thresholds = [ordered]@{
        v1 = $V1Thresholds
        v2 = $V2Thresholds
    }
}

$MetadataPath = Join-Path $OutputDir "run-metadata.json"
$Metadata |
    ConvertTo-Json -Depth 12 |
    Set-Content -LiteralPath $MetadataPath -Encoding utf8

# Build small submission package. Do NOT include model weights/checkpoints. ---
$SubmissionFiles = @(
    $MetadataPath,
    $DefaultEvalPath,
    $TunedV1Path,
    $ThresholdV1Path,
    $TunedV2Path,
    $ThresholdV2Path,
    $ThresholdV2ReportPath,
    $ProbabilitiesPath,
    $LabelsPath,
    (Join-Path $OutputDir "pip-freeze.txt"),
    (Join-Path $OutputDir "git-status.txt"),
    (Join-Path $OutputDir "nvidia-smi.txt"),
    $LogPath
)

foreach ($File in $SubmissionFiles) {
    if (Test-Path -LiteralPath $File -PathType Leaf) {
        Copy-Item -LiteralPath $File -Destination $SubmissionDir -Force
    }
}

Compress-Archive `
    -Path (Join-Path $SubmissionDir "*") `
    -DestinationPath $SubmissionZip `
    -Force

# Final summary ---------------------------------------------------------------
Write-Section "V6.2 EXPERIMENT COMPLETE"

Write-Host "Run ID       : $RunId"
Write-Host "PC ID        : $SafePcId"
Write-Host "Seed         : $Seed"
Write-Host "Git commit   : $GitCommitShort"
Write-Host "Training sec : $([math]::Round($TrainingSeconds, 2))"
Write-Host "Total sec    : $([math]::Round($TotalSeconds, 2))"

Write-Host ""
Write-Host "Primary result (V1)"
Write-Host "  Macro F1   : $($V1Metrics.macro_f1)"
Write-Host "  Micro F1   : $($V1Metrics.micro_f1)"
Write-Host "  Exact Match: $($V1Metrics.exact_match)"

Write-Host ""
Write-Host "Secondary result (V2)"
Write-Host "  Macro F1   : $($V2Metrics.macro_f1)"
Write-Host "  Micro F1   : $($V2Metrics.micro_f1)"
Write-Host "  Exact Match: $($V2Metrics.exact_match)"

Write-Host ""
Write-Host "Full artifact : $OutputDir"
Write-Host "Share this ZIP: $SubmissionZip"
Write-Host ""
Write-Host "IMPORTANT: Do not evaluate the frozen Benchmark 210 at this stage."
