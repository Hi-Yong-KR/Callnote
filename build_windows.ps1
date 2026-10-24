param([switch]$Gpu, [switch]$ModernGpu)

$ErrorActionPreference = "Stop"
if ($Gpu -and $ModernGpu) { throw "Choose only one GPU profile." }

$profile = if ($Gpu) { "pascal" } elseif ($ModernGpu) { "modern" } else { "cpu" }
$environment = if ($profile -eq "pascal") { ".venv-pascal" } elseif ($profile -eq "modern") { ".venv-modern" } else { ".venv" }
$distPath = if ($profile -eq "cpu") { "dist" } else { "dist-$profile" }
$workPath = if ($profile -eq "cpu") { "build" } else { "build-$profile" }

if (-not (Test-Path "$environment\Scripts\python.exe")) {
    py -3 -m venv $environment
}
$python = "$environment\Scripts\python.exe"
& $python -m pip install --upgrade pip
if ($Gpu) {
    # CUDA 12.6 is the last official PyTorch wheel line that supports Pascal GPUs (e.g. Titan Xp).
    & $python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw "Pascal-compatible PyTorch install failed with exit code $LASTEXITCODE" }
} elseif ($ModernGpu) {
    # CUDA 13.2 build for Turing and newer GPUs, including RTX 50-series.
    & $python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu132
    if ($LASTEXITCODE -ne 0) { throw "CUDA-enabled PyTorch install failed with exit code $LASTEXITCODE" }
}
& $python -m pip install -r requirements.txt huggingface-hub

# Bundle FFmpeg in the app; model weights download to the user's local app data on first launch.
& $python prepare_runtime.py

& $python -m PyInstaller --noconfirm --windowed --onedir `
    --name Callnote `
    --distpath $distPath `
    --workpath $workPath `
    --collect-all whisper `
    --collect-all tiktoken `
    --collect-submodules transformers.models.qwen3 `
    --collect-submodules huggingface_hub `
    app.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed with exit code $LASTEXITCODE" }

Copy-Item -Recurse -Force bin "$distPath\Callnote\bin"
Write-Host "Build complete: $distPath\Callnote\Callnote.exe"
Write-Host "Zip the full $distPath\Callnote folder to distribute the standalone app."
