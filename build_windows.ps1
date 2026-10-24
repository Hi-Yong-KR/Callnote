$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3 -m venv .venv
}
$python = ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt huggingface-hub

# Bundle FFmpeg in the app; model weights download to the user's local app data on first launch.
& $python prepare_runtime.py

& $python -m PyInstaller --noconfirm --windowed --onedir `
    --name Callnote `
    --collect-all whisper `
    --collect-all tiktoken `
    --collect-submodules transformers.models.qwen3 `
    --collect-submodules huggingface_hub `
    app.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed with exit code $LASTEXITCODE" }

Copy-Item -Recurse -Force bin dist\Callnote\bin
Write-Host "Build complete: dist\Callnote\Callnote.exe"
Write-Host "Zip the full dist\Callnote folder to distribute the standalone app."
