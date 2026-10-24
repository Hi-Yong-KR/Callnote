param(
    [ValidateSet("cpu", "pascal", "modern")][string]$Profile = "cpu",
    [string]$SevenZipPath = "C:\Program Files\7-Zip\7z.exe"
)

$ErrorActionPreference = "Stop"
$workspace = $PSScriptRoot
$sourceDirectory = if ($Profile -eq "cpu") { "dist" } else { "dist-$Profile" }
$sourcePath = Join-Path $workspace "$sourceDirectory\Callnote"
$outputDirectory = Join-Path $workspace "dist"
if (-not (Test-Path -LiteralPath "$sourcePath\Callnote.exe")) { throw "Build $Profile first." }
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

if ($Profile -eq "cpu") {
    $archivePath = Join-Path $outputDirectory "Callnote-Windows-CPU.zip"
    Compress-Archive -LiteralPath $sourcePath -DestinationPath $archivePath -CompressionLevel Optimal -Force
} else {
    $sevenZip = (Get-Item -LiteralPath $SevenZipPath).FullName
    $sevenZipDirectory = Split-Path -Parent $sevenZip
    $sfxPath = Join-Path $sevenZipDirectory "7z.sfx"
    if (-not (Test-Path -LiteralPath $sfxPath)) { throw "7z.sfx is required next to 7z.exe." }
    Copy-Item -LiteralPath (Join-Path $sevenZipDirectory "License.txt") -Destination "$sourcePath\7-Zip-License.txt" -Force
    $archivePath = Join-Path $outputDirectory "Callnote-$((Get-Culture).TextInfo.ToTitleCase($Profile))-GPU.exe"
    if (Test-Path -LiteralPath $archivePath) { Remove-Item -LiteralPath $archivePath }
    Push-Location (Split-Path -Parent $sourcePath)
    try {
        & $sevenZip a $archivePath Callnote "-sfx$sfxPath" -t7z -mx=9 -m0=LZMA2 -md=64m -ms=on -mmt=8 -bsp0
        if ($LASTEXITCODE -ne 0) { throw "Packaging failed: $LASTEXITCODE" }
        & $sevenZip t $archivePath -bsp0
        if ($LASTEXITCODE -ne 0) { throw "Archive verification failed: $LASTEXITCODE" }
    } finally { Pop-Location }
}

if ((Get-Item -LiteralPath $archivePath).Length -ge 2GB) { throw "The package exceeds GitHub's 2 GiB limit." }
$checksum = Get-FileHash -LiteralPath $archivePath -Algorithm SHA256
"$($checksum.Hash.ToLowerInvariant())  $(Split-Path -Leaf $archivePath)" | Set-Content -LiteralPath "$archivePath.sha256" -Encoding ascii
Write-Host "Package complete: $archivePath"
