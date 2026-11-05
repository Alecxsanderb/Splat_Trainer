# AI-ASSISTED (non-core) — see CLAUDE.md
#
# Download and extract the official Mip-NeRF 360 dataset into data\mipnerf360\.
# No conversion or downsampling: the release already ships images_2/4/8 and sparse/0.
# PowerShell equivalent of download_mipnerf360.sh.
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts\download_mipnerf360.ps1 [-Extra]
#   -Extra  also fetch the two extra scenes (flowers, treehill)
#
# Uses curl.exe and tar.exe, both built into Windows 10/11. tar.exe handles zip and is
# much faster than Expand-Archive on multi-GB archives.
param(
    [switch]$Extra
)

$ErrorActionPreference = "Stop"

$Root    = Split-Path -Parent $PSScriptRoot
$Dest    = Join-Path $Root "data\mipnerf360"
$Dl      = Join-Path $Root "data\_downloads"
$BaseUrl = "https://storage.googleapis.com/gresearch/refraw360"

$Zips = @("360_v2.zip")
if ($Extra) { $Zips += "360_extra_scenes.zip" }

New-Item -ItemType Directory -Force -Path $Dest, $Dl | Out-Null

foreach ($z in $Zips) {
    $zipPath = Join-Path $Dl $z
    if (-not (Test-Path $zipPath)) {
        Write-Host "Downloading $z ..."
        & curl.exe -L --fail --retry 3 -C - -o "$zipPath.part" "$BaseUrl/$z"
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $z (curl exit $LASTEXITCODE)" }
        Move-Item -Force "$zipPath.part" $zipPath
    } else {
        Write-Host "Already downloaded: $z"
    }
    Write-Host "Extracting $z -> $Dest"
    # -k: don't overwrite existing files (same as unzip -n in the .sh script)
    & tar.exe -xkf $zipPath -C $Dest
    if ($LASTEXITCODE -ne 0) { throw "Extraction failed: $z (tar exit $LASTEXITCODE)" }
}

# Flatten if the archive wrapped scenes in a top-level folder (e.g. 360_v2\).
foreach ($w in @("360_v2", "360_extra_scenes")) {
    $wrapper = Join-Path $Dest $w
    if (Test-Path $wrapper -PathType Container) {
        Get-ChildItem $wrapper | ForEach-Object {
            $target = Join-Path $Dest $_.Name
            if (-not (Test-Path $target)) { Move-Item $_.FullName $target }
        }
        if (-not (Get-ChildItem $wrapper)) { Remove-Item $wrapper }
    }
}

Write-Host ""
Write-Host "Scenes in ${Dest}:"
Get-ChildItem $Dest -Directory | ForEach-Object {
    $flags = ""
    $flags += if (Test-Path (Join-Path $_.FullName "images_8")) { " images_8" } else { " (no images_8!)" }
    $flags += if (Test-Path (Join-Path $_.FullName "sparse\0")) { " sparse/0" } else { " (no sparse/0!)" }
    Write-Host "  $($_.Name):$flags"
}
Write-Host ""
Write-Host "Zips kept in $Dl (delete to reclaim ~12 GB)."
