#!/usr/bin/env bash
# AI-ASSISTED (non-core) — see CLAUDE.md
#
# Download and extract the official Mip-NeRF 360 dataset into data/mipnerf360/.
# No conversion or downsampling: the release already ships images_2/4/8 and sparse/0.
#
# Usage: bash scripts/download_mipnerf360.sh [--extra]
#   --extra  also fetch the two extra scenes (flowers, treehill)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/mipnerf360"
DL="$ROOT/data/_downloads"
BASE_URL="https://storage.googleapis.com/gresearch/refraw360"

ZIPS=("360_v2.zip")
if [[ "${1:-}" == "--extra" ]]; then
  ZIPS+=("360_extra_scenes.zip")
fi

mkdir -p "$DEST" "$DL"

for z in "${ZIPS[@]}"; do
  if [[ ! -f "$DL/$z" ]]; then
    echo "Downloading $z ..."
    curl -L --fail --retry 3 -C - -o "$DL/$z.part" "$BASE_URL/$z"
    mv "$DL/$z.part" "$DL/$z"
  else
    echo "Already downloaded: $z"
  fi
  echo "Extracting $z -> $DEST"
  unzip -q -n "$DL/$z" -d "$DEST"
done

# Flatten if the archive wrapped scenes in a top-level folder (e.g. 360_v2/).
for wrapper in "$DEST"/360_v2 "$DEST"/360_extra_scenes; do
  if [[ -d "$wrapper" ]]; then
    mv -n "$wrapper"/* "$DEST"/ && rmdir "$wrapper" 2>/dev/null || true
  fi
done

echo
echo "Scenes in $DEST:"
for d in "$DEST"/*/; do
  s="$(basename "$d")"
  flags=""
  [[ -d "$d/images_8" ]] && flags+=" images_8" || flags+=" (no images_8!)"
  [[ -d "$d/sparse/0" ]] && flags+=" sparse/0" || flags+=" (no sparse/0!)"
  echo "  $s:$flags"
done
echo
echo "Zips kept in $DL (delete to reclaim ~12 GB)."
