#!/usr/bin/env bash
# Download official LLP (AVVP, ECCV 2020) pre-extracted features (~7.8 GiB)
# Verified direct CLI download (HTTP 206, no quota block as of check).
set -euo pipefail

DEST_DIR="$(cd "$(dirname "$0")/data/llp" && pwd)"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "[LLP] Downloading feats.zip (7.8 GiB) into $DEST_DIR ..."
curl --http1.1 -L -C - --retry 5 --retry-delay 2 -o feats.zip \
  "https://drive.usercontent.google.com/download?id=10CeaI1G9uIyz5dKnd7XTeH9-qOehC7CL&export=download&confirm=t"

echo "[LLP] Extracting feats.zip ..."
unzip -q -o feats.zip
echo "[LLP] Done. Contents:"
ls -lh "$DEST_DIR/feats"
