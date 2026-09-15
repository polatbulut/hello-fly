#!/usr/bin/env bash
# Clone the fly-brain multi-backend connectome repo (Git LFS) into ~/hello-fly/external.
set -euo pipefail

EXT="$HOME/hello-fly/external"
mkdir -p "$EXT"
cd "$EXT"

git lfs install --skip-repo

if [ -d fly-brain/.git ]; then
    echo "=== fly-brain already cloned, fetching ==="
    cd fly-brain && git fetch --quiet --all && cd ..
else
    echo "=== cloning eonsystemspbc/fly-brain ==="
    git clone --depth 1 https://github.com/eonsystemspbc/fly-brain.git
fi

cd fly-brain
echo
echo "=== pulling LFS objects ==="
git lfs pull || echo "(git lfs pull reported an issue; checking files below)"

echo
echo "=== HEAD ==="
git log --oneline -1

echo
echo "=== data files ==="
ls -lh data/ 2>/dev/null | sed 's/^/  /'

echo
echo "=== are the big files real, or unresolved LFS pointers? ==="
for f in data/2025_Connectivity_783.parquet data/2025_Completeness_783.csv; do
    if [ -f "$f" ]; then
        sz=$(stat -c%s "$f")
        head_bytes=$(head -c 40 "$f" | tr -d '\0')
        if [[ "$head_bytes" == *"version https://git-lfs"* ]]; then
            echo "  $f : LFS POINTER (${sz} B) -- NOT the real file"
        else
            echo "  $f : real file, $(numfmt --to=iec "$sz")"
        fi
    else
        echo "  $f : MISSING"
    fi
done

echo
echo "=== code/ ==="
ls code/*.py 2>/dev/null | sed 's/^/  /'

echo
echo "=== clone DONE ==="
