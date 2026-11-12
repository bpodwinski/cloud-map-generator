#!/bin/bash
# Makes generated 2048 x 1024 maps (realistic parameters) for training the refiner on what it will be given:
#   bash training/make_generated.sh <cloud-map-generator exe> <first seed> <count> [raw]
# Matched to Earth (--match-earth) into training/data/generated/, or with "raw" as the generator makes them into training/data/generated_raw/.
# Each map's 8-bit PNG is kept, the rest of the run's files are removed.
EXE=${1:-rust/target/release/cloud-map-generator.exe}; FIRST=${2:-100}; COUNT=${3:-200}; MODE=${4:-matched}
if [ "$MODE" = raw ]; then OUT=training/data/generated_raw; FLAG=""; TAG=""; else OUT=training/data/generated; FLAG="--match-earth"; TAG="_m"; fi
mkdir -p "$OUT"; TMP=$(mktemp -d)
for ((s=FIRST; s<FIRST+COUNT; s++)); do
  [ -f "$OUT/s$s.png" ] && continue
  "$EXE" gen --seed $s --width 2048 --compute cpu $FLAG --out "$TMP/r" >/dev/null 2>&1
  f=$(ls "$TMP"/r/coverage_earth_s${s}${TAG}_2048x1024.png 2>/dev/null)
  [ -n "$f" ] && mv "$f" "$OUT/s$s.png"
  rm -rf "$TMP/r"
done
