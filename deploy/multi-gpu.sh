#!/usr/bin/env bash
# Run cscan across N GPUs in parallel by sharding the pad range [PAD_START, PAD_END).
# Each GPU gets an equal contiguous slice; output goes to $OUT_PREFIX.$i.json.
#
# Usage:
#   deploy/multi-gpu.sh <targets.txt> <GPUS> <PAD_START> <PAD_END> [extra cscan args...]
#
# Examples (4x RTX 5080):
#   deploy/multi-gpu.sh targets/cek.txt 4 0 4294967296 -w scan/bip39_english.txt \
#       -F 2 -A 18 -C 17 -P 0:0:20 -P 1:0:20
#
#   # only GPUs 2 and 3 (device ids passed via -d):
#   GPUS="2 3" deploy/multi-gpu.sh targets/cek.txt auto 0 4294967296 -F 2 -A 18 -C 17
#
# Notes:
#   * Requires CUDA >= 12.8 (sm_120) to compile for Blackwell (RTX 50xx).
#   * If a shard finds a hit, cscan exits 0 and writes its own output file;
#     this script then stops remaining shards that are still starting.
set -euo pipefail

BIN=${BIN:-gpu/cuda/cscan}
OUT_PREFIX=${OUT_PREFIX:-hits_gpu}
LOG_PREFIX=${LOG_PREFIX:-cscan_gpu}

TARGETS=${1:?usage: multi-gpu.sh <targets.txt> <GPUS|auto> <PAD_START> <PAD_END> [args...]}
GPU_SPEC=${2:?missing GPUS (e.g. 4 or "0 1 2 3")}
PAD_START=${3:?missing PAD_START}
PAD_END=${4:?missing PAD_END}
shift 4

if [[ "$GPU_SPEC" == auto ]]; then
    mapfile -t DEVICES <<< "$(nvidia-smi --query-gpu=index --format=csv,noheader)"
else
    read -r -a DEVICES <<< "$GPU_SPEC"
fi

N=${#DEVICES[@]}
TOTAL=$(( PAD_END - PAD_START ))
(( TOTAL > 0 && TOTAL % N == 0 )) || { echo "pad range ($TOTAL) must split evenly over $N GPUs"; exit 2; }
SLICE=$(( TOTAL / N ))

echo "[multi-gpu] $N workers: devices=[${DEVICES[*]}], pad slice=$SLICE each"
PIDS=()
for i in "${!DEVICES[@]}"; do
    dev=${DEVICES[$i]}
    lo=$(( PAD_START + i * SLICE ))
    hi=$(( lo + SLICE ))
    out="${OUT_PREFIX}.${dev}.json"
    log="${LOG_PREFIX}.${dev}.log"
    echo "[multi-gpu] gpu$dev: pads [$lo,$hi) -> $out (log: $log)"
    "$BIN" -t "$TARGETS" -d "$dev" -p "$lo" -e "$hi" -o "$out" "$@" > "$log" 2>&1 &
    PIDS+=($!)
done

FAIL=0; HIT=""
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        HIT="$HIT ${OUT_PREFIX}.${DEVICES[$i]}.json"
    else
        rc=$?
        (( rc == 1 )) || FAIL=1   # exit 1 = exhausted, no hit (normal)
    fi
done

if [[ -n "$HIT" ]]; then
    echo "[multi-gpu] *** HIT in:$HIT ***"
    exit 0
fi
(( FAIL == 0 )) && echo "[multi-gpu] all shards exhausted, no hit" || {
    echo "[multi-gpu] at least one shard failed; check ${LOG_PREFIX}.*.log"; exit 1; }
