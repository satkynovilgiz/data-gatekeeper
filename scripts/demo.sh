#!/usr/bin/env bash
# Data Gatekeeper demo: one command, both datasets (Sprint 12).
#
#   scripts/demo.sh            # Bank Marketing + UCI Adult, impact step skipped (~10 min)
#   scripts/demo.sh --full     # also the impact study on both (adds roughly an hour)
#   scripts/demo.sh --local    # do not use Docker even if it is installed
#   scripts/demo.sh --fixture  # no network: two small committed files (tests and CI); uses
#                              # Docker when installed, like the real demo (add --local to avoid it)
#
# Steps: 1. build the Docker image (when Docker is installed; otherwise run locally, said loudly)
#        2. download Bank + Adult; the downloaders stop on a SHA-256 mismatch
#        3. the files the pipelines need: the Bank drift reference, the Adult train/test parts
#        4. gatekeeper run on both (Adult with contracts/adult_decisions.yaml)
#        5. print where the two report.html files are
# A pipeline stopped at a gate (exit 1) is a RESULT, shown in its report; only a crash (exit 2)
# stops the demo.
set -euo pipefail
cd "$(dirname "$0")/.."

FULL=0; LOCAL=0; FIXTURE=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;; --local) LOCAL=1 ;; --fixture) FIXTURE=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (use --full, --local, --fixture)" >&2; exit 2 ;;
  esac
done
PY="${PYTHON:-python}"
start=$(date +%s)
say() { printf '\n== %s\n' "$*"; }

if [ "$LOCAL" -eq 0 ] && command -v docker >/dev/null 2>&1; then
  say "1. building the Docker image data-gatekeeper"
  docker build -q -t data-gatekeeper . >/dev/null
  gatekeeper() { docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" -w /work data-gatekeeper "$@"; }
  where="inside Docker"
else
  [ "$LOCAL" -eq 0 ] && \
    echo "!! Docker is not installed here: running the same commands LOCALLY with $PY" >&2
  gatekeeper() { "$PY" gatekeeper.py "$@"; }
  where="locally"
fi

if [ "$FIXTURE" -eq 1 ]; then
  say "2-3. --fixture: no download, two committed files"
  configs=(tests/fixtures/demo_bank_sample.yaml tests/fixtures/demo_customers.yaml)
else
  say "2. downloading Bank Marketing and UCI Adult (checksums checked)"
  "$PY" scripts/download_bank_marketing.py
  "$PY" scripts/download_adult.py --split
  say "3. the drift reference for Bank (seed 42)"
  "$PY" scripts/make_drift_demo.py
  configs=(pipeline.yaml pipeline_adult.yaml)
fi

skip=(--skip impact); [ "$FULL" -eq 1 ] && skip=()
reports=()
for cfg in "${configs[@]}"; do
  say "4. gatekeeper run --config $cfg ${skip[*]:-} ($where)"
  code=0
  out=$(gatekeeper run --config "$cfg" "${skip[@]}" 2>/dev/null) || code=$?
  echo "$out"
  if [ "$code" -eq 2 ]; then echo "!! $cfg crashed (exit 2); see its run.log" >&2; exit 2; fi
  reports+=("$(echo "$out" | sed -n 's/^Report  -> //p')  (exit $code)")
done

say "5. done $where in $(( $(date +%s) - start )) s. Open the reports:"
printf '   %s\n' "${reports[@]}"
