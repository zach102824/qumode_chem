#!/usr/bin/env bash
# Serial --no-bs L=11 for new 5-pt bonds: 0.50, 1.50, 2.00
set -u
ROOT=/workspace/qumode_chem
cd "$ROOT"
STATUS="$ROOT/data/results/nobs_l11_5pt_chain.status"
PY="$ROOT/.venv/bin/python"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS" >&2; }
wait_pid() {
  local pid=$1
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
}
tag_for() {
  # 0.50 -> r050, 1.50 -> r150, 2.00 -> r200
  python3 -c "print(f'r{float(\"$1\"):05.2f}'.replace('.',''))"
}
run_one() {
  local r=$1
  local tag
  tag=$(tag_for "$r")
  local out="data/results/h4_hybrid_L11_nobs_${tag}.json"
  local stdout="data/results/h4_hybrid_L11_nobs_${tag}.stdout"
  if [[ -f "$out" ]]; then
    log "skip R=$r — $out exists"
    return 0
  fi
  log "starting --no-bs L=11 --bonds $r -> $out"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup "$PY" -u scripts/h4_hybrid.py --no-bs \
      --layers-min 11 --layers-max 11 --init random \
      --bonds "$r" --out "$out" >"$stdout" 2>&1 &
  local pid=$!
  echo "$pid" > "data/results/h4_hybrid_L11_nobs_${tag}.pid"
  log "started R=$r pid=$pid"
  wait_pid "$pid"
  local rc=0
  wait "$pid" || rc=$?
  if [[ -f "$out" ]]; then
    log "finished R=$r ok (rc=$rc)"
  else
    log "finished R=$r MISSING $out (rc=$rc); stop"
    return 1
  fi
}
: > "$STATUS"
log "5-pt chain: --no-bs L=11 bonds 0.50 1.50 2.00 (reuse 0.88 2.50)"
for r in 0.50 1.50 2.00; do
  run_one "$r" || exit 1
done
log "5-pt chain complete"
