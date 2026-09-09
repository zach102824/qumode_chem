#!/usr/bin/env python3
"""Until Asia/Shanghai 11:00 next morning: find min L for chem per bond,
then fill more R points. One job at a time. --no-bs hybrid.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path("/workspace/qumode_chem")
RES = ROOT / "data/results"
STATUS = RES / "nobs_until_11am.status"
STATE = RES / "nobs_until_11am.state.json"
PY = ROOT / ".venv/bin/python"
CHEM = 1.6e-3
SH = timezone(timedelta(hours=8))
DEADLINE = datetime(2026, 9, 9, 11, 0, tzinfo=SH)
# stop launching ~10 min before deadline
LAUNCH_CUTOFF = DEADLINE - timedelta(minutes=10)

# Prefer denser coverage of [0.5, 2.5]; known seeds first
SEED_BONDS = [0.50, 0.88, 1.50, 2.00, 2.50]
EXTRA_BONDS = [0.70, 1.00, 1.20, 1.75, 2.25, 0.60, 1.35, 1.90, 2.35, 0.80, 1.10]

os.chdir(ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def tag(r: float) -> str:
    # 0.5 -> r0050, 0.88 -> r0088, 2.5 -> r0250
    return f"r{r:05.2f}".replace(".", "")


def out_path(r: float, L: int) -> Path:
    return RES / f"h4_hybrid_L{L}_nobs_{tag(r)}.json"


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"results": {}, "notes": []}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2) + "\n")


def ingest_existing(st: dict) -> None:
    """Pull chem flags from any existing result JSONs into state."""
    for p in RES.glob("h4_hybrid_L*_nobs*.json"):
        if "paused" in p.name or "summary" in p.name:
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        for rk, geo in d.get("geometries", {}).items():
            r = float(rk)
            for Lk, rec in geo.get("layers", {}).items():
                L = int(Lk)
                key = f"{r:.2f}:{L}"
                st["results"][key] = {
                    "r": r,
                    "L": L,
                    "chem": bool(rec.get("chemical_acc")),
                    "err": float(rec.get("best_abs_error", 1e9)),
                    "energy": float(rec.get("best_energy", 0.0)),
                    "nit": int(rec["trials"][0]["nit"]) if rec.get("trials") else None,
                    "path": str(p),
                }
    save_state(st)


def known_for(st: dict, r: float) -> dict[int, dict]:
    out = {}
    for k, v in st["results"].items():
        if abs(v["r"] - r) < 1e-9:
            out[int(v["L"])] = v
    return out


def min_chem_L(st: dict, r: float) -> int | None:
    chem_Ls = [L for L, v in known_for(st, r).items() if v["chem"]]
    return min(chem_Ls) if chem_Ls else None


def max_fail_L(st: dict, r: float) -> int | None:
    fail_Ls = [L for L, v in known_for(st, r).items() if not v["chem"]]
    return max(fail_Ls) if fail_Ls else None


def next_L_for_bond(st: dict, r: float) -> int | None:
    """Pick next L to run for this bond toward min chem L (or first probe)."""
    known = known_for(st, r)
    mc = min_chem_L(st, r)
    mf = max_fail_L(st, r)

    if mc is None:
        # no chem yet: if we have failures, go above max fail; else probe L=11
        if mf is None:
            return 11
        nxt = mf + 1
        if nxt in known:
            # find next missing above
            for L in range(mf + 1, 31):
                if L not in known:
                    return L
            return None
        return min(nxt, 30)

    # have chem: find min L — search gap (mf, mc) or go below mc
    lo = mf if mf is not None else max(1, mc - 4)
    # binary-ish: try midpoint of (lo, mc) if gap, else mc-1
    candidates = []
    if mf is not None and mc - mf > 1:
        mid = (mf + mc) // 2
        if mid not in known and mid > mf:
            candidates.append(mid)
        for L in range(mf + 1, mc):
            if L not in known:
                candidates.append(L)
                break
    elif mc > 1 and (mc - 1) not in known:
        candidates.append(mc - 1)
    # also try a bit lower if we only know high chem
    if not candidates and mc > 1:
        for L in range(mc - 1, max(0, mc - 6), -1):
            if L not in known and L >= 1:
                candidates.append(L)
                break

    for L in candidates:
        if L not in known and 1 <= L <= 30:
            return L
    return None  # min L resolved (adjacent fail/chem or L=1 chem)


def bond_resolved(st: dict, r: float) -> bool:
    mc = min_chem_L(st, r)
    if mc is None:
        return False
    if mc == 1:
        return True
    mf = max_fail_L(st, r)
    # resolved if we have a fail at mc-1, or all below checked... require fail at mc-1
    if mf is not None and mf == mc - 1:
        return True
    # if we tried mc-1 and it's chem, not yet min — next_L continues
    known = known_for(st, r)
    if (mc - 1) in known and known[mc - 1]["chem"]:
        return False
    if (mc - 1) in known and not known[mc - 1]["chem"]:
        return True
    return False


def wait_for_l11_chain() -> None:
    """Don't collide with nobs_l11_5pt_chain.sh still running 1.50 / 2.00."""
    status_path = RES / "nobs_l11_5pt_chain.status"
    while True:
        ingest_existing(load_state())
        st = load_state()
        need = []
        for r in (1.50, 2.00):
            if f"{r:.2f}:11" not in st["results"] and not out_path(r, 11).exists():
                need.append(r)
        chain_done = status_path.exists() and ("5-pt chain complete" in status_path.read_text())
        # Once L11 outputs exist (or chain status says complete), leave wait.
        # Do NOT gate on stray/orphaned h4_hybrid PIDs — run_job adopts those.
        if not need and chain_done:
            log("L11 5pt chain settled; starting until-11am planner")
            return
        if not need and not (RES / "nobs_l11_5pt_chain.pid").exists():
            log("L11 5pt chain settled; starting until-11am planner")
            return
        # pid file may linger; treat dead pid as done
        pidf = RES / "nobs_l11_5pt_chain.pid"
        alive = False
        if pidf.exists():
            try:
                cpid = int(pidf.read_text().strip())
                alive = (Path(f"/proc/{cpid}").exists())
            except Exception:
                alive = False
        if not need and not alive:
            log("L11 5pt chain settled; starting until-11am planner")
            return
        log(f"waiting for L11 chain (need L11 for {need}; chain_alive={alive}; chain_done={chain_done})")
        time.sleep(60)




def run_job(r: float, L: int) -> bool:
    out = out_path(r, L)
    if out.exists():
        log(f"skip existing {out.name}")
        return True
    stdout = out.with_suffix(".stdout")
    pidfile = out.with_suffix(".pid")
    # Adopt an already-running job for this out path (e.g. after planner restart)
    if pidfile.exists():
        try:
            old = int(pidfile.read_text().strip())
        except Exception:
            old = None
        if old and Path(f"/proc/{old}").exists():
            try:
                cmd = Path(f"/proc/{old}/cmdline").read_bytes()
            except Exception:
                cmd = b""
            if out.name.encode() in cmd or f"L{L}".encode() in cmd:
                log(f"ADOPT R={r:.2f} L={L} pid={old} -> {out.name}")
                while Path(f"/proc/{old}").exists():
                    time.sleep(30)
                if out.exists():
                    log(f"DONE R={r:.2f} L={L} rc=adopted")
                    return True
                log(f"FAIL R={r:.2f} L={L} adopted pid exited missing {out}")
                return False
    cmd = [
        str(PY),
        "-u",
        "scripts/h4_hybrid.py",
        "--no-bs",
        "--layers-min",
        str(L),
        "--layers-max",
        str(L),
        "--init",
        "random",
        "--bonds",
        str(r),
        "--out",
        str(out.relative_to(ROOT)),
    ]
    log(f"START R={r:.2f} L={L} -> {out.name}")
    with stdout.open("w") as so:
        proc = subprocess.Popen(cmd, stdout=so, stderr=subprocess.STDOUT, cwd=str(ROOT))
    pidfile.write_text(str(proc.pid) + "\n")
    rc = proc.wait()
    if out.exists():
        log(f"DONE R={r:.2f} L={L} rc={rc}")
        return True
    log(f"FAIL R={r:.2f} L={L} rc={rc} missing {out}")
    return False



def pick_next(st: dict) -> tuple[float, int] | None:
    # Phase 1: resolve min-L for seed bonds
    for r in SEED_BONDS:
        if bond_resolved(st, r):
            continue
        L = next_L_for_bond(st, r)
        if L is not None:
            return r, L
    # Phase 2: extras — probe then resolve
    for r in EXTRA_BONDS:
        if bond_resolved(st, r):
            continue
        L = next_L_for_bond(st, r)
        if L is not None:
            return r, L
    return None


def main() -> int:
    STATUS.write_text("")
    log(f"until-11am planner start; deadline={DEADLINE.isoformat()}")
    wait_for_l11_chain()
    while True:
        now = datetime.now(SH)
        if now >= LAUNCH_CUTOFF:
            log(f"launch cutoff reached ({now.isoformat()}); stop starting new jobs")
            break
        st = load_state()
        ingest_existing(st)
        st = load_state()
        nxt = pick_next(st)
        if nxt is None:
            log("all planned bonds resolved; nothing left to run")
            break
        r, L = nxt
        # skip if somehow already there
        if f"{r:.2f}:{L}" in st["results"]:
            continue
        ok = run_job(r, L)
        st = load_state()
        ingest_existing(st)
        if not ok:
            log("job failed; sleep 30s then continue")
            time.sleep(30)
    # summary
    st = load_state()
    ingest_existing(st)
    st = load_state()
    log("=== summary min-L ===")
    bonds = sorted({round(v["r"], 2) for v in st["results"].values()})
    for r in bonds:
        mc = min_chem_L(st, r)
        mf = max_fail_L(st, r)
        log(f"R={r:.2f} min_chem_L={mc} max_fail_L={mf} resolved={bond_resolved(st, r)}")
    log("planner exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
