"""Points-based league runner.

Usage:
  python tools/league.py --agent agent.py --opponents all --seeds 1-3 --deals 300
Opponents: baseline names (always_half, always_pot, calling_station, random), league kinds
(eq_tight ...), or a path to any bot .py. Prints per-match verdict and total points
(win=1, chop=0.5, loss=0). Bots are started fresh per match (like the tournament).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(name):
    if os.path.isfile(name):
        return name
    for p in (f"baseline-bots/baseline_{name}.py", f"league/bots/{name}.py"):
        if os.path.isfile(os.path.join(ROOT, p)):
            return os.path.join(ROOT, p)
    raise SystemExit(f"unknown bot {name}")


def all_opponents():
    names = [os.path.basename(p)[9:-3] for p in glob.glob(os.path.join(ROOT, "baseline-bots", "baseline_*.py"))]
    names += [os.path.basename(p)[:-3] for p in glob.glob(os.path.join(ROOT, "league", "bots", "*.py"))
              if not p.endswith("__init__.py")]
    return sorted(names)


def start(path, port, extra_env=None):
    env = dict(os.environ, PORT=str(port), OMP_NUM_THREADS="1", **(extra_env or {}))
    p = subprocess.Popen([sys.executable, path], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
            return p
        except Exception:
            time.sleep(0.2)
    p.kill()
    raise RuntimeError(f"{path} did not start")


def run_match(agent, opp, seed, deals, port1, port2, tmp):
    out = os.path.join(tmp, f"m_{port1}_{seed}.json")
    pa = start(agent, port1)
    pb = start(opp, port2)
    try:
        r = subprocess.run([sys.executable, "judge_engine.py", "--agent1", f"http://127.0.0.1:{port1}",
                            "--agent2", f"http://127.0.0.1:{port2}", "--seed", str(seed),
                            "--deals", str(deals), "--json", out],
                           cwd=ROOT, capture_output=True, text=True, timeout=1800)
    finally:
        pa.kill()
        pb.kill()
    verdict = next((ln.split("RESULT:")[1].strip() for ln in r.stdout.splitlines() if "RESULT:" in ln), "NONE")
    bb = None
    try:
        d = json.load(open(out))
        bb = d.get("mean_bb_per_deal") or d.get("bb_per_deal")
    except Exception:
        pass
    return verdict, bb, r.stdout


def run_match_json(agent, opp, seed, deals, port1, port2, tmp, agent_env=None, opp_env=None):
    """Like run_match but returns the judge's parsed JSON (None on failure) and takes per-bot env."""
    out = os.path.join(tmp, f"mj_{port1}_{seed}.json")
    pa = start(agent, port1, agent_env)
    try:
        pb = start(opp, port2, opp_env)
    except Exception:
        pa.kill()
        return None
    try:
        subprocess.run([sys.executable, "judge_engine.py", "--agent1", f"http://127.0.0.1:{port1}",
                        "--agent2", f"http://127.0.0.1:{port2}", "--seed", str(seed),
                        "--deals", str(deals), "--json", out],
                       cwd=ROOT, capture_output=True, text=True, timeout=1800)
        return json.load(open(out))
    except Exception:
        return None
    finally:
        pa.kill()
        pb.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="agent.py")
    ap.add_argument("--opponents", default="all")
    ap.add_argument("--seeds", default="1-3")
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--port", type=int, default=5100)
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    opps = all_opponents() if a.opponents == "all" else a.opponents.split(",")
    agent = os.path.abspath(a.agent)
    pts, rows = 0.0, []
    with tempfile.TemporaryDirectory() as tmp:
        for o in opps:
            for s in seeds:
                v, bb, _ = run_match(agent, resolve(o), s, a.deals, a.port, a.port + 1, tmp)
                p = {"AGENT1_WIN": 1.0, "CHOP": 0.5}.get(v, 0.0)
                pts += p
                rows.append((o, s, v, bb))
                print(f"{o:18s} seed {s}: {v:12s} bb={bb}", flush=True)
    print(f"TOTAL POINTS {pts} / {len(rows)}")


if __name__ == "__main__":
    main()
