"""Self-training loop: hill-climb strategy.P against a mixed, changing field. Stops on plateau.

  python tools/train.py                      # run until plateau (Ctrl+C anytime, resumable)
  python tools/train.py --iterations 2 --deals 200 --workers 4     # quick test

Candidate = champion with 1-3 mutated params. Both play the same seeds against the same pool;
accepted only if it is clearly better (paired) with no errors. Accepted params go to params.json.
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)      # ROOT first so `league` is the package, not tools/league.py
os.environ["STRAT_IGNORE_FILE"] = "1"

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("tools_league", os.path.join(HERE, "league.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
from search_space import BOOLS, NUMERIC  # noqa: E402
import strategy  # noqa: E402
from league.genome import fuzz, random_genome  # noqa: E402

DEFAULTS = {k: v for k, v in strategy.P.items()}
TRAIN = os.path.join(ROOT, "training")
PARAMS = os.path.join(ROOT, "params.json")
LUIS = r"E:\Claude\Coding Projects\opp-luis-jan-poker-bot\agent.py"
GENBOT = os.path.join(ROOT, "league", "bots", "genbot.py")
FIXED = ["exploiter", "tag", "lag", "eq_medium"]
BASELINES = ["always_half", "always_pot", "calling_station", "random"]


def diff_params(p):
    return {k: v for k, v in p.items() if DEFAULTS.get(k) != v}


def mutate(champ, rng):
    c = dict(champ)
    changed = []
    for k in rng.sample(list(NUMERIC) + BOOLS, rng.randint(1, 3)):
        if k in BOOLS:
            c[k] = 0 if c.get(k, 1) else 1
        else:
            lo, hi, step = NUMERIC[k]
            v = c.get(k, DEFAULTS[k]) + rng.choice([-1, 1]) * step * rng.randint(1, 3)
            v = min(hi, max(lo, v))
            c[k] = int(round(v)) if isinstance(DEFAULTS[k], int) and not isinstance(DEFAULTS[k], bool) else round(v, 4)
        changed.append(k)
    return c, changed


def build_pool(it, rng, hof, luis):
    """List of (name, path, env, weight)."""
    pool = []
    if luis:
        pool.append(("luis", LUIS, {}, 2.0))
    for n in FIXED:
        pool.append((n, L.resolve(n), {}, 1.0))
    pool.append((rng.choice(BASELINES), L.resolve(rng.choice(BASELINES)), {}, 0.5))
    gen = random.Random(10_000 + it)
    made = 0
    while made < 3:
        g = random_genome(gen)
        if fuzz(g):
            pool.append((f"gen{it}_{made}", GENBOT, {"BOT_GENES": json.dumps(g)}, 1.0))
            made += 1
    for gid, g in rng.sample(hof, min(3, len(hof))):
        pool.append((f"hof_{gid}", GENBOT, {"BOT_GENES": json.dumps(g)}, 1.0))
    return pool


def play_all(jobs, workers, deals, tmp):
    """jobs: list of (key, agent_env, opp_path, opp_env, seed). Returns {key: result json or None}."""
    def work(args):
        idx, (key, aenv, opath, oenv, seed) = args
        p1 = 5100 + 2 * (idx % workers)
        return key, L.run_match_json(os.path.join(ROOT, "agent.py"), opath, seed, deals,
                                     p1, p1 + 1, tmp, aenv, oenv)
    # one port pair per worker slot: run in waves so two matches never share ports
    out = {}
    for start in range(0, len(jobs), workers):
        wave = list(enumerate(jobs[start:start + workers]))
        with ThreadPoolExecutor(workers) as ex:
            for key, res in ex.map(work, wave):
                out[key] = res
    return out


def bb(res):
    return None if not res else float(res.get("mean_bb_per_deal", 0.0))


def bad(res):
    return (not res) or res.get("errors", 0) or res.get("forfeited")


def evaluate(cand, champ, pool, seed0, deals, workers, tmp):
    """Paired evaluation. Returns (weighted mean diff, se, per-opp diffs, clean, results_by_opp)."""
    jobs = []
    for i, (name, path, env, w) in enumerate(pool):
        for tag, params in (("cand", cand), ("champ", champ)):
            aenv = {"STRAT_PARAMS": json.dumps(diff_params(params)), "STRAT_IGNORE_FILE": "1"}
            jobs.append(((name, tag), aenv, path, env, seed0 + i))
    res = play_all(jobs, workers, deals, tmp)
    diffs, weights, clean, rows = [], [], True, {}
    for name, _, _, w in pool:
        rc, rh = res.get((name, "cand")), res.get((name, "champ"))
        if bad(rc) or bad(rh):
            clean = clean and not bad(rc)
            if bad(rc):
                continue
        if rc and rh:
            d = bb(rc) - bb(rh)
            diffs.append(d)
            weights.append(w)
            rows[name] = (bb(rc), bb(rh))
    if not diffs:
        return 0.0, 1.0, {}, False, rows
    m = sum(d * w for d, w in zip(diffs, weights)) / sum(weights)
    se = (statistics.stdev(diffs) / math.sqrt(len(diffs))) if len(diffs) > 1 else 1.0
    worst = min(diffs)
    return m, se, {"worst": worst}, clean, rows


def holdout(champ, workers, deals, tmp, luis):
    """Champion vs defaults / luis / exploiter on held-out seeds. Returns (sum of bb, detail)."""
    cenv = {"STRAT_PARAMS": json.dumps(diff_params(champ)), "STRAT_IGNORE_FILE": "1"}
    denv = {"STRAT_PARAMS": "{}", "STRAT_IGNORE_FILE": "1"}
    targets = [("defaults", os.path.join(ROOT, "agent.py"), denv), ("exploiter", L.resolve("exploiter"), {})]
    if luis:
        targets.append(("luis", LUIS, {}))
    jobs = [((n, s), cenv, p, e, 9000 + s) for n, p, e in targets for s in range(2)]
    res = play_all(jobs, workers, deals, tmp)
    detail = {}
    for n, _, _ in targets:
        v = [bb(res.get((n, s))) for s in range(2) if res.get((n, s))]
        detail[n] = round(sum(v) / len(v), 3) if v else None
    tot = sum(x for x in detail.values() if x is not None)
    return tot, detail


def load_log():
    path = os.path.join(TRAIN, "log.csv")
    n = 0
    if os.path.isfile(path):
        with open(path, newline="") as f:
            n = max(0, sum(1 for _ in f) - 1)
    return path, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=0, help="0 = until plateau")
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--no-luis", action="store_true")
    ap.add_argument("--holdout-every", type=int, default=25)
    ap.add_argument("--holdout-deals", type=int, default=1000)
    ap.add_argument("--min-gain", type=float, default=0.05)
    a = ap.parse_args()
    luis = (not a.no_luis) and os.path.isfile(LUIS)
    os.makedirs(os.path.join(TRAIN, "hall_of_fame"), exist_ok=True)
    logp, done = load_log()
    champ = dict(DEFAULTS)
    if os.path.isfile(PARAMS):
        champ.update(json.load(open(PARAMS)))
    hof = []
    for fn in sorted(os.listdir(os.path.join(TRAIN, "hall_of_fame")))[-10:]:
        hof.append((fn[:-5], json.load(open(os.path.join(TRAIN, "hall_of_fame", fn)))))
    if not os.path.isfile(logp):
        with open(logp, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "changed", "old", "new", "mean_diff", "se", "accepted", "note"])
    t0 = time.time()
    it = done
    accepts = []          # 1/0 per candidate
    hold_hist, flat = [], 0
    start_hold = None
    with tempfile.TemporaryDirectory() as tmp:
        while True:
            if a.iterations and it >= done + a.iterations:
                break
            if a.max_hours and time.time() - t0 > a.max_hours * 3600:
                print("max-hours reached")
                break
            it += 1
            rng = random.Random(it)
            cand, changed = mutate(champ, rng)
            pool = build_pool(it, rng, hof, luis)
            seed0 = 100 * it
            m, se, extra, clean, rows = evaluate(cand, champ, pool, seed0, a.deals, a.workers, tmp)
            note = ""
            ok = clean and m > 0 and m - 1.0 * se > 0 and extra.get("worst", 0) > -1.0
            if not ok and clean and m > 0.02 and extra.get("worst", 0) > -1.0:
                m2, se2, extra2, clean2, rows2 = evaluate(cand, champ, pool, seed0 + 50, a.deals, a.workers, tmp)
                m, se = (m + m2) / 2, max(se, se2) / math.sqrt(2) * 1.2
                ok = clean2 and m > 0 and m - 1.28 * se > 0 and extra2.get("worst", 0) > -1.0
                note = "2 batches"
            if not clean:
                note = (note + " errors").strip()
            old = {k: champ.get(k) for k in changed}
            new = {k: cand.get(k) for k in changed}
            accepts.append(1 if ok else 0)
            with open(logp, "a", newline="") as f:
                csv.writer(f).writerow([it, ";".join(changed), json.dumps(old), json.dumps(new),
                                        round(m, 3), round(se, 3), int(ok), note])
            print(f"[{it}] {changed} diff={m:+.3f} se={se:.3f} {'ACCEPT' if ok else 'reject'} {note}", flush=True)
            if ok:
                champ = cand
                with open(PARAMS, "w") as f:
                    json.dump(diff_params(champ), f, indent=1, sort_keys=True)
            # hall of fame: generated opponents the champion could not clearly beat
            for name, (c_bb, h_bb) in rows.items():
                if name.startswith("gen") and h_bb <= 0.1:
                    path = os.path.join(TRAIN, "hall_of_fame", f"{name}.json")
                    for p_ in pool:
                        if p_[0] == name:
                            g = json.loads(p_[2]["BOT_GENES"])
                            json.dump(g, open(path, "w"))
                            hof.append((name, g))
                            hof = hof[-10:]
            if it % a.holdout_every == 0:
                tot, detail = holdout(champ, a.workers, a.holdout_deals, tmp, luis)
                if start_hold is None:
                    start_hold = (tot, detail)
                prev = hold_hist[-1] if hold_hist else None
                hold_hist.append(tot)
                print(f"HOLDOUT it={it}: total={tot:.3f} {detail}", flush=True)
                if prev is not None:
                    flat = flat + 1 if tot - prev < a.min_gain else 0
                if len(hold_hist) >= 4 and flat >= 3 and sum(accepts[-40:]) < 2:
                    print("PLATEAU: stopping")
                    break
    # final report
    with tempfile.TemporaryDirectory() as tmp2:
        tot, detail = holdout(champ, a.workers, a.holdout_deals, tmp2, luis)
    with open(os.path.join(TRAIN, "final_report.md"), "w") as f:
        f.write(f"# Training report\n\niterations: {it}\naccepted: {sum(accepts)}\nfinal holdout: {detail}\n"
                f"holdout at first check: {start_hold}\nparams: {json.dumps(diff_params(champ))}\n")
    print("final holdout:", detail, "| params.json:", diff_params(champ))


if __name__ == "__main__":
    main()
