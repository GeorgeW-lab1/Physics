#!/usr/bin/env python3
"""
EXP-space exploration of exact cut packages using the paper's decoder (solve/verify).

Idea: the package space is 2^50. We sample it by forcing a random subset of categories OUT
(drop them) and a random subset IN (subtract from target), then hand the residual
subset-sum instance to the paper decoder (randomised LLL frames -> vote pooling -> Chase).
Every returned package is re-verified with exact integers. DP is used ONLY to count how many
exact packages exist, so we can report how much of the space the decoder sampling covered.
"""
import json, random, time, sys
from collections import defaultdict
import macro_calc_50 as M

au = M.australia()
names = list(au.spending)
vals = [au.spending[k] for k in names]
n = len(names)
parent = lambda s: s.split(":")[0] if ":" in s else s

def dp_count(target):
    cnt = [0]*(target+1); cnt[0] = 1
    for v in vals:
        for s in range(target, v-1, -1):
            cnt[s] += cnt[s-v]
    return cnt[target]

def sample(target, attempts, seed):
    rng = random.Random(seed)
    found = {}
    stats = {"attempts": 0, "decoder_hits": 0}
    t0 = time.time()
    for k in range(attempts):
        idx = list(range(n)); rng.shuffle(idx)
        n_in = rng.randint(0, 4)
        n_out = rng.randint(10, 26)
        forced_in = idx[:n_in]
        forced_out = idx[n_in:n_in+n_out]
        rest = [i for i in idx[n_in+n_out:]]
        T = target - sum(vals[i] for i in forced_in)
        stats["attempts"] += 1
        if T < 0:
            continue
        if T == 0:
            pkg = tuple(sorted(forced_in))
        else:
            a = [vals[i] for i in rest]
            x, info = M.solve(a, T, max_frames=6, L=14, t=3, q_max=30_000,
                              best_first=True, seed=rng.randrange(10**9), time_limit=8.0)
            if x is None:
                continue
            pkg = tuple(sorted(forced_in + [rest[j] for j, b in enumerate(x) if b]))
        # independent exact re-check on the full 50-item instance
        assert sum(vals[i] for i in pkg) == target
        stats["decoder_hits"] += 1
        found.setdefault(pkg, k)
    stats["seconds"] = round(time.time()-t0, 1)
    return found, stats

def describe(pkg):
    by = defaultdict(int)
    for i in pkg: by[parent(names[i])] += vals[i]
    return dict(by)

if __name__ == "__main__":
    targets = {"2% (A$9.6b)": 9_623, "A$30b": 30_000, "A$50b": 50_000, "A$100b": 100_000, "A$150b": 150_000, "A$200b": 200_000}
    attempts = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out = {}
    protected = {"Health", "Defence", "Education"}
    for label, T in targets.items():
        exact_total = dp_count(T)
        found, st = sample(T, attempts, seed=T)
        pk = list(found)
        rows = []
        for p in pk:
            d = describe(p)
            rows.append({"items": [names[i] for i in p], "n_items": len(p),
                         "by_function_aud_m": d,
                         "spares_health_defence_education": not (set(d) & protected)})
        out[label] = {"target": T, "exact_packages_in_space_DP": exact_total,
                      "distinct_found_by_decoder": len(pk), **st, "packages": rows}
        print(f"\n=== {label}: target {T:,} ===")
        print(f"exact packages that exist (DP count): {exact_total:,}  of 2^50 = {2**50:,}")
        print(f"decoder: {st['decoder_hits']}/{st['attempts']} attempts hit, "
              f"{len(pk)} DISTINCT packages, {st['seconds']}s")
        sp = [r for r in rows if r["spares_health_defence_education"]]
        print(f"packages sparing Health+Defence+Education: {len(sp)}/{len(rows)}")
        for r in sorted(rows, key=lambda r: r["n_items"])[:3]:
            print(f"  [{r['n_items']} items] " + "; ".join(r["items"]))
    json.dump(out, open("exp_space_results.json", "w"), indent=1)
    print("\nsaved exp_space_results.json")
