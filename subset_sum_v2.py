#!/usr/bin/env python3
"""
Subset-sum solver v2: scales the paper's "pool noisy frames -> list-decode -> exact verifier"
pipeline to much larger instances.

What changed versus v1 (the greedy version, which failed at n=50):

 LOW-DENSITY regime  (n / log2(max a) < ~0.94)   -> LATTICE FRAMES
   Each randomised LLL reduction of the Lagarias-Odlyzko / CJLOSS embedding is one "frame".
   A reduced-basis row with last coordinate 0 is a readout of 2x-1 in {-1,0,+1}^n:
        +1 / -1  -> a valid bit,     0 -> an ERASURE (the paper's erasure channel, Sec. 2.3)
   Rows are sign-aligned and pooled (the sufficient statistic D_i), then decoded with the
   paper's Chase-type search over the least-reliable positions, done exactly by
   meet-in-the-middle because the sum check exposes a residual.

 HIGH-DENSITY regime (many solutions)             -> POSTERIOR-RESTRICTED MITM
   Fix all but m uncertain items by sampling from the pooled posterior, then solve the
   remaining m-item problem exactly with a vectorised meet-in-the-middle (2^(m/2) work).
   With many solutions, a random restriction keeps a solution alive with decent probability.

 The verifier is always exact integer arithmetic. The solver can fail (return None) but it
 can never return a wrong answer. It does NOT break NP-hardness: the hard middle band
 (density ~ 1, large n) remains out of reach; see the proof harness for measured limits.
"""
import json
import math
import random
import sys
import time
from typing import List, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------------
# Exact verifier (the "commitment")
# ----------------------------------------------------------------------------
def verify(a: Sequence[int], T: int, x: Sequence[int]) -> bool:
    return len(a) == len(x) and all(v in (0, 1) for v in x) and \
        sum(ai for ai, xi in zip(a, x) if xi) == T


# ----------------------------------------------------------------------------
# Integral LLL (Cohen, Alg. 2.6.7). Exact big-integer arithmetic, delta = 0.99.
# ----------------------------------------------------------------------------
def lll(B: List[List[int]], dn: int = 99, dd: int = 100) -> List[List[int]]:
    B = [row[:] for row in B]
    n = len(B)
    dot = lambda u, v: sum(p * q for p, q in zip(u, v))
    D = [1] + [0] * n
    lam = [[0] * n for _ in range(n)]
    D[1] = dot(B[0], B[0])
    k, kmax = 1, 0

    def redi(k, l):
        if 2 * abs(lam[k][l]) > D[l + 1]:
            q = (2 * lam[k][l] + D[l + 1]) // (2 * D[l + 1])
            B[k] = [x - q * y for x, y in zip(B[k], B[l])]
            lam[k][l] -= q * D[l + 1]
            for i in range(l):
                lam[k][i] -= q * lam[l][i]

    def swapi(k):
        B[k], B[k - 1] = B[k - 1], B[k]
        for j in range(k - 1):
            lam[k][j], lam[k - 1][j] = lam[k - 1][j], lam[k][j]
        l = lam[k][k - 1]
        Bn = (D[k - 1] * D[k + 1] + l * l) // D[k]
        for i in range(k + 1, kmax + 1):
            t = lam[i][k]
            lam[i][k] = (D[k + 1] * lam[i][k - 1] - l * t) // D[k]
            lam[i][k - 1] = (Bn * t + l * lam[i][k]) // D[k + 1]
        D[k] = Bn

    while k < n:
        if k > kmax:
            kmax = k
            for j in range(k + 1):
                u = dot(B[k], B[j])
                for i in range(j):
                    u = (D[i + 1] * u - lam[k][i] * lam[j][i]) // D[i]
                if j < k:
                    lam[k][j] = u
                else:
                    D[k + 1] = u
        redi(k, k - 1)
        if dd * D[k + 1] * D[k - 1] < dn * D[k] ** 2 - dd * lam[k][k - 1] ** 2:
            swapi(k)
            k = max(1, k - 1)
        else:
            for l in range(k - 2, -1, -1):
                redi(k, l)
            k += 1
    return B


# ----------------------------------------------------------------------------
# Meet-in-the-middle on a subset of items (numpy, int64)
# ----------------------------------------------------------------------------
def _all_sums(ws):
    s = np.zeros(1, dtype=np.int64)
    for w in ws:
        s = np.concatenate([s, s + np.int64(w)])
    return s                                   # index bit j  <->  item j


def mitm_exact(ws: List[int], target: int):
    """Subset of ws (list of Python ints, total < 2^62) summing to target, or None."""
    h = len(ws) // 2
    L, R = _all_sums(ws[:h]), _all_sums(ws[h:])
    order = np.argsort(L, kind="stable")
    Ls = L[order]
    need = np.int64(target) - R
    pos = np.searchsorted(Ls, need)
    pos[pos >= len(Ls)] = len(Ls) - 1
    hit = np.nonzero(Ls[pos] == need)[0]
    if len(hit) == 0:
        return None
    ri = int(hit[0])
    li = int(order[pos[ri]])
    out = [(li >> j) & 1 for j in range(h)] + [(ri >> j) & 1 for j in range(len(ws) - h)]
    return out


def flip_search_mitm(a, T, xhat, margins, L):
    """Chase-type decoding done exactly: choose which of the L weakest bits to flip."""
    n = len(a)
    L = min(L, n)
    U = sorted(range(n), key=lambda i: margins[i])[:L]
    s0 = sum(ai for ai, xi in zip(a, xhat) if xi)
    if s0 == T:
        return list(xhat)
    delta = [(-a[i] if xhat[i] else a[i]) for i in U]
    # bring to non-negative weights for the numpy routine: flip sign trick
    off = sum(-d for d in delta if d < 0)
    ws = [abs(d) for d in delta]
    # sum_{S} delta_i = sum_{S, d>0} |d| - sum_{S, d<0} |d|;  substitute y_i = 1-s_i for d<0
    neg = [d < 0 for d in delta]
    need = (T - s0) + off                    # = sum over (s_i if d>0 else 1-s_i) * |d_i|
    if need < 0 or need > sum(ws) or sum(ws) >= 2 ** 62:
        return None
    sol = mitm_exact(ws, need)
    if sol is None:
        return None
    y = list(xhat)
    for j, i in enumerate(U):
        flip = (1 - sol[j]) if neg[j] else sol[j]
        if flip:
            y[i] ^= 1
    return y


# ----------------------------------------------------------------------------
# LOW-DENSITY: lattice frames -> erasure channel -> pooling -> decoding
# ----------------------------------------------------------------------------
def lattice_frame(a, T, rng):
    """One 'exposure': randomised LLL of the CJLOSS embedding. Returns list of rows (e, last)."""
    n = len(a)
    perm = list(range(n))
    rng.shuffle(perm)
    Nw = int(math.isqrt(n)) + 1 + rng.randrange(0, 3)    # random embedding scale
    B = []
    for r, i in enumerate(perm):
        row = [0] * (n + 1)
        row[r] = 2
        row[n] = Nw * a[i]
        B.append(row)
    B.append([1] * n + [Nw * T])
    R = lll(B)
    inv = {r: i for r, i in enumerate(perm)}
    rows = []
    for v in R:
        if v[n] != 0:
            continue
        e = [0] * n
        for r in range(n):
            e[inv[r]] = v[r]
        rows.append(e)
    return rows


def solve_lattice(a, T, max_frames=12, L=22, seed=None, deadline=None):
    rng = random.Random(seed)
    n = len(a)
    D = [0.0] * n
    for frame in range(1, max_frames + 1):
        rows = lattice_frame(a, T, rng)
        for e in rows:
            for sgn in (1, -1):                       # the +/- ambiguity of a lattice vector
                f = [sgn * v for v in e]
                if all(abs(v) == 1 for v in f):       # a clean readout -> try directly
                    x = [1 if v > 0 else 0 for v in f]
                    if verify(a, T, x):
                        return x, {"frames": frame, "how": "clean short vector"}
        # pool: align signs by closeness of the hard decode to the target, accumulate votes
        for e in rows:
            if all(v == 0 for v in e) or max(abs(v) for v in e) > 2:
                continue
            best = min((1, -1), key=lambda s: abs(
                sum(ai for ai, v in zip(a, e) if s * v > 0) - T))
            for i, v in enumerate(e):
                D[i] += best * v
        xhat = [1 if d > 0 else 0 for d in D]
        y = flip_search_mitm(a, T, xhat, [abs(d) for d in D], L)
        if y is not None and verify(a, T, y):
            return y, {"frames": frame, "how": f"pooled votes + MITM list decode (L={L})"}
        if deadline and time.time() > deadline:
            break
    return None, {"frames": max_frames, "how": "lattice failed"}


# ----------------------------------------------------------------------------
# HIGH-DENSITY: posterior-restricted MITM
# ----------------------------------------------------------------------------
def greedy_votes(a, T, rng, cycles=60):
    n = len(a)
    D = [0.0] * n
    scale = max(1.0, sum(a) / n)
    for _ in range(cycles):
        order = sorted(range(n), key=lambda i: -a[i] * (1 + 0.5 * rng.random()))
        x, s = [0] * n, 0
        for i in order:
            if s + a[i] <= T:
                x[i] = 1
                s += a[i]
        w = math.exp(-abs(s - T) / scale)
        for i in range(n):
            D[i] += w if x[i] else -w
    return D


def solve_dense(a, T, m=None, max_trials=400, seed=None, deadline=None):
    rng = random.Random(seed)
    n = len(a)
    if sum(a) >= 2 ** 62:
        return None, {"trials": 0, "how": "weights too large for int64 dense path"}
    bits = max(a).bit_length()
    m = m or min(n, bits + 3, 44)
    D = greedy_votes(a, T, rng)
    sc = (max(abs(d) for d in D) or 1.0)
    for trial in range(1, max_trials + 1):
        # free set: random, biased toward uncertain items (small |D|)
        keyed = sorted(range(n), key=lambda i: abs(D[i]) / sc + rng.random())
        free, fixed = keyed[:m], keyed[m:]
        s_fix, x = 0, [0] * n
        for i in fixed:
            p = 0.5 + 0.5 * math.tanh(2.0 * D[i] / sc)   # posterior (flat if votes useless)
            if rng.random() < p:
                x[i] = 1
                s_fix += a[i]
        Tp = T - s_fix
        ws = [a[i] for i in free]
        if Tp < 0 or Tp > sum(ws):
            continue
        sol = mitm_exact(ws, Tp)
        if sol is not None:
            for j, i in enumerate(free):
                x[i] = sol[j]
            if verify(a, T, x):
                return x, {"trials": trial, "how": f"restricted MITM (m={m} free of n={n})"}
        if deadline and time.time() > deadline:
            break
    return None, {"trials": max_trials, "how": "dense failed"}


# ----------------------------------------------------------------------------
# Auto dispatcher
# ----------------------------------------------------------------------------
def solve(a, T, seed=None, time_limit=120.0):
    n = len(a)
    bits = max(a).bit_length()
    density = n / bits
    t_end = time.time() + time_limit
    info = {"n": n, "bits": bits, "density": round(density, 3)}
    if n <= 40 and sum(a) < 2 ** 62:
        x = mitm_exact(list(a), T)
        if x is not None and verify(a, T, x):
            return x, {**info, "how": "exact MITM (n<=40)"}
        return None, {**info, "how": "exact MITM: no solution"}
    order = ["lattice", "dense"] if density < 0.94 else ["dense", "lattice"]
    for method in order:
        if time.time() > t_end:
            break
        if method == "lattice":
            x, i2 = solve_lattice(a, T, seed=seed, deadline=t_end)
        else:
            x, i2 = solve_dense(a, T, seed=seed, deadline=t_end)
        if x is not None and verify(a, T, x):
            return x, {**info, **i2}
    return None, {**info, "how": "failed"}


# ----------------------------------------------------------------------------
# PROOF HARNESS
# ----------------------------------------------------------------------------
def independent_check(a, T, x):
    """Re-verification written separately from verify(): plain loop, Python big ints."""
    total = 0
    for i in range(len(a)):
        assert x[i] in (0, 1)
        if x[i] == 1:
            total += a[i]
    return total == T, total


def planted(n, bits, seed):
    rng = random.Random(seed)
    a = [rng.getrandbits(bits) | 1 for _ in range(n)]
    x = [rng.randint(0, 1) for _ in range(n)]
    return a, sum(ai for ai, xi in zip(a, x) if xi)


def prove(configs, seeds=3, time_limit=150.0, out="proof_results.json"):
    results = []
    print(f"{'n':>4} {'bits':>4} {'dens':>5} {'seed':>4} {'ok':>5} {'sec':>7}  method")
    print("-" * 78)
    for n, bits in configs:
        for sd in range(seeds):
            a, T = planted(n, bits, seed=1000 * n + sd)
            t0 = time.time()
            x, info = solve(a, T, seed=sd, time_limit=time_limit)
            dt = time.time() - t0
            ok = False
            tot = None
            if x is not None:
                ok, tot = independent_check(a, T, x)
            print(f"{n:>4} {bits:>4} {n/bits:>5.2f} {sd:>4} {str(ok):>5} {dt:>7.2f}  {info['how']}")
            results.append({"n": n, "bits": bits, "seed": sd, "solved": ok, "seconds": round(dt, 3),
                            "method": info["how"], "target": T, "achieved_sum": tot,
                            "chosen_indices": [i for i, v in enumerate(x or []) if v]})
    # negative control: impossible target must never be "solved"
    a, T = planted(60, 120, seed=7)
    x, info = solve(a, sum(a) + 1, seed=0, time_limit=20)
    neg_ok = x is None
    print(f"\nnegative control (target > sum of all weights): returned None = {neg_ok}")
    with open(out, "w") as f:
        json.dump({"results": results, "negative_control_passed": neg_ok}, f, indent=1)
    solved = sum(r["solved"] for r in results)
    print(f"\nsolved {solved}/{len(results)} instances; every success re-verified with exact integer sums.")
    return results


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    cfg = [(30, 20), (40, 24),                       # small / exact
           (50, 32), (56, 40),                       # dense regime (v1 failed here)
           (40, 100), (60, 130), (80, 200), (100, 400)]   # low density, up to 400-bit weights
    if "--big" in sys.argv:
        cfg += [(64, 48)]                            # dense, ~100 s each in pure Python
    prove(cfg, seeds=2)
