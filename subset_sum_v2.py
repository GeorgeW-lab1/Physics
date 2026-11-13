#!/usr/bin/env python3
"""
Subset-sum recovery using ONLY the paper's decoding pipeline (no meet-in-the-middle).

  Paper stage                          Here
  -----------------------------------  ------------------------------------------------------
  Oracle commitment c                  instance (a, T); verifier V(x) = [a.x == T]
  Camera frame / parallel pooling      one randomised LLL reduction of the CJLOSS embedding;
                                       a reduced row with last coord 0 is a readout of 2x-1:
                                       +1 / -1 valid bit, 0 = ERASURE (Sec. 2.3), |v|>1 = invalid
  Vote accumulation (Sec. 4.1)         D_i = sum_t w_t * sign-aligned readout  (vectorised, numpy)
  Majority decision (ties -> 0)        x_hat_i = [D_i > 0]
  Chase-type list decoder (Sec. 5.2)   flip <= t bits among the L least-reliable positions,
                                       candidates tested against the exact verifier
  Best-first enumeration (Sec. 5.6)    optional: test flip sets in increasing W(S)=sum |D_i|
  Candidate budget Q_max (Sec. 5.1)    hard cap on verifier calls

The decoder never searches the 2^L space exhaustively and never uses meet-in-the-middle:
it can only fix <= t wrong bits located inside the L least-reliable positions (Sec. 5.3:
success iff K <= t and E subset of U).
Frames come from lattice reduction because the greedy heuristic of v1 was not an informative
channel (it carried no usable per-bit signal at n >= 50).
"""
import heapq
import itertools
import json
import math
import random
import sys
import time
from math import comb
from typing import List, Optional, Sequence

import numpy as np


def verify(a: Sequence[int], T: int, x: Sequence[int]) -> bool:
    return len(a) == len(x) and sum(ai for ai, xi in zip(a, x) if xi) == T


# ---------------------------------------------------------------- integral LLL (Cohen 2.6.7)
def lll(B, dn=99, dd=100):
    B = [r[:] for r in B]
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


# ---------------------------------------------------------------- one frame
def lattice_frame(a, T, rng):
    """Randomised LLL -> matrix of readouts E (rows x n), entries 0 (erasure) / +-1 / other(invalid)."""
    n = len(a)
    perm = list(range(n))
    rng.shuffle(perm)
    Nw = math.isqrt(n) + 1 + rng.randrange(0, 3)
    B = []
    for r, i in enumerate(perm):
        row = [0] * (n + 1)
        row[r] = 2
        row[n] = Nw * a[i]
        B.append(row)
    B.append([1] * n + [Nw * T])
    R = lll(B)
    rows = []
    for v in R:
        if v[n] != 0:
            continue
        e = [0] * n
        for r in range(n):
            e[perm[r]] = v[r]
        if any(e):
            rows.append(e)
    return rows


# ---------------------------------------------------------------- pooling (vectorised)
class Pool:
    """Sufficient statistic D_i plus the paper's channel diagnostics."""

    def __init__(self, a, T):
        self.a = np.array([float(x) for x in a])
        self.T = float(T)
        self.n = len(a)
        self.D = np.zeros(self.n)
        self.valid_obs = np.zeros(self.n, dtype=int)    # N_i: valid observations of cell i
        self.rows_seen = 0
        self.cover_frame = None

    def add_frame(self, rows, frame_no):
        if not rows:
            return
        E = np.array(rows, dtype=np.int64)                      # (rows, n)
        # confidence gate (Sec. 2.3): a readout with any |v| > 1 is out of range -> the whole
        # row is invalid and is dropped (erased), instead of polluting the vote
        E = E[np.abs(E).max(axis=1) <= 1]
        if len(E) == 0:
            return
        V = (np.abs(E) == 1)                                    # valid readouts
        S = np.where(V, E, 0).astype(float)                     # erased / invalid -> 0
        # sign alignment: lattice vectors are defined up to +-1; pick the sign whose hard
        # decode lands closer to T (one batched matrix-vector product, no per-bit loop)
        plus = ((S > 0) * self.a).sum(axis=1)
        minus = ((S < 0) * self.a).sum(axis=1)
        sgn = np.where(np.abs(plus - self.T) <= np.abs(minus - self.T), 1.0, -1.0)
        S = S * sgn[:, None]
        w = V.mean(axis=1) ** 2                                 # reveal-rate weighting
        self.D += (S * w[:, None]).sum(axis=0)
        self.valid_obs += V.sum(axis=0).astype(int)
        self.rows_seen += len(E)
        if self.cover_frame is None and (self.valid_obs > 0).all():
            self.cover_frame = frame_no

    @property
    def reveal_rate(self):
        return float((self.valid_obs > 0).mean())

    def hard_decode(self):
        return [1 if d > 0 else 0 for d in self.D]               # ties -> 0 (paper Sec. 4.3)

    def margins(self):
        return np.abs(self.D)


# ---------------------------------------------------------------- Chase decoding
def chase(a, T, xhat, margins, L=16, t=4, budget=200_000, best_first=False):
    """Returns (solution or None, candidates_tested). No meet-in-the-middle."""
    n = len(a)
    L = min(L, n)
    # ascending reliability; np.argsort stable -> deterministic ties (paper Sec. 9, item 5)
    U = [int(i) for i in np.argsort(margins, kind="stable")[:L]]
    s0 = sum(ai for ai, xi in zip(a, xhat) if xi)
    tested = 1
    if s0 == T:
        return list(xhat), tested
    delta = [(-a[i] if xhat[i] else a[i]) for i in U]           # effect of flipping U[j]

    if not best_first:                                          # Sec. 5.2: by cardinality
        for k in range(1, min(t, L) + 1):
            for S in itertools.combinations(range(L), k):
                tested += 1
                if tested > budget:
                    return None, tested
                if s0 + sum(delta[j] for j in S) == T:
                    y = list(xhat)
                    for j in S:
                        y[U[j]] ^= 1
                    return y, tested
        return None, tested

    # Sec. 5.6: best-first, lowest W(S) = sum of margins first, via a heap
    w = [float(margins[i]) for i in U]
    heap = [(w[0], (0,))]
    while heap:
        cost, S = heapq.heappop(heap)
        tested += 1
        if tested > budget:
            return None, tested
        if s0 + sum(delta[j] for j in S) == T:
            y = list(xhat)
            for j in S:
                y[U[j]] ^= 1
            return y, tested
        last = S[-1]
        if last + 1 < L:
            heapq.heappush(heap, (cost - w[last] + w[last + 1], S[:-1] + (last + 1,)))   # shift
            if len(S) < t:
                heapq.heappush(heap, (cost + w[last + 1], S + (last + 1,)))              # extend
    return None, tested


# ---------------------------------------------------------------- top level
def solve(a, T, max_frames=12, L=16, t=4, q_max=200_000, best_first=True,
          seed=None, time_limit=120.0):
    rng = random.Random(seed)
    pool = Pool(a, T)
    t_end = time.time() + time_limit
    total_tested = 0
    last_key = None
    info = {}
    for f in range(1, max_frames + 1):
        rows = lattice_frame(a, T, rng)
        pool.add_frame(rows, f)
        xhat = pool.hard_decode()
        if verify(a, T, xhat):
            return xhat, {"frames": f, "how": "majority vote (0 flips)", "candidates": total_tested + 1,
                          "cover_frame": pool.cover_frame, "reveal_rate": pool.reveal_rate}
        if pool.cover_frame is None:                             # decoder waits for full cover
            continue
        key = (tuple(xhat), tuple(np.argsort(pool.margins(), kind="stable")[:L]))
        if key != last_key:
            y, tested = chase(a, T, xhat, pool.margins(), L, t, q_max - total_tested, best_first)
            total_tested += tested
            last_key = key
            if y is not None and verify(a, T, y):
                return y, {"frames": f, "how": f"Chase ({'best-first' if best_first else 'by weight'}, "
                           f"L={L}, t={t})", "candidates": total_tested,
                           "cover_frame": pool.cover_frame, "reveal_rate": pool.reveal_rate}
        if total_tested >= q_max or time.time() > t_end:
            break
    return None, {"frames": f, "how": "failed", "candidates": total_tested,
                  "cover_frame": pool.cover_frame, "reveal_rate": pool.reveal_rate}


# ---------------------------------------------------------------- predictions & proof harness
def predicted_budget(L, t, q_max):
    V = sum(comb(L, k) for k in range(t + 1))
    return {"V(L,t)": V, "cycles_to_Qmax": math.ceil(q_max / V),
            "False-accept bound": "0 (exact integer verifier, no hash collisions)"}


def planted(n, bits, seed):
    rng = random.Random(seed)
    a = [rng.getrandbits(bits) | 1 for _ in range(n)]
    x = [rng.randint(0, 1) for _ in range(n)]
    return a, sum(ai for ai, xi in zip(a, x) if xi)


def independent_check(a, T, x):
    total = 0
    for i in range(len(a)):
        assert x[i] in (0, 1)
        if x[i]:
            total += a[i]
    return total == T, total


def prove(configs, seeds=3, out="proof_paper_results.json", **kw):
    rows = []
    print(f"{'n':>4} {'bits':>4} {'dens':>5} {'seed':>4} {'ok':>5} {'frm':>3} {'cov':>3} {'cand':>7} {'sec':>6}  how")
    print("-" * 92)
    for n, bits in configs:
        for sd in range(seeds):
            a, T = planted(n, bits, 1000 * n + 7 * bits + sd)
            t0 = time.time()
            x, info = solve(a, T, seed=sd, **kw)
            dt = time.time() - t0
            ok, tot = (False, None)
            if x is not None:
                ok, tot = independent_check(a, T, x)
            print(f"{n:>4} {bits:>4} {n/bits:>5.2f} {sd:>4} {str(ok):>5} {info['frames']:>3} "
                  f"{str(info['cover_frame']):>3} {info['candidates']:>7} {dt:>6.1f}  {info['how']}")
            rows.append({"n": n, "bits": bits, "seed": sd, "solved": ok, "seconds": round(dt, 2),
                         "target": T, "achieved_sum": tot, **info,
                         "indices": [i for i, v in enumerate(x or []) if v]})
    a, T = planted(60, 120, 7)
    x, _ = solve(a, sum(a) + 1, seed=0, max_frames=3, time_limit=30)
    print(f"\nnegative control (unreachable target) -> None: {x is None}")
    json.dump({"results": rows, "negative_control_passed": x is None}, open(out, "w"), indent=1)
    print(f"solved {sum(r['solved'] for r in rows)}/{len(rows)}; each re-summed with exact integers.")
    return rows


def validate_decoder(m=56, r=0.45, eps=0.02, trials=200, max_frames=60, L=16, t=4,
                     best_first=False, seed=0):
    """
    DECODER VALIDATION, not a solver: frames are synthetic (the paper's Sec. 2.5 channel),
    generated from a planted solution, so this only tests pooling + Chase against the
    paper's predictions (cover time ~ (ln m + gamma)/(-ln(1-r)); success shortly after).
    """
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    cover, success, used = [], [], []
    for _ in range(trials):
        a = [rng.getrandbits(40) | 1 for _ in range(m)]
        x = np.array([rng.randint(0, 1) for _ in range(m)])
        T = int(sum(ai for ai, xi in zip(a, x) if xi))
        ones = np.zeros(m, int); zeros = np.zeros(m, int)
        cov = None
        for f in range(1, max_frames + 1):
            reveal = nrng.random(m) < r
            flip = nrng.random(m) < eps
            obs = x ^ flip
            ones += reveal & (obs == 1)
            zeros += reveal & (obs == 0)
            if cov is None and ((ones + zeros) > 0).all():
                cov = f
            if cov is None:
                continue
            Dv = ones - zeros
            xh = [1 if d > 0 else 0 for d in Dv]
            y, _ = chase(a, T, xh, np.abs(Dv), L, t, 200_000, best_first)
            if y is not None and verify(a, T, y):
                cover.append(cov); success.append(f); break
    return {"trials": trials, "solved": len(success),
            "mean_cover_frames": round(float(np.mean(cover)), 2) if cover else None,
            "mean_frames_to_solve": round(float(np.mean(success)), 2) if success else None,
            "predicted_cover": round((math.log(m) + 0.5772) / -math.log(1 - r), 2)}


if __name__ == "__main__":
    print("Predicted budget (L=16,t=4,Qmax=2e5):", predicted_budget(16, 4, 200_000), "\n")
    cfg = [(30, 30), (40, 55), (50, 80), (60, 110), (60, 130), (80, 200), (100, 400), (120, 600), (140, 800)]
    if "--hard" in sys.argv:
        cfg = [(30, 30), (40, 45), (40, 55), (50, 60), (60, 75), (60, 90)]
    prove(cfg, seeds=2)
    print("\nDecoder validation on the paper's synthetic channel (planted-noise frames):")
    print(" by weight :", validate_decoder(best_first=False))
    print(" best-first:", validate_decoder(best_first=True))
