#!/usr/bin/env python3
"""
Australia fiscal scenario simulator using the paper's subset-sum decoder.

Requires subset_sum_paper.py in the same folder:
    from subset_sum_paper import solve, verify

For every scenario, the decoder chooses a package of Budget function cuts that
sums EXACTLY to the scenario's cut target. If no exact package exists, the
script reports the closest reachable package and its residual difference.

MONEY UNITS: AUD millions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
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


ELASTICITY = 0.03
FREEDOM_KEYS = [
    "property_rights", "judicial_effectiveness", "government_integrity",
    "tax_burden", "government_spending", "fiscal_health",
    "business_freedom", "labor_freedom", "monetary_freedom",
    "trade_freedom", "investment_freedom", "financial_freedom",
]


@dataclass
class SocialClass:
    name: str
    pop_share: float
    income_share: float
    tax_rate: float
    transfer_per_capita: float = 0.0


@dataclass
class Country:
    name: str
    population: int
    gdp: float
    classes: List[SocialClass]
    spending: Dict[str, float]
    welfare: float
    debt: float
    interest_rate: float
    base_growth: float
    inflation: float
    freedom: Dict[str, float] = field(default_factory=dict)

    def revenue(self, tax_add=0.0, top_add=0.0, gdp=None):
        gdp = self.gdp if gdp is None else gdp
        total = 0.0
        for c in self.classes:
            rate = c.tax_rate + tax_add
            if c.name.startswith("Upper"):
                rate += top_add
            total += c.income_share * gdp * rate
        return total

    def transfers(self, transfer_mult=1.0):
        return sum(
            c.transfer_per_capita * c.pop_share * self.population / 1e6 * transfer_mult
            for c in self.classes
        )

    def programme_spending(self):
        return sum(self.spending.values())

    def interest(self, debt=None, rate=None):
        debt = self.debt if debt is None else debt
        rate = self.interest_rate if rate is None else rate
        return debt * rate

    def freedom_score(self):
        vals = [self.freedom[k] for k in FREEDOM_KEYS if k in self.freedom]
        return sum(vals) / len(vals) if vals else 60.0

    def project(self, years=10, cut_package: Optional[List[str]] = None,
                tax_add=0.0, top_add=0.0, transfer_mult=1.0,
                spend_growth=None, rate=None, freedom_shift=0.0):
        score = self.freedom_score() + freedom_shift
        real_growth = (self.base_growth + ELASTICITY * (score - 60)) / 100
        nominal_growth = real_growth + self.inflation / 100
        spend_growth = nominal_growth if spend_growth is None else spend_growth

        cut_total = sum(self.spending[k] for k in (cut_package or []))
        programme_base = self.programme_spending() - cut_total

        gdp, debt, spending_index = self.gdp, self.debt, 1.0
        rows = []

        for year in range(1, years + 1):
            revenue = self.revenue(tax_add, top_add, gdp)
            transfers = self.transfers(transfer_mult) * spending_index
            interest = self.interest(debt, rate)
            programme = programme_base * spending_index
            welfare = self.welfare * transfer_mult * spending_index
            spending = programme + welfare + transfers + interest

            deficit = spending - revenue
            gdp *= 1 + nominal_growth
            debt += deficit
            spending_index *= 1 + spend_growth

            rows.append({
                "year": year,
                "gdp_aud_b": round(gdp / 1000, 1),
                "debt_aud_b": round(debt / 1000, 1),
                "deficit_aud_b": round(deficit / 1000, 2),
                "debt_%gdp": round(100 * debt / gdp, 1),
            })

        return rows

    def budget_fit(self, target, seed=0, **kw):
        """Exact subset-sum search using the paper decoder."""
        names = list(self.spending)
        amounts = [max(1, int(round(self.spending[k]))) for k in names]
        target_int = int(round(target))

        solution, info = solve(amounts, target_int, seed=seed, **kw)
        if solution is None:
            return None, info

        chosen = [name for name, selected in zip(names, solution) if selected]
        assert verify(amounts, target_int, solution)
        return chosen, info

    def budget_fit_closest(self, target):
        """DP fallback: closest reachable subset when the decoder finds no exact match."""
        pairs = [(name, max(1, int(round(value)))) for name, value in self.spending.items()]
        target_int = int(round(target))

        reachable = {0: ()}
        for name, value in pairs:
            updated = dict(reachable)
            for total, chosen in reachable.items():
                new_total = total + value
                if new_total <= target_int and new_total not in updated:
                    updated[new_total] = chosen + (name,)
            reachable = updated

        if not reachable:
            return None, None

        best_total = min(reachable, key=lambda s: abs(s - target_int))
        return list(reachable[best_total]), best_total


def australia():
    return Country(
        name="Australia",
        population=27_600_000,
        gdp=2_930_326,  # AUD millions
        classes=[
            SocialClass("Lower 40%", 0.40, 0.15, 0.095, 22_222),
            SocialClass("Middle 45%", 0.45, 0.48, 0.250, 4_938),
            SocialClass("Upper 15%", 0.15, 0.37, 0.375, 2_156),
        ],
        spending={
            "General public services": 32_499,
            "Defence": 52_055,
            "Public order and safety": 9_709,
            "Education": 57_349,
            "Health": 136_918,
            "Housing and community amenities": 11_454,
            "Recreation and culture": 6_292,
            "Fuel and energy": 15_207,
            "Agriculture, forestry and fishing": 4_748,
            "Mining, manufacturing and construction": 5_563,
            "Transport and communication": 15_338,
            "Other economic affairs": 13_773,
            "General purpose intergovernmental transactions": 111_062,
            "Natural disaster relief": 865,
            "Contingency reserve": 3_925,
            "Other purposes / residual": 4_404,
        },
        welfare=308_737,
        debt=1_051_000,
        interest_rate=0.0413,
        base_growth=1.75,
        inflation=2.5,
        freedom={
            "property_rights": 88.2,
            "judicial_effectiveness": 96.3,
            "government_integrity": 88.1,
            "tax_burden": 61.8,
            "government_spending": 56.6,
            "fiscal_health": 89.0,
            "business_freedom": 92.3,
            "labor_freedom": 61.5,
            "monetary_freedom": 77.2,
            "trade_freedom": 89.8,
            "investment_freedom": 80.0,
            "financial_freedom": 80.0,
        },
    )


SCENARIOS = {
    "Baseline": {
        "target": 0,
    },
    "Programme cut: 2%": {
        "target": 9_623,
    },
    "Programme cut: 5%": {
        "target": 24_058,
    },
    "Programme cut: 10%": {
        "target": 48_116,
    },
    "A$30 billion cut": {
        "target": 30_000,
    },
    "A$50 billion cut": {
        "target": 50_000,
    },
    "A$100 billion cut": {
        "target": 100_000,
    },
    "Tax rates +1pp": {
        "target": 0,
        "tax_add": 0.01,
    },
    "Top-class tax +3pp": {
        "target": 0,
        "top_add": 0.03,
    },
    "Transfers -5%": {
        "target": 0,
        "transfer_mult": 0.95,
    },
    "Interest rate +1pp": {
        "target": 0,
        "rate": 0.0513,
    },
    "Freedom score -10": {
        "target": 0,
        "freedom_shift": -10,
    },
    "Austerity package": {
        "target": 9_538,  # exact: agriculture + disaster relief + contingency
        "transfer_mult": 0.95,
    },
}


def run_all(au: Country, years=10, solver_kwargs=None):
    solver_kwargs = solver_kwargs or {
        "max_frames": 12,
        "L": 16,
        "t": 4,
        "q_max": 200_000,
        "best_first": True,
        "seed": 0,
        "time_limit": 120.0,
    }

    base_nominal_growth = au.base_growth / 100 + au.inflation / 100
    results = []

    print("=== Australia subset-sum fiscal scenarios ===")
    print(f"Programme spending available for cuts: A${au.programme_spending():,}m\n")

    for name, cfg in SCENARIOS.items():
        target = cfg.get("target", 0)
        cut_package = None
        exact = False
        solver_info = None

        if target > 0:
            cut_package, solver_info = au.budget_fit(target, **solver_kwargs)

            if cut_package is None:
                cut_package, closest_total = au.budget_fit_closest(target)
                exact = False
                difference = closest_total - target
            else:
                exact = True
                closest_total = target
                difference = 0
        else:
            closest_total = 0
            difference = 0

        settings = {
            "cut_package": cut_package,
            "tax_add": cfg.get("tax_add", 0.0),
            "top_add": cfg.get("top_add", 0.0),
            "transfer_mult": cfg.get("transfer_mult", 1.0),
            "rate": cfg.get("rate", au.interest_rate),
            "freedom_shift": cfg.get("freedom_shift", 0.0),
        }

        if name == "Spending growth -1pp":
            settings["spend_growth"] = base_nominal_growth - 0.01
        if name == "Spending growth -2pp":
            settings["spend_growth"] = base_nominal_growth - 0.02

        path = au.project(years=years, **settings)

        result = {
            "scenario": name,
            "target_aud_m": target,
            "exact_subset": exact,
            "achieved_cut_aud_m": closest_total,
            "difference_aud_m": difference,
            "cut_items": cut_package or [],
            "year1_deficit_aud_b": path[0]["deficit_aud_b"],
            "year10_deficit_aud_b": path[-1]["deficit_aud_b"],
            "debt_gdp_year5": path[4]["debt_%gdp"],
            "debt_gdp_year10": path[-1]["debt_%gdp"],
            "cumulative_deficit_aud_b": round(sum(r["deficit_aud_b"] for r in path), 1),
            "solver_info": solver_info,
        }
        results.append(result)

        print(f"\n--- {name} ---")
        print(f"Target cut: A${target:,}m")
        if target == 0:
            print("No spending-cut package required.")
        elif exact:
            print(f"EXACT package found: A${closest_total:,}m")
            print(f"Items: {cut_package}")
            print(f"Decoder: {solver_info['how']}")
        else:
            print(f"No exact package; closest reachable cut: A${closest_total:,}m "
                  f"({difference:+,}m)")
            print(f"Items: {cut_package}")

        print(f"Year-1 deficit: A${result['year1_deficit_aud_b']:.1f}b")
        print(f"Year-10 deficit: A${result['year10_deficit_aud_b']:.1f}b")
        print(f"Debt/GDP year 5: {result['debt_gdp_year5']:.1f}%")
        print(f"Debt/GDP year 10: {result['debt_gdp_year10']:.1f}%")
        print(f"Cumulative 10-year deficit: A${result['cumulative_deficit_aud_b']:.1f}b")

    return results


if __name__ == "__main__":
    au = australia()

    print(f"== {au.name} ==")
    print(f"Population: {au.population:,}")
    print(f"Nominal GDP: A${au.gdp / 1e6:.3f} trillion")
    print(f"GDP per capita: A${au.gdp * 1e6 / au.population:,.0f}")
    print(f"Freedom score: {au.freedom_score():.1f}")
    print(f"Gross debt: A${au.debt / 1000:.1f}b ({100 * au.debt / au.gdp:.1f}% of GDP)")

    results = run_all(au)

    import json
    with open("australia_subset_sum_scenarios.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved detailed results to australia_subset_sum_scenarios.json")
