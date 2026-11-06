"""
RP-style optoelectronic decision circuit driven by a USB camera  (single file)

    pip install numpy opencv-python
    python rp_circuit_camera.py

Every camera frame is one trial of the circuit:

  1. Motion detection: the frame is shrunk to 128x96, blurred and differenced
     against the previous frame. Pixels that changed by more than a threshold
     form the motion mask.
  2. Instance x = number of moving pixels. (No motion -> no instance, Q = 0.)
     The language is L = { odd composite numbers }.
  3. Driver A (randomized witness test, Detector A = motion energy):
        the same camera frame feeds both detectors and no sensor noise is
        used. r is derived from the motion pattern itself (the size of the
        change at every moving pixel, hashed with SHA-256), so it is
        unpredictable the way the movement is. A = 1 if r is a Miller-Rabin
        witness that x is composite. A = 0 for every prime, so the error is
        one-sided.
  4. Driver B (deterministic check, Detector B = motion mask):
        c(x) = "x is odd and > 2". It drives capacitor C through R for one
        frame period; the NPN emitter follower gives V(B) = V(C) - 0.7 V and
        B is a valid logic HIGH at >= 3.5 V.
  5. Gates: OR and AND (tied outputs read as AND), 10 uF node (low-pass),
     time code Y (which frames may pass), 74HC74 latches D on the clock edge.
     Here the clock edge is the end of each camera frame.
  6. SELECT = "any Q = 1 in the last k frames" (the repetition step).

Driver analysis (button "Driver analysis"): every instance frame is scored against
the truth (is x really an odd composite?). For Driver A, Driver B and the final Q
it reports TPR/FPR/PPV/NPV/accuracy and mutual information with "x in L", checks
the RP properties the bounds rest on (A never fires outside L, A's witness rate
p >= 1/2, B never misses a member of L), calibrates A against the exact witness
density of every x, and tests the independence the (1-p)^N bound assumes.

Time code Y (proved in time_code_proof_check.py, shown live in the window):
  * Gating reading, D = A.B.Y: x not in L is never accepted for ANY Y (Thm 1).
    For x in L the false-reject probability is exactly (1-p)^N, N = open bins
    in the k-frame window (Thm 2), so Y sets the error exponent, not the sidedness.
  * Y accepts arithmetic: "1000|0011" OR-adds codes (N never drops, disjoint
    codes multiply their errors), "1111&1011" ANDs them (N never rises),
    "1011+0001" adds them as binary numbers: popcount(a+b) = popcount(a) +
    popcount(b) - carries, so each carry loses one open bin (Thm 3).
  * "forces HIGH" read as OR (checkbox) lets a forced bin accept any x, so the
    one-sided guarantee is lost (Thm 4). Use it to watch false accepts appear.

Concurrency: the camera read runs in a worker thread; Driver A and Driver B
process each frame concurrently as asyncio tasks; the flip-flop stage runs
once both are finished. The whole pipeline lives in a background thread with
its own asyncio loop and talks to Tk through a queue.Queue.

Two sequential even states: a hash is mined, and the hidden number is guessed, only on a
frame whose x is even AND whose previous frame's x was also even. An odd x or a no-motion
frame in between breaks the run, so the next even frame only starts a new one.

Higher/lower search (separate from both): a secret of N bits (default 1000) and a
higher/lower oracle. On every armed frame (two sequential evens) one query yields one bit
of the secret, which is POOLED; no equality check is made per query. Only when all N bits
are pooled is the candidate compared with the secret, once. A 1000-bit secret falls in
1000 pooled queries plus 1 final check instead of about 2^1000 blind guesses. A found secret earns
one time-code bin and a new secret is hidden.

Hidden-number test (separate from the bit search): a secret integer is drawn in
0..range-1. A separate clock thread cycles a candidate through that range at a frequency
that you set in the GUI (1 Hz to 2 kHz, default 17 Hz). The thread never sees the
secret. Only when two sequential EVEN states have arrived does the frame loop spot-check the candidate AT
THAT MOMENT; if it equals the secret, one time-code bin is added (same mechanism as a found
secret). After EVERY spot check, hit or miss, a new secret is hidden. Expected hits = checks / range.
"New secret" (and changing the range) draws a fresh secret and zeroes the spot checks,
hits and luck factor. The luck factor is hits / expected hits in percent (100% = average),
shown with P(>= hits by chance), the binomial tail probability.

Factoriser (separate source, own panel): N is factored through the same higher/lower bit search.
The oracle holds the smallest prime factor p of N and answers only "p >= trial?". Each armed frame
pools one bit; one final check runs when the pool is full: 1 < p < N, N % p == 0 and the full prime
factorisation multiplies back to N. A time-code bin is earned ONLY after that verified solve.
Typed N is really factored by a background engine (trial division, Fermat, Pollard rho, p-1, ECM,
quadratic sieve; pip install gmpy2 for speed); the oracle is not ready until the engine finishes.
Random semiprimes keep their factors inside the oracle, which shows the mechanism only. A
higher/lower oracle for a factor cannot exist without already knowing it, so the search does not
make factoring itself cheaper. Camera motion hashes seed the engine's randomness.

Needs a real USB camera and OpenCV: motion comes only from what the camera
sees. There is no simulated source; if the camera can't be opened, it says so.
"""
import asyncio
import hashlib
import json
import math
import multiprocessing as mp
import queue
import secrets
import sys
import threading
import time
import urllib.request
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from tkinter import ttk, messagebox, filedialog

import numpy as np

try:
    import cv2
except ImportError:                      # reported when you press Start
    cv2 = None

# ------------------------------------------------------------------ constants
SW, SH = 128, 96                  # processing size
PW, PH = 288, 216                 # live preview size
WARMUP = 8                        # frames ignored while auto-exposure settles
VS = 5.0                          # driver B output voltage
V_BE = 0.7                        # NPN turn-on
V_IH = 3.5                        # 74HC input-high at 5 V
RC_B = 0.004                      # s, IF/THEN link RC (must settle within a frame)
TAU_NODE = 0.004                  # s, 10 uF node low-pass
LOGIC_TIME = RC_B * math.log(VS / (VS - V_IH - V_BE))   # 1.83 RC for these values


# ------------------------------------------------------------------ maths
def miller_rabin_witness(a: int, n: int) -> bool:
    """True if a proves n composite. Never True for a prime n."""
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    y = pow(a, d, n)
    if y in (1, n - 1):
        return False
    for _ in range(s - 1):
        y = y * y % n
        if y == n - 1:
            return False
    return True


def is_odd_composite(x: int) -> bool:
    """Exact for x < 3.3e24 (deterministic Miller-Rabin). The old trial division ran ~2e9
    iterations on 64-bit hashes and froze the whole program."""
    if x < 9 or x % 2 == 0:
        return False
    for p in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if x % p == 0:
            return x != p
    return any(miller_rabin_witness(a, x) for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37))


# ------------------------------------------------------------------ big-number engine setup
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)     # allow numbers with more than 4300 digits


# ------------------------------------------------------------------ big-number factoriser
# Ladder: trial division -> perfect power -> Fermat -> Pollard rho (Brent) -> Pollard p-1
#         -> ECM (Montgomery curves, stage 1 + stage 2) -> quadratic sieve.
# Optional speed-up:  pip install gmpy2   (used automatically when present)
try:
    import gmpy2
    _mpz = gmpy2.mpz
    _gcd = gmpy2.gcd
    _isqrt = gmpy2.isqrt
    _invert = gmpy2.invert
    HAVE_GMP = True
except ImportError:                                   # plain Python ints are fine, just slower
    gmpy2 = None
    _mpz = int
    _gcd = math.gcd
    _isqrt = math.isqrt
    _invert = lambda a, n: pow(a, -1, n)
    HAVE_GMP = False


class Cancelled(Exception):
    pass

class ToyPowChain:
    """Toy proof-of-work chain over price samples. Deterministic and verifiable.
    It commits to PAST data only, so it cannot predict the next move."""

    def __init__(self, difficulty_bits=12):
        self.difficulty = difficulty_bits
        self.prev = b"\x00" * 32
        self.last_nonce = 0
        self.last_hash = self.prev

    def _digest(self, prev, price, nonce):
        return hashlib.sha256(prev + repr(round(price, 8)).encode() +
                              nonce.to_bytes(8, "big")).digest()

    def mine(self, price):
        target = 1 << (256 - self.difficulty)
        nonce = 0
        while True:
            d = self._digest(self.prev, price, nonce)
            if int.from_bytes(d, "big") < target:
                self.prev, self.last_nonce, self.last_hash = d, nonce, d
                return d, nonce
            nonce += 1

    def verify(self, prev, price, nonce):
        d = self._digest(prev, price, nonce)
        return int.from_bytes(d, "big") < (1 << (256 - self.difficulty))

    def feature_bit(self):
        """One deterministic bit from the committed hash (usable as tree context)."""
        return "1" if self.last_hash[-1] & 1 else "0"
class EntropyPool:
    """Randomness for the factoriser. Camera motion frames are hashed into the pool when the
    circuit runs; the operating-system generator is always mixed in, so it works with no camera."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = secrets.token_bytes(32)
        self.frames = 0

    def feed(self, data: bytes):
        with self._lock:
            self._state = hashlib.sha256(self._state + data).digest()
            self.frames += 1

    def rand_bits(self, bits: int) -> int:
        out, ctr = b"", 0
        with self._lock:
            seed = self._state
            self._state = hashlib.sha256(self._state + b"\x01").digest()
        need = (bits + 7) // 8
        while len(out) < need:
            out += hashlib.sha256(seed + secrets.token_bytes(16) + ctr.to_bytes(4, "big")).digest()
            ctr += 1
        return int.from_bytes(out[:need], "big") & ((1 << bits) - 1)

    def randrange(self, lo: int, hi: int) -> int:
        span = hi - lo
        return lo + self.rand_bits(span.bit_length() + 32) % span


def _primes_upto(limit: int):
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    sieve[4::2] = False
    for i in range(3, int(limit ** 0.5) + 1, 2):
        if sieve[i]:
            sieve[i * i::2 * i] = False
    return np.nonzero(sieve)[0]


_SMALL_PRIMES = _primes_upto(200_000)


def is_probable_prime(n: int, rounds: int = 24) -> bool:
    n = int(n)
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    if gmpy2 is not None:
        return bool(gmpy2.is_prime(_mpz(n), rounds))
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if n.bit_length() <= 1024:
        bases += [2 + secrets.randbelow(n - 3) for _ in range(max(0, rounds - len(bases)))]
    for a in bases:
        a %= n
        if a < 2:
            continue
        y = pow(a, d, n)
        if y in (1, n - 1):
            continue
        for _ in range(s - 1):
            y = y * y % n
            if y == n - 1:
                break
        else:
            return False
    return True


def _iroot(n: int, k: int) -> int:
    """floor(n ** (1/k)) by Newton iteration."""
    if n < 2:
        return n
    x = 1 << ((n.bit_length() + k - 1) // k)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            return x
        x = y


def _perfect_power(n: int):
    """(base, exp) with base ** exp == n and exp > 1, else None."""
    for k in range(2, min(n.bit_length(), 128) + 1):
        r = _iroot(n, k)
        if r < 2:
            break
        if r ** k == n:
            return r, k
    return None


class Factoriser:
    def __init__(self, pool: EntropyPool, cancel, log, progress=None):
        self.pool = pool
        self.cancel = cancel                  # callable -> bool
        self.log = log
        self.progress = progress or (lambda *_: None)

    def check(self):
        if self.cancel():
            raise Cancelled()

    # ------------------------------------------------------------ top level
    def factor(self, n: int):
        """Return sorted list of prime factors (with repetition)."""
        n = int(n)
        if n < 2:
            return [n] if n else [0]
        out = []
        # trial division by small primes
        for p in _SMALL_PRIMES[:20000]:
            p = int(p)
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    out.append(p)
                    n //= p
        self.check()
        if n > 1:
            self.log(f"testing {len(str(n))}-digit N (primality, then factor search)")
            self._split(n, out)
        out.sort()
        return out

    def _split(self, n: int, out: list):
        self.check()
        if n == 1:
            return
        if is_probable_prime(n):
            out.append(n)
            return
        pp = _perfect_power(n)
        if pp:
            b, k = pp
            sub = []
            self._split(b, sub)
            out.extend(sub * k)
            return
        d = self.find_factor(n)
        self._split(d, out)
        self._split(n // d, out)

    # ------------------------------------------------------------ method ladder
    def find_factor(self, n: int) -> int:
        digits = len(str(n))
        self.log(f"composite, {digits} digits: searching for a factor")
        for name, fn in (("Fermat", self.fermat), ("Pollard rho", self.rho),
                         ("Pollard p-1", self.pm1)):
            self.check()
            d = fn(n)
            if d and 1 < d < n:
                self.log(f"  {name} found a factor ({len(str(d))} digits)")
                return int(d)
        # ECM through the t25 level first, then QS if the number is within its reach
        schedule = [(2_000, 25), (11_000, 90), (50_000, 220), (250_000, 450), (1_000_000, 900),
                    (3_000_000, 2300), (11_000_000, 4500)]
        qs_ok = digits <= 85
        for li, (b1, curves) in enumerate(schedule):
            for c in range(curves):
                self.check()
                self.progress(f"ECM B1={b1:,}  curve {c + 1}/{curves}",
                              (c + 1) / curves)
                d = self.ecm_curve(n, b1)
                if d and 1 < d < n:
                    self.log(f"  ECM (B1={b1:,}, curve {c + 1}) found a {len(str(d))}-digit factor")
                    return int(d)
            self.log(f"  no factor at ECM level B1={b1:,} ({curves} curves)")
            if qs_ok and b1 >= 50_000:
                self.log("  switching to the quadratic sieve")
                d = self.qs(n)
                if d and 1 < d < n:
                    return int(d)
                qs_ok = False
        raise RuntimeError("ran out of methods")

    # ------------------------------------------------------------ Fermat
    def fermat(self, n: int, iters: int = 20000):
        a = _isqrt(_mpz(n))
        if a * a < n:
            a += 1
        b2 = a * a - n
        for _ in range(iters):
            b = _isqrt(b2)
            if b * b == b2:
                return int(a - b)
            a += 1
            b2 += 2 * a - 1
        return None

    # ------------------------------------------------------------ Pollard rho, Brent variant
    def rho(self, n: int, max_iter: int = 400_000):
        n = _mpz(n)
        for attempt in range(3):
            y = _mpz(self.pool.randrange(1, int(n)))
            c = _mpz(self.pool.randrange(1, int(n)))
            m = 128
            g = r = q = _mpz(1)
            x = ys = y
            total = 0
            while g == 1 and total < max_iter:
                self.check()
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = q * abs(x - y) % n
                    g = _gcd(q, n)
                    k += m
                total += r
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = _gcd(abs(x - ys), n)
            if 1 < g < n:
                return int(g)
        return None

    # ------------------------------------------------------------ Pollard p-1
    def pm1(self, n: int, b1: int = 200_000):
        n = _mpz(n)
        a = _mpz(2)
        for p in _SMALL_PRIMES:
            p = int(p)
            if p > b1:
                break
            pe = p
            while pe * p <= b1:
                pe *= p
            a = pow(a, pe, n)
        g = _gcd(a - 1, n)
        return int(g) if 1 < g < n else None

    # ------------------------------------------------------------ ECM
    _stage1_cache = {}
    _prime_cache = {}

    @classmethod
    def _stage1_scalar(cls, b1):
        if b1 not in cls._stage1_cache:
            k = 1
            for p in _primes_upto(b1):
                p = int(p)
                pe = p
                while pe * p <= b1:
                    pe *= p
                k *= pe
            cls._stage1_cache[b1] = k
        return cls._stage1_cache[b1]

    @classmethod
    def _primes_for(cls, b2):
        if b2 not in cls._prime_cache:
            cls._prime_cache.clear()
            cls._prime_cache[b2] = _primes_upto(b2)
        return cls._prime_cache[b2]

    def ecm_curve(self, n: int, b1: int):
        """One Montgomery curve (Suyama parametrisation), stage 1 then stage 2."""
        n = _mpz(n)
        sigma = _mpz(self.pool.randrange(6, 1 << 62))
        u = (sigma * sigma - 5) % n
        v = 4 * sigma % n
        x0 = pow(u, 3, n)
        z0 = pow(v, 3, n)
        denom = 16 * x0 * v % n
        g = _gcd(denom, n)
        if 1 < g < n:
            return int(g)
        if g == n:
            return None
        a24 = pow(v - u, 3, n) * (3 * u + v) % n * _invert(denom, n) % n

        def dbl(x, z):
            s = (x + z) ** 2 % n
            d = (x - z) ** 2 % n
            t = s - d
            return s * d % n, t * (d + a24 * t) % n

        def add(xp, zp, xq, zq, xd, zd):
            uu = (xp - zp) * (xq + zq) % n
            vv = (xp + zp) * (xq - zq) % n
            s = uu + vv
            d = uu - vv
            return zd * (s * s % n) % n, xd * (d * d % n) % n

        def mul(k, x, z):
            """[k](x:z) by the Montgomery ladder."""
            r0 = (x, z)
            r1 = dbl(x, z)
            for bit in bin(k)[3:]:
                if bit == "1":
                    r0 = add(r0[0], r0[1], r1[0], r1[1], x, z)
                    r1 = dbl(r1[0], r1[1])
                else:
                    r1 = add(r0[0], r0[1], r1[0], r1[1], x, z)
                    r0 = dbl(r0[0], r0[1])
            return r0

        # ---- stage 1
        qx, qz = mul(self._stage1_scalar(b1), x0, z0)
        g = _gcd(qz, n)
        if 1 < g < n:
            return int(g)
        if g == n:
            return None

        # ---- stage 2 (baby-step / giant-step over primes in (B1, B2])
        b2 = min(50 * b1, 20_000_000)
        D = 2310
        half = D // 2
        # baby steps [j]Q for 1 <= j <= D/2, kept when gcd(j, D) == 1
        baby = {}
        px, pz = qx, qz                       # [1]Q
        baby[1] = (px, pz)
        p2x, p2z = dbl(qx, qz)                # [2]Q
        prev = (px, pz)
        cur = (p2x, p2z)
        if math.gcd(2, D) == 1:
            baby[2] = cur
        for j in range(3, half + 1):
            nxt = add(cur[0], cur[1], qx, qz, prev[0], prev[1])
            prev, cur = cur, nxt
            if math.gcd(j, D) == 1:
                baby[j] = cur
        # giant steps [mD]Q, each new one = previous + [D]Q with the one before as difference
        dx, dz = mul(D, qx, qz)
        m = max(2, b1 // D)
        r_prev = mul((m - 1) * D, qx, qz)
        rx, rz = mul(m * D, qx, qz)
        primes = self._primes_for(b2)
        pi = int(np.searchsorted(primes, b1, side="right"))
        np_ = len(primes)
        acc = _mpz(1)
        while pi < np_:
            self.check()
            c = m * D
            while pi < np_ and primes[pi] <= c + half:
                p = int(primes[pi])
                j = abs(p - c)
                if p > 11 and j in baby:
                    bx, bz = baby[j]
                    acc = acc * ((rx * bz - bx * rz) % n) % n
                pi += 1
            nx, nz = add(rx, rz, dx, dz, r_prev[0], r_prev[1])
            r_prev, (rx, rz) = (rx, rz), (nx, nz)
            m += 1
        g = _gcd(acc, n)
        if 1 < g < n:
            return int(g)
        return None

    # ------------------------------------------------------------ quadratic sieve
    def qs(self, n: int):
        n = int(n)
        digits = len(str(n))
        ln_n = math.log(n)
        bound = int(math.exp(0.5 * math.sqrt(ln_n * math.log(ln_n))))
        bound = max(bound, 300)
        # factor base: primes p with n a quadratic residue mod p
        fb, roots = [], []
        for p in _primes_upto(bound):
            p = int(p)
            if p == 2:
                fb.append(2)
                roots.append(1)
                continue
            if pow(n % p, (p - 1) // 2, p) == 1:
                fb.append(p)
                roots.append(_tonelli(n % p, p))
        nfb = len(fb)
        need = nfb + 1 + 24                  # +1 for the sign column
        self.log(f"  QS: {digits} digits, factor base {nfb} primes (bound {bound:,}), "
                 f"need {need} relations")
        s = math.isqrt(n) + 1
        c0 = s * s - n                       # Q(x) = x*(x + 2s) + c0
        fb_np = np.array(fb, dtype=np.int64)
        roots_np = np.array(roots, dtype=np.int64)
        s_mod = np.array([s % p for p in fb], dtype=np.int64)
        r1 = np.array([(r - s) % p for r, p in zip(roots, fb)], dtype=np.int64)       # x = r - s
        r2 = np.array([(-r - s) % p for r, p in zip(roots, fb)], dtype=np.int64)
        logp = np.log2(fb_np.astype(np.float64)).astype(np.float32)
        sieve_from = 6                                    # skip the tiniest primes while sieving
        BS = 1 << 16 if digits < 45 else 1 << 17
        slack = 1.4 * math.log2(fb[-1]) + 6
        piv = {}                       # high bit -> (vector, history)
        rels = []                      # (s + x, {prime_index: exponent}, sign)
        block = 0
        found = None
        s_f = float(s)
        while found is None:
            for sign_blk in (1, -1):
                self.check()
                start = block * BS if sign_blk == 1 else -(block + 1) * BS
                xs = np.arange(start, start + BS, dtype=np.int64)
                xf = xs.astype(np.float64)
                qf = np.abs(xf * (xf + 2 * s_f) + float(c0))
                qf[qf < 1] = 1
                target = np.log2(qf).astype(np.float32) - slack
                acc = np.zeros(BS, dtype=np.float32)
                for i in range(sieve_from, nfb):
                    p = fb[i]
                    lg = logp[i]
                    for r in ((int(r1[i]), int(r2[i])) if r1[i] != r2[i] else (int(r1[i]),)):
                        off = (r - start) % p
                        acc[off::p] += lg
                cand = np.nonzero(acc >= target)[0]
                for ci in cand:
                    x = int(start + ci)
                    val = x * (x + 2 * s) + c0
                    sign = 1 if val < 0 else 0
                    v = abs(val)
                    if v == 0:
                        continue
                    # which FB primes divide this value: x + s == +-root (mod p)
                    xm = (x + s_mod) % fb_np
                    hit = np.nonzero((xm == roots_np) | (xm == (fb_np - roots_np) % fb_np))[0]
                    expo = {}
                    for i in hit:
                        p = fb[int(i)]
                        e = 0
                        while v % p == 0:
                            v //= p
                            e += 1
                        if e:
                            expo[int(i)] = e
                    if v != 1:
                        continue
                    vec = sign
                    for i, e in expo.items():
                        if e & 1:
                            vec |= 1 << (i + 1)
                    rels.append((x + s, expo, sign))
                    idx = len(rels) - 1
                    hist = 1 << idx
                    while vec:
                        hb = vec.bit_length() - 1
                        if hb in piv:
                            pv, ph = piv[hb]
                            vec ^= pv
                            hist ^= ph
                        else:
                            piv[hb] = (vec, hist)
                            break
                    if vec == 0:
                        d = self._qs_try(n, rels, hist, fb)
                        if d:
                            return d
                self.progress(f"QS  {len(rels)}/{need} relations", min(1.0, len(rels) / need))
            block += 1
        return None

    def _qs_try(self, n, rels, hist, fb):
        x = 1
        tot = {}
        sign_count = 0
        i = 0
        while hist:
            if hist & 1:
                xv, expo, sg = rels[i]
                x = x * xv % n
                sign_count += sg
                for k, e in expo.items():
                    tot[k] = tot.get(k, 0) + e
            hist >>= 1
            i += 1
        y = 1
        for k, e in tot.items():
            y = y * pow(fb[k], e // 2, n) % n
        g = math.gcd(x - y, n)
        return g if 1 < g < n else None


def _tonelli(a: int, p: int) -> int:
    """Square root of a mod an odd prime p (a is a quadratic residue)."""
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, b * b % p, t * b * b % p, r * b % p
    return r


# ------------------------------------------------------------------ hidden-number guesser
HIDDEN_RANGE_DEFAULT = 100
HIDDEN_RANGE_MAX = 1_000_000
GUESS_HZ_MIN, GUESS_HZ_MAX = 20.0, 2000.0     # the clock wanders between these
GUESS_RETUNE_S = 0.3                          # how often the clock re-reads the frequency setting
GUESS_HZ_DEFAULT = 17.0                       # starting clock frequency (adjustable in the GUI)
GUESS_HZ_LIMITS = (1.0, 2000.0)               # allowed range for the frequency box


def prob_at_least(hits: int, n: int, p: float) -> float:
    """P(X >= hits) for X ~ Binomial(n, p): the chance of being at least this lucky."""
    if hits <= 0:
        return 1.0
    if hits > n:
        return 0.0
    s = 0.0
    for i in range(hits):
        s += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                      + i * math.log(p) + (n - i) * math.log(1 - p))
    return min(1.0, max(0.0, 1.0 - s))


class HiddenNumber:
    """A secret integer in [0, size) and a separate thread that clock-cycles a candidate
    through the same range at a randomly varying frequency.

    The thread only counts. It never compares against the secret. The comparison happens
    in spot_check(), which the frame loop calls when an even state arrives: it reads the
    candidate AT THAT INSTANT (no scanning, no waiting) and reports whether it equals the
    secret. So a hit is pure timing luck, with probability about 1/size per check."""

    def __init__(self, size=HIDDEN_RANGE_DEFAULT, rng=None):
        self._rng = rng or np.random.default_rng()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.size = 1
        self.candidate = 0
        self.hz = GUESS_HZ_DEFAULT
        self.target_hz = GUESS_HZ_DEFAULT
        self._wake = threading.Event()      # interrupts the clock's wait on a change
        self.ticks = 0
        self.checks = 0
        self.hits = 0
        self.rounds = 0                     # how many secrets have been guessed so far
        self.last_check = None              # (candidate, hit)
        self.last_found = None              # (number, checks it took) of the latest hit
        self._secret = 0
        self.revealed = False
        self.set_range(size)
        self._thread = threading.Thread(target=self._run, daemon=True)

    # ---- control
    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()

    def set_hz(self, hz: float):
        """Change the clock frequency (takes effect immediately)."""
        lo, hi = GUESS_HZ_LIMITS
        hz = max(lo, min(hi, float(hz)))
        with self._lock:
            self.target_hz = hz
        self._wake.set()

    def _reset_counts_locked(self):
        """Zero the statistics (caller holds the lock)."""
        self.checks = 0
        self.hits = 0
        self.rounds = 0
        self.last_check = None
        self.last_found = None

    def set_range(self, size: int):
        """Change the range, draw a fresh secret and restart the counts."""
        size = max(2, min(HIDDEN_RANGE_MAX, int(size)))
        with self._lock:
            self.size = size
            self.candidate %= size
            self._secret = int(self._rng.integers(0, size))
            self.revealed = False
            self._reset_counts_locked()

    def new_secret(self):
        """Draw a fresh secret and restart the counts (and the luck factor)."""
        with self._lock:
            self._secret = int(self._rng.integers(0, self.size))
            self.revealed = False
            self._reset_counts_locked()

    def reveal(self):
        """Show the secret to the user (never used by the clock thread)."""
        with self._lock:
            self.revealed = True
            return self._secret

    # ---- the separate clock thread: cycles the candidate, frequency varies
    def _run(self):
        hz = self.target_hz
        period = 1.0 / hz
        nxt = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                new_hz = self.target_hz
            if new_hz != hz:                       # frequency was changed from the GUI
                hz = new_hz
                period = 1.0 / hz
                nxt = time.perf_counter()
            with self._lock:
                self.hz = hz
                self.candidate = (self.candidate + 1) % self.size
                self.ticks += 1
            nxt += period
            delay = nxt - time.perf_counter()
            if delay > 0:
                self._wake.wait(delay)             # wakes early on stop or frequency change
                self._wake.clear()
            elif delay < -0.05:                    # fell behind: do not burst to catch up
                nxt = time.perf_counter()

    # ---- the spot check, called when an even-state frame arrives
    def spot_check(self) -> bool:
        with self._lock:
            cand = self.candidate                  # the value on the clock right now
            hit = cand == self._secret
            self.checks += 1
            self.last_check = (cand, hit)
            if hit:
                self.hits += 1
                self.rounds += 1
                self.last_found = (cand, self.checks)   # the correct number, as hidden
            # hit or miss, the number is thrown away and a new one is hidden after every check
            self._secret = int(self._rng.integers(0, self.size))
            self.revealed = False
            return hit

    def snapshot(self) -> dict:
        with self._lock:
            p = 1.0 / self.size
            expected = self.checks * p
            luck = (self.hits / expected * 100.0) if expected > 0 else None
            tail = prob_at_least(self.hits, self.checks, p) if self.checks else None
            return dict(size=self.size, candidate=self.candidate, hz=self.hz,
                        ticks=self.ticks, checks=self.checks, hits=self.hits,
                        rounds=self.rounds, last_check=self.last_check,
                        last_found=self.last_found, expected=expected, luck=luck, luck_tail=tail,
                        secret=self._secret if self.revealed else None)


# ------------------------------------------------------------------ higher/lower (binary) search
SEARCH_BITS_DEFAULT = 1000
SEARCH_BITS_MAX = 4096


def short_int(n: int, keep: int = 20) -> str:
    """Hex form of a big integer, shortened to its first and last digits."""
    h = format(n, "x")
    return h if len(h) <= 2 * keep + 3 else f"{h[:keep]}...{h[-keep:]} ({len(h) * 4} bit hex)"


class BinarySearch:
    """A secret integer of `bits` bits and a higher/lower oracle, with POOLED bits.

    Each guess() is one oracle query and yields exactly one bit of the secret
    (most significant bit first): "is secret >= pooled-so-far with this bit set?".
    The bit is added to a pool and NO equality check is made on that query. Only
    when all `bits` bits have been pooled is the pooled candidate compared with the
    secret, once. A secret of n bits therefore costs n pooled queries plus one
    final check, instead of an independent equality check after every guess.
    When the final check passes one time-code bin is earned and a new secret is
    hidden."""

    def __init__(self, bits=SEARCH_BITS_DEFAULT):
        self._lock = threading.Lock()
        self.bits = 2
        self.guesses = 0                    # all oracle queries since the last reset
        self.final_checks = 0               # equality checks made (one per full pool)
        self.solves = 0                     # secrets found since the last reset
        self.solved_guesses = 0             # queries spent on the solved secrets
        self.round_guesses = 0              # queries spent on the current secret
        self.last = None                    # (verdict, guess_bits)
        self.last_found = None              # (number, guesses it took) of the latest find
        self.revealed = False
        self.set_bits(bits)

    def _new_round_locked(self):
        self._secret = secrets.randbelow(1 << self.bits)
        self.pooled = 0                     # bits collected so far, MSB first
        self.pooled_n = 0                   # how many bits are in the pool
        self.round_guesses = 0
        self.revealed = False

    def _reset_counts_locked(self):
        self.guesses = self.solves = self.solved_guesses = self.final_checks = 0
        self.last = None
        self.last_found = None

    def set_bits(self, bits: int):
        """Change the size of the secret, draw a new one and restart the counts."""
        with self._lock:
            self.bits = max(2, min(SEARCH_BITS_MAX, int(bits)))
            self._new_round_locked()
            self._reset_counts_locked()

    def new_secret(self):
        with self._lock:
            self._new_round_locked()
            self._reset_counts_locked()

    def reveal(self):
        with self._lock:
            self.revealed = True
            return self._secret

    def guess(self) -> bool:
        """One oracle query. Pools one bit; only the query that completes the pool
        runs the single final equality check. True when that check finds the secret."""
        with self._lock:
            self.guesses += 1
            self.round_guesses += 1
            shift = self.bits - 1 - self.pooled_n
            trial = self.pooled | (1 << shift)
            bit = 1 if self._secret >= trial else 0      # oracle: higher/lower, no equality
            self.pooled |= bit << shift
            self.pooled_n += 1
            if self.pooled_n < self.bits:
                self.last = (f"pooled bit {self.pooled_n}/{self.bits} = {bit}", self.pooled_n)
                return False

            # pool is full: the ONE final check
            self.final_checks += 1
            if self.pooled == self._secret:
                self.solves += 1
                self.solved_guesses += self.round_guesses
                self.last_found = (self.pooled, self.round_guesses)
                self.last = ("FOUND (final check passed)", self.bits)
                self._new_round_locked()
                return True
            self.last = ("final check failed, pool cleared", self.bits)
            self.pooled = 0
            self.pooled_n = 0
            return False

    def snapshot(self) -> dict:
        with self._lock:
            return dict(bits=self.bits, guesses=self.guesses, solves=self.solves,
                        round_guesses=self.round_guesses,
                        pooled_n=self.pooled_n, final_checks=self.final_checks,
                        interval_bits=self.bits - self.pooled_n,
                        avg=(self.solved_guesses / self.solves if self.solves else None),
                        last=self.last, last_found=self.last_found,
                        secret=short_int(self._secret) if self.revealed else None)



# ------------------------------------------------------------------ factoring by higher/lower search
FACTOR_BITS_DEFAULT = 256
FACTOR_BITS_MAX = 4096


def _engine_proc(n: int, seed: bytes, q):
    """Runs in a CHILD PROCESS: big-int pow() never releases the GIL, so doing this work in a
    thread froze the GUI and the camera loop. Results come back through the queue."""
    try:
        pool = EntropyPool()
        pool.feed(seed)
        fz = Factoriser(pool, lambda: False,
                        lambda s: q.put(("log", s)), lambda t, f: q.put(("prog", t, f)))
        fs = [int(f) for f in fz.factor(n)]
        ok = math.prod(fs) == n and all(is_probable_prime(f) for f in fs)
        q.put(("done", fs, ok))
    except BaseException as e:                                   # report, never hang the parent
        q.put(("err", f"{type(e).__name__}: {e}"))


def short_dec(n: int, keep: int = 14) -> str:
    s = str(n)
    return s if len(s) <= 2 * keep + 3 else f"{s[:keep]}...{s[-keep:]} ({len(s)} digits)"


def parse_number(text: str) -> int:
    """Plain integer (spaces, commas, underscores allowed) or a form like 2^127-1."""
    import re
    t = re.sub(r"[\s,_]", "", text).lstrip("+")
    if re.fullmatch(r"0[xX][0-9a-fA-F]+", t):
        return int(t, 16)
    t = re.sub(r"[lLnN]$", "", t) if re.fullmatch(r"\d+[lLnN]", t) else t
    m = re.fullmatch(r"(\d+)\^(\d+)([+-]\d+)?", t)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        if a < 2 or b * math.log2(a) > 20000:
            raise ValueError("power too large (limit about 20000 bits)")
        return a ** b + c
    if t.isdigit():
        return int(t)
    raise ValueError("type digits, or a form like 2^127-1")


class FactorSearch:
    """Factoring with the higher/lower bit search, and a time code ONLY for a verified factorisation.

    N is the number to factor. The oracle holds a prime factor p of N (the smallest one) and answers
    only "is p >= this trial value?". Each armed frame (two sequential evens) is one query and pools
    one bit of p, most significant bit first. No equality test is made per query. When all bits are
    pooled, ONE final check runs: 1 < p < N, N % p == 0, and the full prime factorisation multiplies
    back to N with every factor a probable prime. Only then is a time-code bin earned.

    Where the oracle gets p:
      * typed N : a background thread really factors N (trial division, Fermat, Pollard rho, p-1,
                  ECM, quadratic sieve). Until it finishes the oracle is not ready, no bits are
                  pooled and no time code can be earned.
      * generated N : a random semiprime is built and its factors are kept inside the oracle. This
                  shows the mechanism; the search does not factor N, the oracle already knows p.
    A higher/lower oracle for a factor of N cannot exist without already knowing that factor, so the
    pooling is a mechanism, not a shortcut: the cost of factoring is paid by the engine thread."""

    def __init__(self, pool):
        self._lock = threading.Lock()
        self.pool = pool
        self._run = 0
        self.gen_bits = FACTOR_BITS_DEFAULT
        self.state = "idle"             # idle generating factoring ready solved prime failed cancelled
        self.mode = ""
        self.n = 0
        self.width = 0
        self._secret = 0
        self.factors = []
        self.verified = False
        self.highlow = False            # True: typed N is searched directly, no engine, no factor checks
        self.progress = ""
        self.msgs = []
        self.revealed = False
        self.pooled = self.pooled_n = 0
        self.round_guesses = 0
        self.guesses = self.final_checks = self.solves = 0
        self.last = None
        self.last_found = None          # (p, q, queries)
        self.history = []               # every number the engine finished / every verified solve
        self.t_start = 0.0
        self.engine_s = None

    # ---------------------------------------------------------- bookkeeping
    def _say(self, run, s):
        with self._lock:
            if run == self._run:
                self.msgs.append(s)

    def _prog(self, run, text, frac):
        with self._lock:
            if run == self._run:
                self.progress = f"{text}  {frac:.0%}"

    def drain(self):
        with self._lock:
            m, self.msgs = self.msgs, []
            return m

    def _begin(self, mode, state, n=0, keep_counts=False):
        with self._lock:
            self._run += 1
            self.mode, self.state, self.n = mode, state, n
            self.width = max(1, math.isqrt(n).bit_length()) if n else 0
            self._secret, self.factors, self.verified, self.highlow = 0, [], False, False
            self.pooled = self.pooled_n = self.round_guesses = 0
            self.progress, self.revealed, self.engine_s = "", False, None
            self.t_start = time.time()
            if not keep_counts:
                self.guesses = self.final_checks = self.solves = 0
                self.last = self.last_found = None
            return self._run

    def _record(self, mode, n, factors, status, **extra):
        """Caller holds self._lock. Big ints are stored as decimal strings (JSON-safe)."""
        self.history.append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), mode=mode, status=status,
                                 n=str(n), bits=int(n).bit_length(),
                                 factors=[str(f) for f in factors],
                                 **{k: (str(v) if isinstance(v, int) and v > 2**53 else v)
                                    for k, v in extra.items()}))

    def export(self) -> dict:
        with self._lock:
            cur = dict(state=self.state, mode=self.mode, n=str(self.n) if self.n else None,
                       factors=[str(f) for f in self.factors],
                       secret=str(self._secret) if self.revealed and self._secret else None)
            return dict(saved=time.strftime("%Y-%m-%d %H:%M:%S"), current=cur,
                        verified_solves=self.solves, queries=self.guesses,
                        history=list(self.history))

    def cancel(self):
        with self._lock:
            self._run += 1
            if self.state in ("generating", "factoring"):
                self.state = "cancelled"
                self.progress = ""
                self.msgs.append("# factoriser: cancelled")

    def reveal(self):
        with self._lock:
            self.revealed = True
            return self._secret

    # ---------------------------------------------------------- typed number
    def search_number(self, n: int):
        """High/low only: no primality test, no factoring engine, no factor verification.
        The oracle holds the typed number itself and answers only "is it >= trial?". Each armed frame
        pools one bit (most significant first); after the last bit ONE equality check runs."""
        n = int(n)
        with self._lock:
            self._run += 1
            self.mode, self.n, self.state, self.highlow = "typed (high/low only)", n, "ready", True
            self.width = max(1, n.bit_length())
            self._secret, self.factors, self.verified = n, [n], True
            self.pooled = self.pooled_n = self.round_guesses = 0
            self.guesses = self.final_checks = self.solves = 0
            self.progress, self.revealed, self.engine_s = "", False, None
            self.last = self.last_found = None
            self.msgs.append(f"# high/low only: N = {short_dec(n)} ({n.bit_length()} bit), no checks; "
                             f"pooling {self.width} bits on armed frames")

    def factor_number(self, n: int):
        n = int(n)
        run = self._begin("typed", "factoring", n)
        if n < 4:
            with self._lock:
                self.state = "failed"
                self.progress = "N must be at least 4"
            return
        self._say(run, f"# factoriser: N = {short_dec(n)} ({n.bit_length()} bit), "
                       f"{'gmpy2' if HAVE_GMP else 'plain Python ints'}")
        threading.Thread(target=self._work_typed, args=(run, n), daemon=True).start()

    def _work_typed(self, run, n):
        t0 = time.time()
        q = mp.Queue()
        proc = mp.Process(target=_engine_proc, args=(n, self.pool.rand_bits(256).to_bytes(32, "big"), q),
                          daemon=True)
        proc.start()
        fs = ok = None
        try:
            while True:
                if self._run != run:                              # cancelled or a new N was entered
                    return
                try:
                    msg = q.get(timeout=0.1)
                except queue.Empty:
                    if not proc.is_alive():
                        try:
                            msg = q.get(timeout=0.2)              # result may still be in flight
                        except queue.Empty:
                            msg = ("err", "engine process stopped unexpectedly")
                    else:
                        continue
                if msg[0] == "log":
                    self._say(run, msg[1])
                elif msg[0] == "prog":
                    self._prog(run, msg[1], msg[2])
                elif msg[0] == "err":
                    with self._lock:
                        if run == self._run:
                            self.state, self.progress = "failed", f"engine error: {msg[1]}"
                    return
                else:
                    fs, ok = msg[1], msg[2]
                    break
        finally:
            if proc.is_alive():
                proc.terminate()
            q.close()
        with self._lock:
            if run != self._run:
                return
            self.engine_s = time.time() - t0
            self.progress = ""
            if not ok:
                self.state, self.progress = "failed", "factorisation did not verify"
            elif len(fs) == 1:
                self.state = "prime"
                self._record("typed", n, fs, "prime")
                self.msgs.append(f"# factoriser: N = {short_dec(n)} is (probably) prime ({n.bit_length()} bit). "
                                 "A prime has no factor p with 1 < p < N, so there is no oracle and no "
                                 "time code to earn. Type a composite N, or press Generate.")
            else:
                self.factors, self._secret, self.verified = fs, fs[0], True
                self.state = "ready"
                self._record("typed", n, fs, "factored")
                self.msgs.append("# factoriser: engine finished in "
                                 f"{self.engine_s:.1f} s, oracle ready ({len(fs)} prime factors); "
                                 "pooling bits on armed frames")

    # ---------------------------------------------------------- generated semiprime
    def generate(self, bits: int, keep_counts=False):
        bits = max(16, min(FACTOR_BITS_MAX, int(bits)))
        self.gen_bits = bits
        run = self._begin("generated", "generating", 0, keep_counts)
        threading.Thread(target=self._work_gen, args=(run, bits), daemon=True).start()

    def _work_gen(self, run, bits):
        def prime(b):
            while True:
                if self._run != run:
                    raise Cancelled()
                c = self.pool.rand_bits(b) | (1 << (b - 1)) | 1
                if is_probable_prime(c):
                    return c
        try:
            p = prime(bits // 2)
            q = prime(bits - bits // 2)
            while q == p:
                q = prime(bits - bits // 2)
        except Cancelled:
            return
        n, fs = p * q, sorted((p, q))
        with self._lock:
            if run != self._run:
                return
            self.n, self.width = n, max(1, math.isqrt(n).bit_length())
            self.factors, self._secret, self.verified = fs, fs[0], True
            self.state = "ready"
            self.msgs.append(f"# factoriser: new semiprime N = {short_dec(n)} ({n.bit_length()} bit); "
                             "its factors are held by the oracle")

    # ---------------------------------------------------------- one armed frame = one oracle query
    def guess(self) -> bool:
        """Pools one bit. Only the query that completes the pool runs the single final check.
        True only when that check verifies a correct factorisation."""
        regen = False
        with self._lock:
            if self.state != "ready":
                return False
            self.guesses += 1
            self.round_guesses += 1
            shift = self.width - 1 - self.pooled_n
            trial = self.pooled | (1 << shift)
            bit = 1 if self._secret >= trial else 0            # higher/lower, no equality
            self.pooled |= bit << shift
            self.pooled_n += 1
            if self.pooled_n < self.width:
                self.last = f"pooled bit {self.pooled_n}/{self.width} = {bit}"
                return False
            self.final_checks += 1
            c = self.pooled
            if self.highlow:                                   # one equality check, nothing else
                ok = (c == self._secret)
            else:
                ok = (1 < c < self.n and self.n % c == 0 and self.verified
                      and math.prod(self.factors) == self.n)
            if not ok:
                self.last = "final check failed, pool cleared"
                self.pooled = self.pooled_n = 0
                return False
            self.solves += 1
            self.last_found = (c, 1 if self.highlow else self.n // c, self.round_guesses)
            self._record(self.mode, self.n, self.factors, "solved", p=c, queries=self.round_guesses)
            self.last = ("FOUND: pooled bits equal the number" if self.highlow
                         else "FOUND: N % p == 0 and the factorisation multiplies back to N")
            self.state = "solved"
            regen = self.mode == "generated"
        if regen:
            self.generate(self.gen_bits, keep_counts=True)      # next semiprime, same size
        return True

    def snapshot(self) -> dict:
        with self._lock:
            return dict(state=self.state, mode=self.mode, n=self.n, width=self.width,
                        pooled_n=self.pooled_n, guesses=self.guesses, solves=self.solves,
                        final_checks=self.final_checks, round_guesses=self.round_guesses,
                        progress=self.progress, last=self.last, last_found=self.last_found,
                        engine_s=self.engine_s, factors=list(self.factors) if self.state == "solved" else [],
                        secret=short_dec(self._secret) if self.revealed and self._secret else None)


# ------------------------------------------------------------------ time-code maths
def carries(a: int, b: int, w: int) -> int:
    """Number of carries (including the carry out of the top) when adding a + b."""
    c = n = 0
    for i in range(w):
        t = ((a >> i) & 1) + ((b >> i) & 1) + c
        c = t >> 1
        n += c
    return n


def parse_code(expr: str):
    """Evaluate a time-code expression: '1011', '1000|0011', '1111&1011', '1011+0001'.

    Operands are bit strings read as binary numbers (written order = bin order).
    |  OR-addition   &  AND   +  integer addition (wraps at the code width).
    Returns (Y, info). Raises ValueError on a malformed expression.
    """
    e = expr.replace(" ", "")
    toks, cur = [], ""
    for ch in e:
        if ch in "|&+":
            toks += [cur, ch]
            cur = ""
        else:
            cur += ch
    toks.append(cur)
    operands, ops = toks[0::2], toks[1::2]
    for o in operands:
        if not o or set(o) - {"0", "1"}:
            raise ValueError(f"bad operand '{o}'")
    W = max(len(o) for o in operands)
    mask = (1 << W) - 1
    val, car, wrapped = int(operands[0], 2), 0, 0
    for op, o in zip(ops, operands[1:]):
        b = int(o, 2)
        if op == "|":
            val |= b
        elif op == "&":
            val &= b
        else:
            car += carries(val, b, W)
            wrapped += int(val + b > mask)
            val = (val + b) & mask
    return format(val, f"0{W}b"), dict(carries=car, wrapped=wrapped, expr=e)


def open_in_window(Y: str, k: int, start: int) -> int:
    return sum(int(Y[(start + t) % len(Y)]) for t in range(k))


def window_stats(Y: str, k: int):
    """(worst, best) number of open bins over every k-frame window of the periodic code."""
    counts = [open_in_window(Y, k, s) for s in range(len(Y))]
    return min(counts), max(counts)


def frames_needed(Y: str, n_open: int):
    """Smallest k whose worst-phase window holds n_open open bins (None if Y is all 0)."""
    if "1" not in Y:
        return None
    k = n_open
    while window_stats(Y, k)[0] < n_open:
        k += 1
    return k


# ------------------------------------------------------------------ keystroke injection
# Windows: SendInput through ctypes (no extra package). Elsewhere: pip install pynput.
import ctypes

_KEYEVENTF_KEYUP, _KEYEVENTF_SCANCODE = 0x0002, 0x0008


class _KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_uint16), ("wScan", ctypes.c_uint16), ("dwFlags", ctypes.c_uint32),
                ("time", ctypes.c_uint32), ("dwExtraInfo", ctypes.c_size_t)]


class _MouseInput(ctypes.Structure):               # only here so the union has the real size
    _fields_ = [("dx", ctypes.c_int32), ("dy", ctypes.c_int32), ("mouseData", ctypes.c_uint32),
                ("dwFlags", ctypes.c_uint32), ("time", ctypes.c_uint32), ("dwExtraInfo", ctypes.c_size_t)]


class _HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_uint32), ("wParamL", ctypes.c_uint16), ("wParamH", ctypes.c_uint16)]


class _InputUnion(ctypes.Union):
    _fields_ = [("ki", _KeyBdInput), ("mi", _MouseInput), ("hi", _HardwareInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("ii", _InputUnion)]


def injection_backend_error():
    """None when keystrokes can be injected on this machine, else a message saying why not."""
    if sys.platform == "win32":
        return None
    try:
        import pynput.keyboard                                # noqa: F401
        return None
    except Exception as exc:
        return ("Keystroke injection needs Windows, or 'pip install pynput' on other systems "
                f"(macOS also needs Accessibility permission). Details: {exc}")


def _win_key(vk: int, up: bool, scan: bool):
    user32 = ctypes.windll.user32
    flags = _KEYEVENTF_KEYUP if up else 0
    if scan:                                                  # hardware scan codes: games, DirectInput
        ki = _KeyBdInput(0, user32.MapVirtualKeyW(vk, 0), flags | _KEYEVENTF_SCANCODE, 0, 0)
    else:
        ki = _KeyBdInput(vk, 0, flags, 0, 0)
    inp = _Input(1, _InputUnion(ki=ki))
    if user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp)) != 1:
        raise OSError("SendInput was refused (an administrator window may have focus)")


def inject_combo(key: str, shift: bool = True, hold: float = 0.04, scan: bool = True):
    """Press Shift+key for real, system-wide: it goes to whichever window has keyboard focus."""
    key = key.upper()
    if sys.platform == "win32":
        vk = ord(key)                                         # A-Z and 0-9 equal their virtual-key codes
        if shift:
            _win_key(0x10, False, scan)
            time.sleep(0.01)
        _win_key(vk, False, scan)
        time.sleep(hold)
        _win_key(vk, True, scan)
        if shift:
            time.sleep(0.01)
            _win_key(0x10, True, scan)
        return
    from pynput.keyboard import Controller, Key
    kb = Controller()
    if shift:
        kb.press(Key.shift)
        time.sleep(0.01)
    kb.press(key.lower())
    time.sleep(hold)
    kb.release(key.lower())
    if shift:
        time.sleep(0.01)
        kb.release(Key.shift)


# ------------------------------------------------------------------ tunable variables
# key: (label, default, low, high, step, is_int, tooltip-ish note)
TUNE_DEFAULTS = dict(
    bin_w=1, phase=0, clock="frames", bin_ms=100.0, invert=False, bins_per_hit=1,
    rc_ms=RC_B * 1000, tau_ms=TAU_NODE * 1000, vs=VS, vbe=V_BE, vih=V_IH,
    node_thr=0.5, warmup=WARMUP, blur=5,
    x_div=1, x_off=0, diff_gap=1, dilate=0, a_rounds=1, b_min_x=2, b_odd=True, gate="AND",
    max_fps=0, dt_max_ms=250, pending_cap=0, cam_res="640x480",
    q_flip_prob=0.0, q_flip_mode="random", q_flip_period=10,
    q_ecc="none", q_latch="D", q_reset=False, q_invert=False,
    q_seed=0, q_burst_len=3)


def gate_out(mode: str, st) -> int:
    """Gate stage. AND = the original tied OR/AND outputs."""
    if mode == "OR":
        return st.orv
    if mode == "A only":
        return st.A
    if mode == "B only":
        return st.B
    return st.orv & st.andv


def y_bin_at(ctl: dict, idx: int, elapsed: float):
    """Time-code bin for this frame -> (bit, bin_index).
    clock 'frames': the bin advances every bin_w camera frames.
    clock 'ms'    : the bin advances every bin_ms milliseconds of wall-clock time.
    phase rotates the code, invert flips every bit."""
    y = ctl["y"]
    if ctl.get("clock", "frames") == "ms":
        n = int(elapsed * 1000.0 // max(1.0, float(ctl.get("bin_ms", 100.0))))
    else:
        n = idx // max(1, int(ctl.get("bin_w", 1)))
    b = (n + int(ctl.get("phase", 0))) % len(y)
    bit = int(y[b])
    if ctl.get("invert"):
        bit ^= 1
    return bit, b


def effective_code(ctl: dict) -> str:
    """The code as the circuit sees it frame by frame (one char per frame, periodic).
    Used by the error-bound analysis. In 'ms' mode the bin width in frames is estimated
    from the measured frame rate."""
    y = ctl["y"]
    if ctl.get("clock", "frames") == "ms":
        fps = ctl.get("fps_est") or 30.0
        w = max(1, round(float(ctl.get("bin_ms", 100.0)) / 1000.0 * fps))
    else:
        w = max(1, int(ctl.get("bin_w", 1)))
    ph = int(ctl.get("phase", 0)) % len(y)
    rot = y[ph:] + y[:ph]
    if ctl.get("invert"):
        rot = "".join("1" if c == "0" else "0" for c in rot)
    return "".join(c * w for c in rot)


# ------------------------------------------------------------------ driver analysis
_WD: dict = {}


def witness_density(x: int) -> float:
    """Exact fraction of r in [2, x-2] that are Miller-Rabin witnesses for x
    (0 for primes, even numbers and x < 5). This is Driver A's true success
    probability on instance x. Rabin: >= 3/4 for every odd composite."""
    if x not in _WD:
        if x < 5 or x % 2 == 0:
            _WD[x] = 0.0
        else:
            _WD[x] = sum(miller_rabin_witness(a, x) for a in range(2, x - 1)) / (x - 3)
    return _WD[x]


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mutual_info(cm: dict) -> float:
    """I(driver output ; x in L) in bits from a 2x2 confusion matrix."""
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    N = tp + fp + fn + tn
    if N == 0:
        return 0.0
    pl = (tp + fn) / N
    pd = (tp + fp) / N
    mi = 0.0
    for cnt, d, l in ((tp, pd, pl), (fp, pd, 1 - pl), (fn, 1 - pd, pl), (tn, 1 - pd, 1 - pl)):
        if cnt and d > 0 and l > 0:
            mi += cnt / N * math.log2((cnt / N) / (d * l))
    return mi


def _div(a, b):
    return a / b if b else None


def _f(v, w=5):
    return " " * (w - 1) + "-" if v is None else f"{v:{w}.2f}"


class DriverAnalysis:
    """Predictive properties of Driver A, Driver B and the final Q, scored on
    every instance frame against the ground truth x in L."""

    NAMES = (("A", "Driver A (random)"), ("B", "Driver B (determ.)"), ("Q", "Q  (A.B.Y latch)"))

    def __init__(self):
        self.reset()

    def reset(self):
        self.cm = {n: dict(tp=0, fp=0, fn=0, tn=0) for n, _ in self.NAMES}
        self.n_noinst = 0
        self.link_miss = 0                  # c(x)=1 but B stayed LOW (timing)
        self.q_checked = self.q_mismatch = 0
        self.a_seq, self.w_seq, self.r_seq = [], [], []     # in-L frames, in order
        self.min_w = 1.0

    def update(self, rec: dict, w: float):
        A, B, Q, y = rec["A"], rec["B"], rec["Q"], rec["y_bin"]
        pred = ((A & B) | y) if rec["force"] else (A & B & y)    # what Q should be
        self.q_checked += 1
        self.q_mismatch += int(pred != Q)
        if not rec["motion"]:
            self.n_noinst += 1
            return
        pos = rec["in_L"]
        for name, val in (("A", A), ("B", B), ("Q", Q)):
            key = ("tp" if val else "fn") if pos else ("fp" if val else "tn")
            self.cm[name][key] += 1
        if rec["c_of_x"] and not B:
            self.link_miss += 1
        if pos:
            self.a_seq.append(A)
            self.w_seq.append(w)
            self.r_seq.append(rec["r"])
            self.min_w = min(self.min_w, w)
            if len(self.a_seq) > 4000:
                del self.a_seq[:1000], self.w_seq[:1000], self.r_seq[:1000]

    def lines(self, k: int, force: bool = False):
        a = self.cm["A"]
        n_in = a["tp"] + a["fn"]
        n_out = a["fp"] + a["tn"]
        n = n_in + n_out
        L = [f"DRIVER ANALYSIS   instances n={n}  (x in L: {n_in}, not in L: {n_out});"
             f" no-motion frames: {self.n_noinst}"]
        if n == 0:
            return L + ["waiting for motion ..."]
        base = n_in / n
        H = -sum(q * math.log2(q) for q in (base, 1 - base) if q > 0)
        L.append(f"truth: P(x in L)={base:.2f}   H(L)={H:.2f} bit   (L = odd composite)")
        L.append("")
        L.append(f"{'':20s}{'TPR':>6}{'FPR':>6}{'PPV':>6}{'NPV':>6}{'acc':>6}{'I(D;L)':>9}")
        for name, label in self.NAMES:
            c = self.cm[name]
            tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
            L.append(f"{label:20s}{_f(_div(tp, tp + fn), 6)}{_f(_div(fp, fp + tn), 6)}"
                     f"{_f(_div(tp, tp + fp), 6)}{_f(_div(tn, tn + fn), 6)}"
                     f"{_f(_div(tp + tn, n), 6)}{mutual_info(c):7.2f} b")
        L.append("")
        L.append("RP properties the bounds rely on")
        fa = a["fp"]
        lo, hi = wilson(fa, n_out)
        L.append(f" A fires only inside L (FP=0): {'PASS' if fa == 0 else 'FAIL'}   "
                 f"{fa}/{n_out}, FPR <= {hi:.1%} (95%)")
        lo, hi = wilson(a["tp"], n_in)
        verdict = "PASS" if lo >= 0.5 else ("FAIL" if hi < 0.5 else "inconclusive")
        L.append(f" A witness rate p >= 1/2     : {verdict}   p={_div(a['tp'], n_in) or 0:.3f}"
                 f"  [{lo:.3f}, {hi:.3f}]")
        b = self.cm["B"]
        L.append(f" B never misses L (FN=0)     : {'PASS' if b['fn'] == 0 else 'FAIL'}   "
                 f"{b['fn']}/{n_in}   link timing misses: {self.link_miss}")
        q = self.cm["Q"]
        L.append(f" Q one-sided (FP=0)          : {'PASS' if q['fp'] == 0 else 'FAIL'}   "
                 f"{q['fp']}/{n_out}" + ("   (force mode breaks it, Thm 4)" if force else ""))
        L.append("")
        L.append("Calibration of A (exact witness density w(x) of each member x)")
        m = len(self.a_seq)
        if m >= 10:
            obs = sum(self.a_seq)
            exp = sum(self.w_seq)
            var = sum(w * (1 - w) for w in self.w_seq)
            z = (obs - exp) / math.sqrt(var) if var > 0 else None
            tag = ("n/a" if z is None else ("calibrated" if abs(z) < 2.5 else "MISCALIBRATED"))
            L.append(f" predicted mean {exp / m:.3f}   observed {obs / m:.3f}"
                     f"   z={'-' if z is None else f'{z:+.1f}'}   {tag}")
            L.append(f" lowest density seen {self.min_w:.3f}   "
                     f"(Rabin bound 0.750: {'ok' if self.min_w >= 0.75 else 'VIOLATED'})")
        else:
            L.append(" need >= 10 members of L")
        L.append("")
        L.append("Independence of A across trials (assumed by (1-p)^N)")
        if m >= 10:
            pairs = list(zip(self.a_seq, self.a_seq[1:]))
            n11 = sum(1 for u, v in pairs if u and v)
            n10 = sum(1 for u, v in pairs if u and not v)
            n01 = sum(1 for u, v in pairs if not u and v)
            n00 = sum(1 for u, v in pairs if not u and not v)
            den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
            phi = (n11 * n00 - n10 * n01) / den if den else None
            L.append(f" lag-1 phi of A on successive members: "
                     f"{'n/a (A nearly constant)' if phi is None else f'{phi:+.2f}'}")
            L.append(f" distinct witnesses r: {len(set(self.r_seq)) / m:.0%} of members")
            nb = m // k
            if nb >= 1:
                zeros = exp_z = 0.0
                for i in range(nb):
                    seg = range(i * k, (i + 1) * k)
                    zeros += int(not any(self.a_seq[j] for j in seg))
                    pr = 1.0
                    for j in seg:
                        pr *= 1 - self.w_seq[j]
                    exp_z += pr
                L.append(f" blocks of k={k} members with no witness: observed {int(zeros)}, "
                         f"expected {exp_z:.2f}  (of {nb})")
        else:
            L.append(" need >= 10 members of L")
        L.append("")
        L.append(f"Logic check  Q == " + ("(A.B) OR Y" if force else "A.B.Y") +
                 f" : {self.q_mismatch} mismatches in {self.q_checked} frames")
        return L


# ------------------------------------------------------------------ frame source
class Source:
    """Real USB camera via OpenCV."""

    def __init__(self, index: int, width: int = 640, height: int = 480):
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed.  Run: pip install opencv-python")
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not (cap.isOpened() and cap.read()[0]):
            cap.release()
            raise RuntimeError(
                f"Cannot open camera {index}. Check that it is connected, not in use "
                f"by another app, and that the camera number is right.")
        self.cap = cap
        self.rgb = None                  # latest colour frame (for the preview)

    def read(self):
        """Return a full-resolution uint8 gray frame, or None."""
        ok, bgr = self.cap.read()
        if not ok:
            return None
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def close(self):
        if getattr(self, "_closing", False):
            return
        self._closing = True
        try:
            self.cap.release()
        except Exception:
            pass


def downscale(gray: np.ndarray, blur: int = 5) -> np.ndarray:
    """Shrink and blur so sensor grain does not count as motion."""
    small = cv2.resize(gray, (SW, SH), interpolation=cv2.INTER_AREA)
    k = max(1, int(blur)) | 1                      # kernel must be odd
    return small if k == 1 else cv2.GaussianBlur(small, (k, k), 0)


def make_preview(rgb: np.ndarray, mask: np.ndarray, overlay: bool) -> np.ndarray:
    """Colour camera frame at PW x PH, optionally with the motion mask in red."""
    h, w = rgb.shape[:2]
    pv = rgb[np.arange(PH) * h // PH][:, np.arange(PW) * w // PW].copy()
    if overlay and mask.any():
        big = mask[np.arange(PH) * SH // PH][:, np.arange(PW) * SW // PW]
        pv[big] = (pv[big] * 0.4 + np.array([255, 40, 40]) * 0.6).astype(np.uint8)
    return pv


# ------------------------------------------------------------------ pipeline
@dataclass
class Packet:
    idx: int
    dt: float
    x: int
    motion: bool
    rand_int: int


@dataclass
class St:
    A: int = 0
    r: int = 0
    B: int = 0
    c_of_x: int = 0
    vC: float = 0.0
    vB: float = 0.0
    traj: list = field(default_factory=list)
    node: float = 0.0
    orv: int = 0
    andv: int = 0
    gate: int = 0
    y_bin: int = 1
    y_idx: int = 0
    Q: int = 0
    q_raw: int = 0
    q_tx: int = 0
    q_flip: int = 0
    q_corrected: int = 0
    q_errors: int = 0
    q_codeword: str = ""


def _hamming74_encode(nibble: int) -> list:
    """SEC Hamming(7,4), positions 1,2,4 are parity; data at 3,5,6,7."""
    d = [(int(nibble) >> k) & 1 for k in (3, 2, 1, 0)]
    b = [0] * 8
    b[3], b[5], b[6], b[7] = d
    b[1] = b[3] ^ b[5] ^ b[7]
    b[2] = b[3] ^ b[6] ^ b[7]
    b[4] = b[5] ^ b[6] ^ b[7]
    return b[1:]


def _hamming74_decode(bits: list):
    """Return (decoded nibble, corrected-bit-position); corrects one bit."""
    b = [0] + [int(v) & 1 for v in bits[:7]]
    while len(b) < 8:
        b.append(0)
    syndrome = ((b[1] ^ b[3] ^ b[5] ^ b[7]) |
                ((b[2] ^ b[3] ^ b[6] ^ b[7]) << 1) |
                ((b[4] ^ b[5] ^ b[6] ^ b[7]) << 2))
    if 0 < syndrome <= 7:
        b[syndrome] ^= 1
    nibble = (b[3] << 3) | (b[5] << 2) | (b[6] << 1) | b[7]
    return nibble, syndrome


def q_channel(raw: int, ctl: dict, frame: int, entropy: int):
    """Apply configurable Q inversion, fault injection, and optional ECC.
    Hamming(7,4) encodes 0000/1111 so its decoded majority represents Q."""
    raw = int(bool(raw))
    if ctl.get("q_invert", False):
        raw ^= 1
    mode = ctl.get("q_latch", "D")
    prev = int(ctl.get("_q_prev", 0))
    if mode == "T":
        raw = prev ^ raw
    elif mode == "JK":                       # J=raw, K=0: set or hold
        raw = 1 if raw else prev
    elif mode == "SR":                       # S=raw, R=not raw: set/reset
        raw = 1 if raw else 0
    if ctl.get("q_reset", False):
        raw = 0
    ctl["_q_prev"] = raw

    ecc = ctl.get("q_ecc", "none")
    if ecc == "repetition-3":
        code = [raw] * 3
    elif ecc == "repetition-5":
        code = [raw] * 5
    elif ecc == "repetition-7":
        code = [raw] * 7
    elif ecc == "Hamming(7,4)":
        code = _hamming74_encode(15 if raw else 0)
    else:
        code = [raw]

    mode = ctl.get("q_flip_mode", "random")
    prob = min(1.0, max(0.0, float(ctl.get("q_flip_prob", 0.0))))
    period = max(1, int(ctl.get("q_flip_period", 10)))
    burst = max(1, int(ctl.get("q_burst_len", 3)))
    seed = (int(entropy) ^ (int(frame) * 0x9E3779B97F4A7C15) ^
            (int(ctl.get("q_seed", 0)) * 0xD6E8FEB86659FD93)) & ((1 << 256) - 1)
    flips = 0
    noisy = list(code)
    for i in range(len(noisy)):
        if mode == "force-0":
            flip = bool(noisy[i])
        elif mode == "force-1":
            flip = not bool(noisy[i])
        elif mode == "periodic":
            flip = (frame % period == 0)
        elif mode == "burst":
            flip = (frame % period) < burst
        else:
            h = int.from_bytes(hashlib.sha256(seed.to_bytes(32, "big") + bytes([i])).digest()[:8], "big")
            flip = h / float(1 << 64) < prob
        if flip:
            noisy[i] ^= 1
            flips += 1

    syndrome = 0
    if ecc == "Hamming(7,4)":
        nibble, syndrome = _hamming74_decode(noisy)
        decoded = int(nibble.bit_count() >= 2)
        corrected = int(syndrome != 0)
    elif ecc.startswith("repetition-"):
        decoded = int(sum(noisy) > len(noisy) // 2)
        corrected = int(decoded != raw and flips > 0)  # decoder outcome differs from source
    else:
        decoded = noisy[0]
        corrected = 0
    return decoded, dict(raw=raw, tx=int(bool(code[0])), flips=flips,
                         corrected=corrected, errors=flips,
                         codeword="".join(map(str, code)),
                         received="".join(map(str, noisy)), syndrome=syndrome)


class SimThread(threading.Thread):
    def __init__(self, cam_index, ctl, out):
        super().__init__(daemon=True)
        self.cam_index, self.ctl, self.out = cam_index, ctl, out

    def run(self):
        asyncio.run(self._run())

    async def _run(self):
        ctl, out = self.ctl, self.out
        hidden = search = fsearch = None      # oracle-exclusive: only the BTC oracle releases time codes
        try:
            try:
                cw, ch = (int(v) for v in str(ctl.get("cam_res", "640x480")).split("x"))
            except ValueError:
                cw, ch = 640, 480
            src = Source(self.cam_index, cw, ch)
        except RuntimeError as e:
            out.put(("error", str(e)))
            out.put(("done", None))
            return
        out.put(("info", f"USB camera {self.cam_index} opened"))
        st = St()
        qa, qb, qc = asyncio.Queue(), asyncio.Queue(), asyncio.Queue()
        beta_tau = TAU_NODE

        async def driver_a():                          # randomized witness test
            while True:
                p = await qa.get()
                if p is None:
                    qa.task_done()
                    return
                st.r, st.A = 0, 0
                if p.motion and p.x > 4 and p.x % 2 == 1:
                    span = p.x - 3
                    for j in range(max(1, int(ctl.get("a_rounds", 1)))):
                        h = p.rand_int if j == 0 else int.from_bytes(hashlib.sha256(
                            p.rand_int.to_bytes(32, "big") + bytes([j & 255])).digest(), "big")
                        r = 2 + h % span                          # r in [2, x-2]
                        if j == 0:
                            st.r = r
                        if miller_rabin_witness(r, p.x):
                            st.r, st.A = r, 1
                            break
                qa.task_done()

        async def driver_b():                          # deterministic check + link
            while True:
                p = await qb.get()
                if p is None:
                    qb.task_done()
                    return
                st.c_of_x = int(p.motion and p.x > int(ctl.get("b_min_x", 2))
                                and (p.x % 2 == 1 or not ctl.get("b_odd", True)))
                vs_, vbe_, vih_ = (float(ctl.get("vs", VS)), float(ctl.get("vbe", V_BE)),
                                   float(ctl.get("vih", V_IH)))
                rc_ = max(1e-5, float(ctl.get("rc_ms", RC_B * 1000)) / 1000.0)
                traj, n = [], 24
                for j in range(n + 1):                 # C charges from 0 each frame
                    t = p.dt * j / n
                    vC = vs_ * (1 - math.exp(-t / rc_)) if st.c_of_x else 0.0
                    traj.append((t * 1000, vC, max(0.0, vC - vbe_)))
                st.traj = traj
                st.vC, st.vB = traj[-1][1], traj[-1][2]
                st.B = int(st.vB >= vih_)
                qb.task_done()

        async def clock_ff():                          # gates, node, Y, 74HC74
            while True:
                p = await qc.get()
                if p is None:
                    qc.task_done()
                    return
                st.orv = st.A | st.B
                st.andv = st.A & st.B
                st.gate = gate_out(ctl.get("gate", "AND"), st)                       # tied -> AND
                tau = max(1e-5, float(ctl.get("tau_ms", beta_tau * 1000)) / 1000.0)
                st.node = st.gate * (1 - math.exp(-p.dt / tau))        # 10 uF node
                bit, st.y_idx = y_bin_at(ctl, p.idx, time.perf_counter() - t_start)
                cap = int(ctl.get("pending_cap", 0))
                if cap > 0 and ctl.get("timecode_pending", 0) > cap:
                    ctl["timecode_pending"] = cap
                added = 1 if ctl.get("timecode_pending", 0) > 0 else 0
                st.y_bin = bit | added
                if added:                                # one found number = one added bin
                    ctl["timecode_pending"] -= 1
                d_in = int(st.node > float(ctl.get("node_thr", 0.5)))
                if ctl.get("force"):                             # Y OR-ed in (Thm 4 demo)
                    st.q_raw = d_in | st.y_bin
                else:
                    st.q_raw = d_in & st.y_bin
                st.Q, qdiag = q_channel(st.q_raw, ctl, p.idx, p.rand_int)
                st.q_tx = qdiag["tx"]
                st.q_flip = qdiag["flips"]
                st.q_corrected = qdiag["corrected"]
                st.q_errors = qdiag["errors"]
                st.q_codeword = qdiag["codeword"]
                qc.task_done()

        qn = asyncio.Queue()
        AN = DriverAnalysis()

        # Shared statistics. Defined BEFORE the tasks so it never races the closures.
        S = dict(frames=0, motion=0, in_L=0, acc_in_L=0, false_acc=0, accepts=0,
                 open_inL=0, acc_open_inL=0,
                 mined_even=0, last_x=0, last_even=False, last_armed=False,
                 timecodes_added=0, hidden_timecodes=0, search_timecodes=0,
                 factor_timecodes=0)

        def codes_snapshot():
            """Time-code statistics from the hidden-number and bit searches."""
            return dict(
                circuit=bool(ctl.get("circuit", True)),
                last_x=S["last_x"], last_even=S["last_even"], last_armed=S["last_armed"],
                armed_frames=S["mined_even"],
                timecodes_added=S["timecodes_added"],
                hidden_timecodes=S["hidden_timecodes"],
                search_timecodes=S["search_timecodes"],
                factor_timecodes=S["factor_timecodes"],
                search=S.get("search"),
                fsearch=S.get("fsearch"),
                hidden=S.get("hidden"),
                timecode_pending=int(ctl.get("timecode_pending", 0)))

        async def analyst():                           # runs beside the drivers
            seen = 0
            while True:
                rec = await qn.get()
                if rec is None:
                    out.put(("analysis", AN.lines(ctl["k"], ctl.get("force", False))))
                    return
                if ctl.get("reset_an"):
                    AN.reset()
                    ctl["reset_an"] = False
                w = 0.0
                if rec["in_L"]:                        # exact density, off the loop
                    w = _WD.get(rec["x"])
                    if w is None:
                        w = await asyncio.to_thread(witness_density, rec["x"])
                AN.update(rec, w)
                seen += 1
                if seen % 5 == 0 or qn.empty():
                    out.put(("analysis", AN.lines(ctl["k"], rec["force"])))

        tasks = [asyncio.create_task(f())
                 for f in (driver_a, driver_b, clock_ff, analyst)]

        prev = None
        prev_even = False                  # was the previous frame an even state?
        prev_hist = deque(maxlen=31)       # recent small frames for the "diff gap" setting
        t_prev = time.perf_counter()
        t_start = t_prev                   # wall-clock origin for 'ms' time-code bins
        ctl["fps_est"] = None
        idx = 0
        recent_q = deque(maxlen=60)
        try:
            while not ctl["stop"]:
                mf = float(ctl.get("max_fps", 0))
                if mf > 0:                                   # frame-rate limiter
                    wait = t_prev + 1.0 / mf - time.perf_counter()
                    if wait > 0:
                        await asyncio.sleep(wait)
                gray = await asyncio.to_thread(src.read)
                if gray is None:
                    out.put(("info", "camera stopped delivering frames"))
                    break
                now = time.perf_counter()
                dt = min(max(now - t_prev, 1e-3), max(0.01, float(ctl.get("dt_max_ms", 250)) / 1000.0))
                t_prev = now
                f_inst = 1.0 / dt
                ctl["fps_est"] = f_inst if ctl.get("fps_est") is None else 0.9 * ctl["fps_est"] + 0.1 * f_inst

                # ---- motion detection
                small = downscale(gray, ctl.get("blur", 5))
                if prev is None or idx < int(ctl.get("warmup", WARMUP)):
                    diff = np.zeros((SH, SW), np.int16)
                    mask = np.zeros((SH, SW), bool)
                else:
                    g = min(max(1, int(ctl.get("diff_gap", 1))), len(prev_hist)) if prev_hist else 1
                    ref = prev_hist[-g] if prev_hist else prev
                    diff = np.abs(small.astype(np.int16) - ref.astype(np.int16))
                    mask = diff > ctl["thr"]
                    nd = int(ctl.get("dilate", 0))
                    if nd > 0:
                        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                                          iterations=nd).astype(bool)
                prev = small
                prev_hist.append(small)
                count = int(mask.sum())
                motion = count >= ctl["min_px"]
                x = max(1, count // max(1, int(ctl.get("x_div", 1))) + int(ctl.get("x_off", 0))) if motion else 0

                # ---- witness source: the motion pattern (same camera, no noise)
                energy = np.where(mask, np.clip(diff, 0, 255), 0).astype(np.uint8)
                digest = hashlib.sha256(energy.tobytes()).digest()
                if ctl.get("pool") is not None:
                    ctl["pool"].feed(digest)            # camera motion seeds the factoriser randomness
                pkt = Packet(idx, dt, x, motion, int.from_bytes(digest, "big"))

                # A hidden-number spot check or a search guess is only allowed after TWO SEQUENTIAL even
                # states: this frame and the one right before it must both be even. A frame
                # with no motion (x = 0) or an odd x breaks the run.
                even_now = x > 0 and x % 2 == 0
                even_state = even_now and prev_even        # armed: two evens in a row
                prev_even = even_now
                S["last_x"], S["last_even"], S["last_armed"] = x, even_now, even_state
                if even_state:
                    S["mined_even"] += 1                   # armed frames seen

                # Frequency decision: while even frames are available, sample a square wave at the
                # chosen Hz. First half of the cycle = HIGH, second half = LOW. The oracle uses the latest sample.
                if even_now and ctl.get("btc_oracle") is not None:
                    _hz = float(ctl.get("btc_freq_hz", 17.0))
                    _ph = (time.perf_counter() * _hz) % 1.0
                    ctl["btc_oracle"].set_freq_call("higher" if _ph < 0.5 else "lower", _hz)

                # BTC high/low oracle: one armed release is consumed on the next even state.
                if even_now and ctl.get("btc_release_armed", False):
                    ctl["btc_release_armed"] = False
                    ctl["timecode_pending"] = ctl.get("timecode_pending", 0) + int(ctl.get("bins_per_hit", 1))
                    S["btc_timecodes"] = S.get("btc_timecodes", 0) + 1
                    ctl["btc_released"] = ctl.get("btc_released", 0) + 1
                    S["timecodes_added"] = S.get("timecodes_added", 0) + 1
                    btc_call = ctl.get("btc_last_call", "--")
                    btc_price = ctl.get("btc_last_price", 0.0)
                    out.put(("log",
                        f"BTC HIGH/LOW ORACLE EVEN RELEASE: even frame x={x}; "
                        f"oracle call={btc_call}; BTC/USDT=${btc_price:,.2f}; "
                        f"one time-code bin released; pending={ctl['timecode_pending']}."
                    ))

                # Hidden-number test: only two sequential even states trigger a spot check, and it reads
                # the clock-cycled candidate at this very moment.
                hidden_hit = False
                if hidden is not None and ctl.get("hidden_on", True) and even_state:
                    hidden_hit = hidden.spot_check()
                    if hidden_hit:
                        ctl["timecode_pending"] = ctl.get("timecode_pending", 0) + int(ctl.get("bins_per_hit", 1))
                        S["hidden_timecodes"] += 1
                        S["timecodes_added"] += 1
                        out.put(("hidden_hit", dict(frame=idx, x=x, time=time.time(),
                                                    snap=hidden.snapshot())))
                if hidden is not None:
                    S["hidden"] = hidden.snapshot()

                # Higher/lower search: one oracle guess per armed frame (two sequential evens)
                search_hit = False
                if search is not None and ctl.get("search_on", True) and even_state:
                    if ctl.get("inj_last_bit", False) and search.pooled_n == search.bits - 1:
                        _shift = search.bits - 1 - search.pooled_n
                        _bit = 1 if (search.secret >> _shift) & 1 else 0
                        _key = ctl.get("macro_higher", "w") if _bit else ctl.get("macro_lower", "s")
                        try:
                            inject_combo(
                                _key,
                                shift=True,
                                hold=float(ctl.get("inj_hold_ms", 40)) / 1000.0,
                                scan=bool(ctl.get("inj_scan", True)),
                            )
                            out.put("info", f"INJECTED Shift+{_key.upper()} for final search bit {_bit}")
                        except Exception as _exc:
                            out.put("error", f"final-bit keystroke failed: {_exc}")
                    search_hit = search.guess()
                    if search_hit:
                        ctl["timecode_pending"] = ctl.get("timecode_pending", 0) + int(ctl.get("bins_per_hit", 1))
                        S["search_timecodes"] += 1
                        S["timecodes_added"] += 1
                        sn = search.snapshot()
                        out.put(("search_hit", dict(frame=idx, x=x, bits=sn["bits"],
                                                    solves=sn["solves"], avg=sn["avg"],
                                                    found=sn["last_found"])))
                if search is not None:
                    S["search"] = search.snapshot()

                # Factoriser: one higher/lower query per armed frame; a time code is earned ONLY when the
                # single final check verifies a correct factorisation (separate source, own counter).
                factor_hit = False
                if fsearch is not None and ctl.get("fsearch_on", True) and even_state:
                    if ctl.get("inj_last_bit", False) and fsearch.pooled_n == fsearch.width - 1:
                        _shift = fsearch.width - 1 - fsearch.pooled_n
                        _bit = 1 if (fsearch.secret >> _shift) & 1 else 0
                        _key = ctl.get("macro_higher", "w") if _bit else ctl.get("macro_lower", "s")
                        try:
                            inject_combo(
                                _key,
                                shift=True,
                                hold=float(ctl.get("inj_hold_ms", 40)) / 1000.0,
                                scan=bool(ctl.get("inj_scan", True)),
                            )
                            out.put("info", f"INJECTED Shift+{_key.upper()} for final factor bit {_bit}")
                        except Exception as _exc:
                            out.put("error", f"final-bit keystroke failed: {_exc}")
                    factor_hit = fsearch.guess()
                    if factor_hit:
                        ctl["timecode_pending"] = ctl.get("timecode_pending", 0) + int(ctl.get("bins_per_hit", 1))
                        S["factor_timecodes"] += 1
                        S["timecodes_added"] += 1
                        fn = fsearch.snapshot()
                        out.put(("factor_hit", dict(frame=idx, x=x, found=fn["last_found"],
                                                    solves=fn["solves"])))
                if fsearch is not None:
                    S["fsearch"] = fsearch.snapshot()
                # ---- drivers A and B on this frame, concurrently; then flip-flop
                run_circuit = ctl.get("circuit", True)
                if run_circuit:
                    await qa.put(pkt)
                    await qb.put(pkt)
                    await asyncio.gather(qa.join(), qb.join())
                    await qc.put(pkt)
                    await qc.join()
                else:                                  # preview only: circuit idle
                    for name, val in vars(St()).items():
                        setattr(st, name, val)

                out.put(("codes", codes_snapshot()))

                # ---- bookkeeping
                in_L = bool(motion and is_odd_composite(x))
                select = 0
                if run_circuit:
                    recent_q.append(st.Q)
                    select = int(any(list(recent_q)[-ctl["k"]:]))
                    S["frames"] += 1
                    if st.Q and not in_L:
                        S["false_acc"] += 1                # x not in L was accepted
                    if motion:
                        S["motion"] += 1
                        if in_L:
                            S["in_L"] += 1
                            S["acc_in_L"] += st.Q
                            if st.y_bin:                   # frames where Y is open
                                S["open_inL"] += 1
                                S["acc_open_inL"] += st.Q
                        S["accepts"] += st.Q

                if run_circuit:
                    qn.put_nowait(dict(x=x, motion=motion, in_L=in_L, A=st.A, B=st.B,
                                       c_of_x=st.c_of_x, Q=st.Q, y_bin=st.y_bin,
                                       r=st.r, force=bool(ctl.get("force"))))
                rgb = src.rgb if src.rgb is not None else np.stack([gray] * 3, axis=-1)
                preview = make_preview(rgb, mask, ctl.get("overlay", True))
                out.put(("frame", dict(
                    circuit=run_circuit, preview=preview, idx=idx, dt=dt, x=x, motion=motion, in_L=in_L, small=small,
                    mask=mask, energy=energy, A=st.A, r=st.r, B=st.B,
                    c_of_x=st.c_of_x, vC=st.vC, vB=st.vB, traj=st.traj,
                    node=st.node, orv=st.orv, andv=st.andv, gate=st.gate,
                    y_bin=st.y_bin, y_idx=st.y_idx, Q=st.Q, q_raw=st.q_raw,
                    q_tx=st.q_tx, q_flip=st.q_flip, q_corrected=st.q_corrected,
                    q_errors=st.q_errors, q_codeword=st.q_codeword,
                    select=select, stats=dict(S),
                    hidden_hit=hidden_hit, search_hit=search_hit, factor_hit=factor_hit)))
                idx += 1
        finally:
            for q in (qa, qb, qc, qn):
                await q.put(None)
            await asyncio.gather(*tasks)
            src.close()
            out.put(("done", None))


# ------------------------------------------------------------------ gui
OFF, ON, GOOD = "#f1efe8", "#ffd9a8", "#b9e6b9"
W_OFF, W_ON = "#999999", "#d9480f"


def ppm_photo(rgb: np.ndarray) -> tk.PhotoImage:
    h, w, _ = rgb.shape
    data = b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(rgb, np.uint8).tobytes()
    return tk.PhotoImage(data=data, format="PPM")


def set_text(widget, text, **kw):
    """Update a Tk label only when its text/options really changed."""
    cache = getattr(widget, "_cache", None)
    key = (text, tuple(sorted(kw.items())))
    if cache != key:
        widget.config(text=text, **kw)
        widget._cache = key


def fixed_label(parent, width_px, height_px, **kw):
    """A label inside a fixed-size frame: its text can change without resizing the layout."""
    box = ttk.Frame(parent, width=width_px, height=height_px)
    box.pack_propagate(False)
    lab = ttk.Label(box, **kw)
    lab.pack(anchor="nw")
    return box, lab


FRAME_INTERVAL = 33                # ms between GUI redraws (~30 fps cap)
LOG_INTERVAL = 100                 # ms between log flushes
MINER_INTERVAL = 0.2               # s between miner panel refreshes



class BitcoinHighLowOracle:
    """Binance BTC/USDT HIGH/LOW oracle: continuous up/down bits, binary decision tree,
    one end check per window.

    observe()   : called for EVERY sample (e.g. each second). Records the binary value
                  (U = price rose, D = price fell, unchanged = skipped) in the recent-bit
                  context, and trains the tree with every sample that is now `window_s`
                  seconds old: its outcome is the price now versus the price then.
    end_check() : called ONCE per window. Scores the call made at the start of the window
                  (a tie is not scored) and makes the call for the next window. The call
                  comes from the deepest tree node (up to `depth` bits) with at least
                  MIN_OBS observations and a strict majority; otherwise NO CALL.

    Windows do not overlap, so the binomial p-value of the end-check record is valid.
    BTC over seconds/minutes is close to a random walk: expect ~50%.
    fetch_sample() does network I/O: call it from a worker thread only.
    """

    URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    MAX_DEPTH = 8
    MIN_OBS = 3

    def __init__(self):
        self._lock = threading.RLock()
        self.depth = 3
        self.window_s = 10.0
        self.last_price = None
        self.last_hash = None
        self.last_factual = None
        self.last_checks = None
        self.last_update = None
        self.fresh_count = 0
        self.factual_count = 0
        self.rejected_count = 0
        self.released_count = 0          # kept for compatibility; the GUI shows ctl["btc_released"]
        self.pending_manual = False
        self.m_correct = self.m_total = 0
        self.reset_tree()
        self.pow = ToyPowChain(difficulty_bits=5)
        self.use_hash = False          # A/B switch, off by default
        self.pow_hash = None
    # ---------------------------------------------------------- state
    def reset_tree(self):
        with self._lock:
            self.tree = {}                                   # path tuple -> [ups, downs]
            self.moves = deque(maxlen=self.MAX_DEPTH)        # recent binary values, "U"/"D"
            self.samples = deque(maxlen=20000)               # (time, price, context) waiting for the window
            self._prev_tick_price = None
            self.pending_guess = None
            self.pending_price = None
            self.last_prediction = None
            self.correct = self.total = self.ties = self.abstains = 0
            self.ticks = self.trained = 0
            self.pending_manual = False
            self.m_correct = self.m_total = 0

    def set_depth(self, depth):
        with self._lock:
            self.depth = max(1, min(self.MAX_DEPTH, int(depth)))

    def set_window(self, seconds):
        with self._lock:
            self.window_s = max(1.0, float(seconds))

    def _learn_ctx(self, ctx, outcome):
        for L in range(0, len(ctx) + 1):
            node = self.tree.setdefault(ctx[len(ctx) - L:], [0, 0])
            node[0 if outcome == "U" else 1] += 1

    def _predict(self):
        """(call, depth_used, ups, downs) from the deepest qualifying node, or None."""
        h = list(self.moves)
        for L in range(min(self.depth, len(h)), -1, -1):
            node = self.tree.get(tuple(h[len(h) - L:]))
            if node and sum(node) >= self.MIN_OBS and node[0] != node[1]:
                return ("higher" if node[0] > node[1] else "lower", L, node[0], node[1])
        return None

    def recent_bits(self):
        return "".join(self.moves) or "-"

    def manual_call(self, call):
        """Keyboard macro: replace the pending call with 'higher' or 'lower' at the latest price.
        It is scored at the next end check like any tree call, but counted separately so the
        tree's own record stays clean. Returns the price, or None if no sample has arrived yet."""
        with self._lock:
            if self.last_price is None:
                return None
            self.pending_guess = call
            self.pending_price = self.last_price
            self.pending_manual = True
            self.last_prediction = {"guess": call.upper(), "made_at_price": self.last_price,
                                    "node": None, "manual": True}
            return self.last_price

    # ---------------------------------------------------------- checks (unchanged logic)
    @staticmethod
    def _driver_b_checks(raw, data, price):
        checks = {
            "payload_nonempty": isinstance(raw, bytes) and len(raw) > 0,
            "json_object": isinstance(data, dict),
            "symbol_key": isinstance(data, dict) and data.get("symbol") == "BTCUSDT",
            "price_key": isinstance(data, dict) and "price" in data,
            "price_numeric": isinstance(price, (int, float)) and not isinstance(price, bool),
            "price_finite": isinstance(price, (int, float)) and math.isfinite(float(price)),
            "price_positive": isinstance(price, (int, float)) and float(price) > 0,
        }
        checks["driver_b"] = all(checks.values())
        return checks

    @staticmethod
    def _driver_a_check(raw):
        digest = hashlib.sha256(raw).digest()
        x = int.from_bytes(digest[:8], "big") | 1
        base = int.from_bytes(digest[8:12], "big") % max(2, x - 2) + 2 if x > 3 else 2
        composite = is_odd_composite(x)
        witness = miller_rabin_witness(base, x) if composite else False
        checks = {
            "hash_reproducible": hashlib.sha256(raw).digest() == digest,
            "witness_consistent": (not composite) or witness,
        }
        checks["driver_a"] = all(checks.values())
        return {"digest": digest.hex(), "x": x, "composite": composite, **checks}

    # ---------------------------------------------------------- network
    def fetch_sample(self):
        """One Binance request. Returns (price, factual, a_checks, b_checks)."""
        request = urllib.request.Request(
            self.URL, headers={"User-Agent": "rp-circuit-btc-oracle/1.0"})
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read()
        data = json.loads(raw.decode())
        price = float(data["price"])
        b_checks = self._driver_b_checks(raw, data, price)
        a_checks = self._driver_a_check(raw)
        return price, b_checks["driver_b"] and a_checks["driver_a"], a_checks, b_checks

    # ---------------------------------------------------------- every sample
    MAX_NODES = 5000        # hard cap on tree size
    HASH_DEPTH = 2          # hash bit only splits contexts of length <= this

    # __init__: add before self.reset_tree()
    #   self.pow = ToyPowChain(difficulty_bits=12)
    #   self.use_hash = False          # A/B switch, off by default
    #   self.pow_hash = None

    def set_use_hash(self, on):
        with self._lock:
            self.use_hash = bool(on)

    def _bump(self, key, outcome):
        node = self.tree.get(key)
        if node is None:
            if len(self.tree) >= self.MAX_NODES:
                return                              # cardinality cap: no new nodes
            node = self.tree[key] = [0, 0]
        node[0 if outcome == "U" else 1] += 1

    def _learn_ctx(self, ctx, hbit, outcome):
        for L in range(len(ctx) + 1):
            suffix = ctx[len(ctx) - L:]
            self._bump(suffix, outcome)             # plain keys: unchanged
            if self.use_hash and L <= self.HASH_DEPTH:
                self._bump(("H" + hbit,) + suffix, outcome)   # separate keyspace

    def _predict(self):
        h = list(self.moves)
        hb = "H" + self.pow.feature_bit()           # bit of the LAST committed sample
        for L in range(min(self.depth, len(h)), -1, -1):
            suffix = tuple(h[len(h) - L:])
            keys = [suffix]
            if self.use_hash and L <= self.HASH_DEPTH:
                keys.insert(0, (hb,) + suffix)
            for key in keys:
                node = self.tree.get(key)
                if node and sum(node) >= self.MIN_OBS and node[0] != node[1]:
                    return ("higher" if node[0] > node[1] else "lower", L, node[0], node[1])
        return None

    def observe(self, price, factual, a_checks, b_checks, now=None):
        now = time.time() if now is None else now
        self.pow.mine(price)                        # outside the lock
        hbit = self.pow.feature_bit()
        with self._lock:
            self.last_price = price
            self.last_hash = a_checks["digest"]
            self.pow_hash = self.pow.last_hash.hex()
            self.last_factual = factual
            self.last_checks = {"driver_a": a_checks, "driver_b": b_checks, "factual": factual}
            self.last_update = now
            self.fresh_count += 1
            if factual:
                self.factual_count += 1
            else:
                self.rejected_count += 1

            while self.samples and now - self.samples[0][0] >= self.window_s:
                _t0, p0, ctx, h0 = self.samples.popleft()
                if price != p0:
                    self._learn_ctx(ctx, h0, "U" if price > p0 else "D")
                    self.trained += 1

            if self._prev_tick_price is not None and price != self._prev_tick_price:
                self.moves.append("U" if price > self._prev_tick_price else "D")
            self._prev_tick_price = price
            self.ticks += 1
            self.samples.append((now, price, tuple(self.moves), hbit))
            # ---------------------------------------------------------- once per window
    def end_check(self):
        """Score the previous window's call, then make the call for the next window.
        Returns (price, result_or_None, factual, changed) or None if there is no sample yet."""
        with self._lock:
            if self.last_price is None:
                return None
            price = self.last_price
            changed = self.pending_price is None or price != self.pending_price

            result = None
            if self.pending_guess is not None and self.pending_price is not None:
                if price == self.pending_price:
                    self.ties += 1
                else:
                    went_up = price > self.pending_price
                    is_correct = (self.pending_guess == "higher") == went_up
                    if self.pending_manual:
                        self.m_total += 1
                        self.m_correct += int(is_correct)
                    else:
                        self.total += 1
                        self.correct += int(is_correct)
                    result = {
                        "guess": self.pending_guess.upper(),
                        "previous": self.pending_price,
                        "actual": price,
                        "actual_direction": "HIGH" if went_up else "LOW",
                        "correct": is_correct,
                        "manual": self.pending_manual,
                    }

            pred = self._predict()
            self.pending_manual = False
            if pred is None:
                self.abstains += 1
                self.pending_guess = None
                self.pending_price = None
                self.last_prediction = {"guess": "NO CALL", "made_at_price": price, "node": None}
            else:
                call, used, ups, downs = pred
                self.pending_guess = call
                self.pending_price = price
                self.last_prediction = {"guess": call.upper(), "made_at_price": price,
                                        "node": (used, ups, downs)}
            return price, result, self.last_factual, changed

    def stats(self):
        with self._lock:
            n, k = self.total, self.correct
            return dict(correct=k, total=n, ties=self.ties, abstains=self.abstains,
                        rate=(k / n if n else None),
                        nodes_hash=sum(1 for k in self.tree if k and isinstance(k[0], str) and k[0].startswith("H")),
                        use_hash=self.use_hash,
                        p_value=(prob_at_least(k, n, 0.5) if n else None),
                        nodes=len(self.tree), depth=self.depth, window=self.window_s,
                        ticks=self.ticks, trained=self.trained, bits=self.recent_bits(),
                        m_correct=self.m_correct, m_total=self.m_total)


class ScrollFrame(ttk.Frame):
    """A frame whose content scrolls (wheel + bars) when the window is smaller than it."""

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_outer)
        for w in (self.canvas, self.inner):
            w.bind("<Enter>", self._bind_wheel)
            w.bind("<Leave>", self._unbind_wheel)

    def _on_inner(self, _e=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._schedule_bars()

    def _on_outer(self, _e=None):
        self._schedule_bars()

    def _schedule_bars(self):
        # debounce: size changes can toggle the scrollbars, which change the size again
        if getattr(self, "_bars_job", None) is None:
            self._bars_job = self.after(60, self._run_bars)

    def _run_bars(self):
        self._bars_job = None
        self._toggle_bars()

    def _toggle_bars(self):
        """Show a scrollbar only while the content really is bigger than the window."""
        need_w = self.inner.winfo_reqwidth()
        need_h = self.inner.winfo_reqheight()
        have_w = self.canvas.winfo_width()
        have_h = self.canvas.winfo_height()
        if need_h > have_h + 1:
            self.vbar.grid()
        else:
            self.vbar.grid_remove()
            self.canvas.yview_moveto(0)
        if need_w > have_w + 1:
            self.hbar.grid()
        else:
            self.hbar.grid_remove()
            self.canvas.xview_moveto(0)

    def _bind_wheel(self, _e=None):
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", self._wheel)
        self.canvas.bind_all("<Button-5>", self._wheel)

    def _unbind_wheel(self, _e=None):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _wheel(self, e):
        # leave the wheel alone over spinboxes (it changes their value) and text boxes
        if e.widget.winfo_class() in ("TSpinbox", "Text"):
            return
        if not self.vbar.winfo_ismapped():
            return
        step = -1 if (getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0) else 1
        self.canvas.yview_scroll(step * 3, "units")


class App:
    def __init__(self, window):
        self.window = window                          # the real Tk window
        window.title("RP circuit + hidden number, bit search and BTC HIGH/LOW even-frame oracle")
        self.scroller = ScrollFrame(window)
        self.scroller.pack(fill="both", expand=True)
        root = self.scroller.inner                    # every widget below lives in here
        self.root = root
        root.after = window.after                     # keep timers on the real window
        root.after_idle = window.after_idle
        root.protocol = window.protocol
        root.destroy = window.destroy
        self.q = queue.Queue()
        self.thread = None
        self.inj = dict(tree=False, manual=False, change_only=False, scan=True, delay_ms=0, hold_ms=40)
        self._inject_guard_until = 0.0
        self._last_injected = None
        self.ctl = dict(stop=False, thr=18, min_px=30, k=5, y="1111", circuit=True,
                        overlay=True, force=False,
                        timecode_pending=0,
                        hidden_on=True, search_on=True)
        self.ctl.update(TUNE_DEFAULTS)
        self.ctl.update(
            inj_last_bit=False,
            macro_higher="w",
            macro_lower="s",
            inj_hold_ms=40,
            inj_scan=True,
        )
        self.hidden = HiddenNumber(HIDDEN_RANGE_DEFAULT)
        self.hidden.start()
        self.ctl["hidden"] = self.hidden
        self.search = BinarySearch(SEARCH_BITS_DEFAULT)
        self.ctl["search"] = self.search
        self.pool = EntropyPool()
        self.fsearch = FactorSearch(self.pool)
        self.ctl.update(fsearch=self.fsearch, fsearch_on=True, pool=self.pool)
        self.btc = BitcoinHighLowOracle()
        self.photos = [None, None, None]
        self.btc_auto_job = None
        self.btc_auto_running = False
        self.q_hist = deque(maxlen=84)          # (Q, y_bin, motion)
        self.x_hist = deque(maxlen=150)
        self.last_traj = []
        self.err = False
        self.an_win = None
        self.an_lines = ["Press Start circuit, then move in front of the camera."]
        self.code_info = dict(carries=0, wrapped=0, expr="1111")
        self.y_ok = True

        window.resizable(True, True)
        root.grid_columnconfigure(0, weight=0, minsize=730)     # circuit diagram
        root.grid_columnconfigure(1, weight=0, minsize=440)     # plots + verdict
        root.grid_columnconfigure(2, weight=1, minsize=300)     # camera + analysis
        for r in range(3):
            root.grid_rowconfigure(r, weight=0)
        self._last_render = 0.0
        self._last_codes = 0.0
        self._pending_codes = None
        self._log_buf = []
        self._photo_cache = {}

        self._controls()
        self._miner_panel()
        self._btc_panel()
        self.c = tk.Canvas(root, width=710, height=410, bg="white",
                           highlightthickness=1, highlightbackground="#bbb")
        self.c.grid(row=1, column=0, padx=6, pady=4)
        self.boxes, self.wires = {}, {}
        self._diagram()
        self._right_panel()
        side = ttk.Frame(root)                         # column 2: camera over analysis
        side.grid(row=1, column=2, sticky="n", padx=6, pady=4)
        self.side = side
        self.cam = tk.Canvas(side, width=PW, height=PH, bg="black",
                             highlightthickness=1, highlightbackground="#bbb")
        self.cam.pack(anchor="w")
        self.cam_img = self.cam.create_image(0, 0, anchor="nw")
        self.cam_text = self.cam.create_text(
            8, 8, anchor="nw", fill="#ffd54a", font=("Consolas", 10, "bold"),
            text="Live camera preview\n(press Preview only or Start circuit)")
        self.cam_photo = None
        box, self.analysis = fixed_label(side, PW, 290, text="", font=("Consolas", 8),
                                         justify="left", wraplength=PW - 6)
        box.pack(anchor="w", pady=(6, 0))
        self.analysis_box = box
        # the log sits in the time-code row (see _miner_panel)
        self._y_changed()
        self._compact = False
        self.v_compact = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.ctl_row2, text="Compact (small screens)", variable=self.v_compact,
                        command=self._apply_compact).pack(side="left", padx=(12, 0))
        window.after(200, self._auto_compact)
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(15, self._poll)
        root.after(300, self._factor_tick)

    # ---------------------------------------------------------- compact mode
    def _apply_compact(self):
        """Hide the bulky read-only extras so the essentials fit a laptop screen:
        the IF/THEN link plot and the time-code analysis text."""
        on = self.v_compact.get()
        if on == self._compact:
            return
        self._compact = on
        if on:
            self.analysis_box.pack_forget()
            self.plot1.pack_forget()
            self.plot1_label.pack_forget()
            self.log.config(height=7)
            self.cam.config(width=PW, height=PH)
        else:
            self.plot1_label.pack(anchor="w", before=self.plot2_label_anchor)
            self.plot1.pack(before=self.plot2_label_anchor)
            self.analysis_box.pack(anchor="w", pady=(6, 0))
            self.log.config(height=14)
        self.root.update_idletasks()
        self.scroller._toggle_bars()

    def _auto_compact(self):
        """Switch to compact mode on its own when the window is clearly too small."""
        try:
            need_h = self.root.winfo_reqheight()
            have_h = self.window.winfo_screenheight() - 110
            if need_h > have_h and not self.v_compact.get():
                self.v_compact.set(True)
                self._apply_compact()
                self.window.update_idletasks()
                fit_window(self.window, self.root.winfo_reqwidth() + 24, self.root.winfo_reqheight() + 8)
        except Exception:
            pass

    # ---------------------------------------------------------- controls
    def _controls(self):
        outer = ttk.Frame(self.root)
        outer.grid(row=0, column=0, columnspan=3, sticky="ew", padx=6, pady=6)
        f = ttk.Frame(outer)
        f.pack(anchor="w")
        f2 = ttk.Frame(outer)
        f2.pack(anchor="w", pady=(5, 0))
        self.ctl_row2 = f2
        ttk.Label(f, text="Camera:").pack(side="left")
        self.s_cam = ttk.Spinbox(f, from_=0, to=9, width=3)
        self.s_cam.set(0)
        self.s_cam.pack(side="left", padx=(2, 12))

        def scale(label, key, lo, hi, init):
            ttk.Label(f, text=label).pack(side="left")
            s = ttk.Scale(f, from_=lo, to=hi, value=self.ctl[key], length=90,
                          command=lambda v, key=key: (self.ctl.update({key: int(float(v))}),
                                                      self._refresh_analysis()))
            s.pack(side="left", padx=(2, 10))
        scale("pixel diff >", "thr", 4, 60, 4)
        scale("min moving px", "min_px", 5, 800, 30)
        scale("k frames", "k", 1, 30, 5)

        f3 = ttk.LabelFrame(outer, text="Time code tuning (live)")
        f3.pack(anchor="w", fill="x", pady=(6, 0))
        f4 = ttk.LabelFrame(outer, text="Circuit / detector tuning (live)")
        f4.pack(anchor="w", fill="x", pady=(4, 0))
        f5 = ttk.LabelFrame(outer, text="Instance x, Driver A / B and gate (live)")
        f5.pack(anchor="w", fill="x", pady=(4, 0))
        f6 = ttk.LabelFrame(outer, text="Timing and camera")
        f6.pack(anchor="w", fill="x", pady=(4, 0))
        fq = ttk.LabelFrame(outer, text="Q flip-flop, fault injection and error correction (live)")
        fq.pack(anchor="w", fill="x", pady=(4, 0))
        self.tune_vars = {}

        def spin(parent, label, key, lo, hi, step, integer, width=6):
            ttk.Label(parent, text=label).pack(side="left", padx=(6, 1))
            default = TUNE_DEFAULTS[key]
            var = tk.StringVar(value=str(int(default) if integer else round(default, 4)))
            self.tune_vars[key] = (var, default, integer)

            def changed(*_):
                try:
                    v = float(var.get())
                except ValueError:
                    return                                   # half-typed: keep the last good value
                v = min(max(v, lo), hi)
                self.ctl[key] = int(round(v)) if integer else v
                self._refresh_analysis()
            var.trace_add("write", changed)
            ttk.Spinbox(parent, from_=lo, to=hi, increment=step, width=width,
                        textvariable=var).pack(side="left")

        spin(f3, "frames per bin", "bin_w", 1, 120, 1, True, 4)
        spin(f3, "phase (start bin)", "phase", 0, 255, 1, True, 4)
        ttk.Label(f3, text="clock:").pack(side="left", padx=(10, 1))
        self.v_clock = tk.StringVar(value="frames")
        cb = ttk.Combobox(f3, textvariable=self.v_clock, values=("frames", "ms"), width=7,
                          state="readonly")
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>",
                lambda _e: (self.ctl.update(clock=self.v_clock.get()), self._refresh_analysis()))
        spin(f3, "ms per bin", "bin_ms", 5, 5000, 5, False, 6)
        self.v_invert = tk.BooleanVar(value=False)
        ttk.Checkbutton(f3, text="invert Y", variable=self.v_invert,
                        command=lambda: (self.ctl.update(invert=self.v_invert.get()),
                                         self._refresh_analysis())).pack(side="left", padx=(10, 0))
        spin(f3, "bins added per find", "bins_per_hit", 1, 20, 1, True, 3)

        spin(f4, "RC (ms)", "rc_ms", 0.1, 100, 0.5, False, 6)
        spin(f4, "node tau (ms)", "tau_ms", 0.1, 100, 0.5, False, 6)
        spin(f4, "node thr", "node_thr", 0.05, 0.95, 0.05, False, 5)
        spin(f4, "Vs", "vs", 1.0, 15.0, 0.1, False, 5)
        spin(f4, "V_BE", "vbe", 0.0, 2.0, 0.05, False, 5)
        spin(f4, "V_IH", "vih", 0.5, 12.0, 0.1, False, 5)
        spin(f4, "warm-up frames", "warmup", 0, 200, 1, True, 4)
        spin(f4, "blur kernel", "blur", 1, 31, 2, True, 3)
        ttk.Button(f4, text="Reset tuning", command=self._reset_tuning).pack(side="left", padx=10)

        spin(f5, "x = px /", "x_div", 1, 50, 1, True, 3)
        spin(f5, "+", "x_off", 0, 1000, 1, True, 4)
        spin(f5, "diff gap (frames)", "diff_gap", 1, 30, 1, True, 3)
        spin(f5, "dilate", "dilate", 0, 6, 1, True, 3)
        spin(f5, "A witness rounds", "a_rounds", 1, 16, 1, True, 3)
        spin(f5, "B needs x >", "b_min_x", 0, 100000, 1, True, 6)
        self.v_bodd = tk.BooleanVar(value=True)
        ttk.Checkbutton(f5, text="B needs odd x", variable=self.v_bodd,
                        command=lambda: (self.ctl.update(b_odd=self.v_bodd.get()),
                                         self._refresh_analysis())).pack(side="left", padx=(8, 0))
        ttk.Label(f5, text="gate:").pack(side="left", padx=(10, 1))
        self.v_gate = tk.StringVar(value="AND")
        gcb = ttk.Combobox(f5, textvariable=self.v_gate, width=7, state="readonly",
                           values=("AND", "OR", "A only", "B only"))
        gcb.pack(side="left")
        gcb.bind("<<ComboboxSelected>>",
                 lambda _e: (self.ctl.update(gate=self.v_gate.get()), self._refresh_analysis()))

        spin(f6, "max fps (0 = free)", "max_fps", 0, 240, 1, True, 4)
        spin(f6, "dt clamp (ms)", "dt_max_ms", 10, 2000, 10, True, 5)
        spin(f6, "max pending bins (0 = no cap)", "pending_cap", 0, 1000, 1, True, 4)
        ttk.Label(f6, text="resolution (next Start):").pack(side="left", padx=(10, 1))
        self.v_res = tk.StringVar(value="640x480")
        rcb = ttk.Combobox(f6, textvariable=self.v_res, width=9, state="readonly",
                           values=("320x240", "640x480", "1280x720", "1920x1080"))
        rcb.pack(side="left")
        rcb.bind("<<ComboboxSelected>>", lambda _e: self.ctl.update(cam_res=self.v_res.get()))

        def qspin(label, key, lo, hi, step, integer=False, width=5):
            ttk.Label(fq, text=label).pack(side="left", padx=(6, 1))
            var = tk.StringVar(value=str(self.ctl.get(key, TUNE_DEFAULTS.get(key, 0))))
            self.q_vars[key] = var
            def qchanged(*_):
                try:
                    val = float(var.get())
                except ValueError:
                    return
                val = max(lo, min(hi, val))
                self.ctl[key] = int(round(val)) if integer else val
            var.trace_add("write", qchanged)
            ttk.Spinbox(fq, from_=lo, to=hi, increment=step, width=width,
                        textvariable=var).pack(side="left")

        self.q_vars = {}
        ttk.Label(fq, text="Latch:").pack(side="left", padx=(6, 1))
        self.v_q_latch = tk.StringVar(value="D")
        ttk.Combobox(fq, textvariable=self.v_q_latch, values=("D", "T", "JK", "SR"),
                     width=5, state="readonly").pack(side="left")
        self.v_q_latch.trace_add("write", lambda *_: self.ctl.update(q_latch=self.v_q_latch.get()))
        ttk.Label(fq, text="ECC:").pack(side="left", padx=(6, 1))
        self.v_q_ecc = tk.StringVar(value="none")
        ttk.Combobox(fq, textvariable=self.v_q_ecc,
                     values=("none", "repetition-3", "repetition-5", "repetition-7", "Hamming(7,4)"),
                     width=14, state="readonly").pack(side="left")
        self.v_q_ecc.trace_add("write", lambda *_: self.ctl.update(q_ecc=self.v_q_ecc.get()))
        ttk.Label(fq, text="fault mode:").pack(side="left", padx=(6, 1))
        self.v_q_mode = tk.StringVar(value="random")
        ttk.Combobox(fq, textvariable=self.v_q_mode,
                     values=("random", "periodic", "burst", "force-0", "force-1"),
                     width=9, state="readonly").pack(side="left")
        self.v_q_mode.trace_add("write", lambda *_: self.ctl.update(q_flip_mode=self.v_q_mode.get()))
        qspin("flip p", "q_flip_prob", 0, 1, 0.01, False, 5)
        qspin("period", "q_flip_period", 1, 10000, 1, True, 5)
        qspin("burst len", "q_burst_len", 1, 1000, 1, True, 5)
        qspin("seed", "q_seed", 0, 2147483647, 1, True, 8)
        self.v_q_inv = tk.BooleanVar(value=False)
        ttk.Checkbutton(fq, text="invert Q", variable=self.v_q_inv,
                        command=lambda: self.ctl.update(q_invert=self.v_q_inv.get())).pack(side="left", padx=4)
        self.v_q_reset = tk.BooleanVar(value=False)
        ttk.Checkbutton(fq, text="reset", variable=self.v_q_reset,
                        command=lambda: self.ctl.update(q_reset=self.v_q_reset.get())).pack(side="left", padx=4)
        ttk.Button(fq, text="Reset Q state", command=lambda: self.ctl.update(_q_prev=0)).pack(side="left", padx=4)

        ttk.Label(f2, text="Time code Y:").pack(side="left")
        self.e_y = ttk.Entry(f2, width=22)
        self.e_y.insert(0, "1111")
        self.e_y.pack(side="left", padx=(2, 6))
        self.e_y.bind("<KeyRelease>", self._y_changed)
        self.l_y = ttk.Label(f2, text="", font=("Consolas", 9), width=26)
        self.l_y.pack(side="left")
        ttk.Label(f2, text="ops:  |  OR-add    &  AND    +  integer add",
                  foreground="#666").pack(side="left", padx=(0, 10))
        self.v_force = tk.BooleanVar(value=False)
        ttk.Checkbutton(f2, text="'forces HIGH' as OR (breaks one-sidedness)",
                        variable=self.v_force,
                        command=lambda: (self.ctl.update(force=self.v_force.get()),
                                         self._refresh_analysis())).pack(side="left")
        self.b_start = ttk.Button(f, text="Start circuit", command=lambda: self.start(True))
        self.b_start.pack(side="left", padx=2)
        self.b_prev = ttk.Button(f, text="Preview only", command=lambda: self.start(False))
        self.b_prev.pack(side="left", padx=2)
        self.b_stop = ttk.Button(f, text="Stop", command=self.stop, state="disabled")
        self.b_stop.pack(side="left", padx=2)
        self.b_an = ttk.Button(f, text="Driver analysis", command=self._open_analysis)
        self.b_an.pack(side="left", padx=(10, 2))
        self.v_ov = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="show motion", variable=self.v_ov,
                        command=lambda: self.ctl.update(overlay=self.v_ov.get())
                        ).pack(side="left", padx=8)

    def _reset_tuning(self):
        for key, (var, default, integer) in self.tune_vars.items():
            var.set(str(int(default) if integer else round(default, 4)))
        self.v_clock.set("frames")
        self.v_invert.set(False)
        self.v_bodd.set(True)
        self.v_gate.set("AND")
        self.v_res.set("640x480")
        self.v_q_latch.set("D")
        self.v_q_ecc.set("none")
        self.v_q_mode.set("random")
        self.v_q_inv.set(False)
        self.v_q_reset.set(False)
        for key, var in self.q_vars.items():
            var.set(str(TUNE_DEFAULTS[key]))
        self.ctl.update(clock="frames", invert=False, b_odd=True, gate="AND", cam_res="640x480",
                        q_latch="D", q_ecc="none", q_flip_mode="random",
                        q_invert=False, q_reset=False, _q_prev=0)
        self._refresh_analysis()

    def _btc_panel(self):
        self._btc_stop = None
        self._btc_q = queue.Queue()
        self._btc_tick_s = 1.0
        self._btc_window_s = 10.0
        panel = ttk.LabelFrame(
            self.root,
            text="Bitcoin HIGH/LOW oracle: continuous up/down samples train a binary tree; "
                 "one end check every n seconds; release only after a correct call"
        )
        panel.grid(row=3, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 6))
        row1 = ttk.Frame(panel)
        row1.pack(fill="x", padx=4, pady=(4, 0))
        row2 = ttk.Frame(panel)
        row2.pack(fill="x", padx=4, pady=(2, 4))

        self.btc_price_var = tk.StringVar(value="BTC/USDT: --")
        ttk.Label(row1, textvariable=self.btc_price_var, width=22,
                  font=("TkDefaultFont", 11, "bold")).pack(side="left", padx=8)
        self.btc_call_var = tk.StringVar(value="Tree call: --")
        ttk.Label(row1, textvariable=self.btc_call_var, width=50,
                  font=("TkDefaultFont", 11, "bold")).pack(side="left", padx=10)

        ttk.Label(row1, text="Sample every").pack(side="left", padx=(8, 2))
        self.btc_tick_var = tk.StringVar(value="1")
        ttk.Spinbox(row1, from_=0.05, to=60, increment=0.5, width=5,
                    textvariable=self.btc_tick_var).pack(side="left")
        ttk.Label(row1, text="s   End check every").pack(side="left", padx=(2, 2))
        self.btc_period_var = tk.StringVar(value="10")
        ttk.Spinbox(row1, from_=1, to=3600, width=5, textvariable=self.btc_period_var).pack(side="left")
        ttk.Label(row1, text="s").pack(side="left", padx=(2, 8))

        ttk.Label(row1, text="Depth").pack(side="left", padx=(4, 2))
        self.btc_depth_var = tk.StringVar(value=str(self.btc.depth))
        dsp = ttk.Spinbox(row1, from_=1, to=BitcoinHighLowOracle.MAX_DEPTH, width=3,
                          textvariable=self.btc_depth_var, command=self._btc_depth_changed)
        dsp.pack(side="left")
        for ev in ("<Return>", "<FocusOut>", "<<Increment>>", "<<Decrement>>"):
            dsp.bind(ev, lambda e: self.root.after_idle(self._btc_depth_changed), add="+")
        ttk.Button(row1, text="Reset tree", command=self._btc_reset_tree).pack(side="left", padx=(8, 2))
        self.btc_auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="Run", variable=self.btc_auto_var,
                        command=self.btc_toggle_auto).pack(side="left", padx=6)

        import types
        sbox, slab = fixed_label(row2, 540, 52, text="", font=("Consolas", 9),
                                 justify="left", wraplength=534)
        sbox.pack(side="left", padx=8)
        rbox, rlab = fixed_label(row2, 480, 52, text="", justify="left", wraplength=474)
        rbox.pack(side="left", padx=12)
        self.btc_stats_var = types.SimpleNamespace(set=lambda t: set_text(slab, t))
        self.btc_result_var = types.SimpleNamespace(set=lambda t: set_text(rlab, t))
        self.btc_stats_var.set("tree: no samples yet")
        self.btc_result_var.set("Tick Run. A release is armed only when the previous window's call "
                                "matched the market move on a factual end check.")
        row3 = ttk.Frame(panel)
        row3.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(row3, text="Keyboard macros (override the pending call, scored at the next end check):"
                  ).pack(side="left", padx=(8, 6))
        self.macro_keys = {"higher": "w", "lower": "s"}
        for call, label in (("higher", "HIGHER"), ("lower", "LOWER")):
            ttk.Label(row3, text=f"Shift +").pack(side="left")
            var = tk.StringVar(value=self.macro_keys[call].upper())

            def changed(*_, call=call, var=var):
                v = var.get().strip()[-1:]                 # one character only
                if v.isalnum():
                    self.macro_keys[call] = v.lower()
                if var.get() != v.upper() and v.isalnum():
                    var.set(v.upper())
            var.trace_add("write", changed)
            ttk.Entry(row3, width=2, textvariable=var, justify="center").pack(side="left", padx=(2, 2))
            ttk.Label(row3, text=f"= call {label}").pack(side="left", padx=(0, 14))
        row4 = ttk.Frame(panel)
        row4.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(row4, text="Inject real keystrokes into the focused window:").pack(side="left", padx=(8, 6))
        self.v_inj_tree = tk.BooleanVar(value=False)
        ttk.Checkbutton(row4, text="on tree calls", variable=self.v_inj_tree,
                        command=lambda: self._inject_toggle("tree")).pack(side="left", padx=4)
        self.v_inj_manual = tk.BooleanVar(value=False)
        ttk.Checkbutton(row4, text="on manual macros", variable=self.v_inj_manual,
                        command=lambda: self._inject_toggle("manual")).pack(side="left", padx=4)
        self.v_inj_change = tk.BooleanVar(value=False)
        ttk.Checkbutton(row4, text="only when the call changes", variable=self.v_inj_change,
                        command=lambda: self.inj.update(change_only=self.v_inj_change.get())
                        ).pack(side="left", padx=4)
        self.v_inj_lastbit = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row4,
            text="Shift+W/S on final bit",
            variable=self.v_inj_lastbit,
            command=lambda: self.ctl.update(
                inj_last_bit=self.v_inj_lastbit.get(),
                macro_higher=self.macro_keys["higher"],
                macro_lower=self.macro_keys["lower"],
                inj_hold_ms=self.inj["hold_ms"],
                inj_scan=self.inj["scan"],
            ),
        ).pack(side="left", padx=4)
        self.v_inj_scan = tk.BooleanVar(value=True)
        ttk.Checkbutton(row4, text="scan codes (games)", variable=self.v_inj_scan,
                        command=lambda: self.inj.update(scan=self.v_inj_scan.get())
                        ).pack(side="left", padx=4)
        for label, key, lo, hi, init in (("delay ms", "delay_ms", 0, 10000, 0), ("hold ms", "hold_ms", 10, 1000, 40)):
            ttk.Label(row4, text=label).pack(side="left", padx=(8, 1))
            v = tk.StringVar(value=str(init))

            def inj_changed(*_, key=key, v=v, lo=lo, hi=hi):
                try:
                    self.inj[key] = int(min(max(float(v.get()), lo), hi))
                except ValueError:
                    pass
            v.trace_add("write", inj_changed)
            ttk.Spinbox(row4, from_=lo, to=hi, increment=10, width=6, textvariable=v).pack(side="left")
        ttk.Button(row4, text="Test inject (3 s)", command=self._inject_test).pack(side="left", padx=10)
        self.window.bind_all("<KeyPress>", self._macro_key, add="+")
        self.window.bind_all("<Button-1>", self._macro_defocus, add="+")
        self.root.after(100, self._btc_poll)

    # ---------------------------------------------------------- keyboard macros
    _TYPING = (tk.Entry, ttk.Entry, tk.Text)

    def _typing_focus(self):
        """True when the keyboard focus is in a widget that is accepting typed text."""
        try:
            w = self.window.focus_get()
        except (KeyError, tk.TclError):
            return False
        if not isinstance(w, self._TYPING):
            return False
        try:
            return str(w.cget("state")) not in ("readonly", "disabled")
        except tk.TclError:
            return True

    def _macro_defocus(self, e):
        """Clicking anywhere that is not a text field hands the keyboard back to the macros."""
        if isinstance(e.widget, str) or isinstance(e.widget, self._TYPING):
            return
        try:
            self.window.focus_set()
        except tk.TclError:
            pass

    def _inject_toggle(self, which):
        var = self.v_inj_tree if which == "tree" else self.v_inj_manual
        if var.get():
            err = injection_backend_error()
            if err:
                messagebox.showerror("Inject keystrokes", err)
                var.set(False)
                return
            if not messagebox.askokcancel(
                    "Inject keystrokes",
                    "This sends REAL keystrokes (Shift + the macro key) to whichever window has "
                    "keyboard focus, including other programs.\n\nMake sure the right window is "
                    "focused and that a stray Shift+W / Shift+S there is harmless. Enable?"):
                var.set(False)
                return
        self.inj[which] = var.get()
        self._log(f"Keystroke injection for {which} calls: {'ON' if var.get() else 'off'}.")

    def _inject_call(self, call, extra_delay_ms=0):
        """Send the macro key for 'higher' / 'lower' to the OS from a worker thread."""
        key = self.macro_keys[call]
        delay = max(self.inj["delay_ms"], extra_delay_ms) / 1000.0
        hold = self.inj["hold_ms"] / 1000.0
        scan = self.inj["scan"]
        # our own window must not react to the key we are about to send (would loop)
        self._inject_guard_until = time.time() + delay + hold + 0.6

        def work():
            try:
                if delay:
                    time.sleep(delay)
                inject_combo(key, True, hold, scan)
                self._btc_q.put(("inj", f"INJECTED Shift+{key.upper()} ({call.upper()}) into the focused window."))
            except Exception as exc:
                self._btc_q.put(("inj", f"INJECT FAILED: {exc}"))
        threading.Thread(target=work, daemon=True).start()

    def _inject_test(self):
        err = injection_backend_error()
        if err:
            messagebox.showerror("Inject keystrokes", err)
            return
        self._log(f"Test: Shift+{self.macro_keys['higher'].upper()} will be sent in 3 s. "
                  f"Click the target window now.")
        self._inject_call("higher", 3000)

    def _maybe_inject(self, kind, call):
        if not self.inj.get(kind) or call not in ("higher", "lower"):
            return
        if self.inj["change_only"] and self._last_injected == call:
            return
        self._last_injected = call
        self._inject_call(call)

    def _macro_key(self, e):
        if time.time() < self._inject_guard_until:            # ignore keys we injected ourselves
            return None
        if not (e.state & 0x1) or (e.state & 0x4):          # needs Shift, and no Ctrl
            return None
        key = e.keysym.lower()
        if key == self.macro_keys["higher"]:
            call = "higher"
        elif key == self.macro_keys["lower"]:
            call = "lower"
        else:
            return None
        if self._typing_focus():
            return None
        price = self.btc.manual_call(call)
        if price is None:
            self.btc_result_var.set("Macro ignored: no BTC price yet. Tick Run first so a sample arrives.")
            return "break"
        self._btc_apply_manual(call, price)
        return "break"

    def _btc_apply_manual(self, call, price):
        shown = f"Tree call: {call.upper()}  [manual, set at ${price:,.2f}]"
        self.btc_call_var.set(shown)
        self.ctl["btc_last_call"] = call.upper()
        self.ctl["btc_last_price"] = price
        self.btc_result_var.set(f"MANUAL CALL {call.upper()} at ${price:,.2f}; it is scored at the next "
                                f"end check, and a correct call arms a release.")
        self._log(f"BTC HIGH/LOW ORACLE MANUAL CALL (Shift+{self.macro_keys[call].upper()}): "
                  f"{call.upper()} at ${price:,.2f}.")
        self._maybe_inject("manual", call)

    def _btc_read_timing(self):
        """Main thread only: copy the Tk settings into plain attributes the sampler thread reads."""
        try:
            tick = max(0.5, float(self.btc_tick_var.get()))
        except (ValueError, tk.TclError):
            tick = self._btc_tick_s
        try:
            win = max(tick, float(self.btc_period_var.get()))
        except (ValueError, tk.TclError):
            win = self._btc_window_s
        self._btc_tick_s = tick
        if win != self._btc_window_s:
            self._btc_window_s = win
            self.btc.set_window(win)

    def _btc_depth_changed(self, _e=None):
        try:
            d = int(self.btc_depth_var.get().strip())
        except (ValueError, tk.TclError):
            return
        d = max(1, min(BitcoinHighLowOracle.MAX_DEPTH, d))
        if d != self.btc.depth:
            self.btc.set_depth(d)
            self._log(f"BTC HIGH/LOW ORACLE: tree depth set to {d}.")

    def _btc_reset_tree(self):
        self.btc.reset_tree()
        self.ctl["btc_release_armed"] = False
        self._btc_show_stats()
        self._log("BTC HIGH/LOW ORACLE: tree, samples and score reset; armed release cancelled.")

    def btc_toggle_auto(self):
        """Run / stop the sampler thread."""
        if self.btc_auto_var.get():
            if self._btc_stop is not None:
                self._btc_stop.set()
            self._btc_read_timing()
            self.btc.set_window(self._btc_window_s)
            self._btc_stop = threading.Event()
            threading.Thread(target=self._btc_sampler, args=(self._btc_stop,), daemon=True).start()
            self._log(f"BTC HIGH/LOW oracle: sampling every {self._btc_tick_s:g} s, "
                      f"end check every {self._btc_window_s:g} s.")
        else:
            if self._btc_stop is not None:
                self._btc_stop.set()
            self._btc_stop = None
            self.ctl["btc_release_armed"] = False
            self._log("BTC HIGH/LOW oracle: sampling stopped; armed release cancelled.")
            self.btc_result_var.set("Stopped. No further oracle releases.")

    def _btc_sampler(self, stop):                  # OFF the Tk thread
        next_end = None
        last_err = 0.0
        while not stop.is_set():
            t0 = time.time()
            try:
                price, factual, a, b = self.btc.fetch_sample()
                self.btc.observe(price, factual, a, b)
                now = time.time()
                if next_end is None or now >= next_end:         # the ONE end check of this window
                    res = self.btc.end_check()
                    next_end = now + self._btc_window_s
                    if res is not None:
                        self._btc_q.put(("end", res))
                else:
                    self._btc_q.put(("tick", price))
            except Exception as exc:
                if time.time() - last_err >= 10:                # do not spam the log every second
                    last_err = time.time()
                    self._btc_q.put(("err", exc))
            stop.wait(max(0.05, self._btc_tick_s - (time.time() - t0)))

    def _btc_poll(self):                           # Tk thread
        self._btc_read_timing()
        last_price = None
        try:
            while True:
                kind, payload = self._btc_q.get_nowait()
                try:
                    if kind == "tick":
                        last_price = payload
                    elif kind == "end":
                        self._btc_apply(*payload)
                        last_price = None
                    elif kind == "inj":
                        self._log(payload)
                    else:
                        self._btc_fail(payload)
                except Exception as exc:
                    self.ctl["btc_release_armed"] = False       # never leave a release armed after an error
                    self._log(f"BTC panel error: {exc!r}")
        except queue.Empty:
            pass
        if last_price is not None:
            self.btc_price_var.set(f"BTC/USDT: ${last_price:,.2f}")
            self._btc_show_stats()
        self.root.after(100, self._btc_poll)

    def _btc_fail(self, exc):
        # a failed sample is simply retried on the next tick; an armed release is kept
        self.btc_result_var.set(f"Binance sample failed: {exc}; retrying.")
        self._log(f"BTC HIGH/LOW ORACLE ERROR: {exc}; retrying.")

    def _btc_show_stats(self):
        s = self.btc.stats()
        rel = self.ctl.get("btc_released", 0)
        rec = (f"{s['correct']}/{s['total']} = {s['rate']:.1%}  P(>= by chance)={s['p_value']:.2f}"
               if s["total"] else "no scored end checks yet")
        self.btc_stats_var.set(
            f"depth {s['depth']}, {s['nodes']} nodes, trained {s['trained']}: {rec}\n"
            f"samples {s['ticks']}  last bits {s['bits']}  no-call {s['abstains']}  "
            f"ties {s['ties']}  released {rel}"
            + (f"  manual {s['m_correct']}/{s['m_total']}" if s["m_total"] else ""))

    def _btc_apply(self, price, result, factual, changed):
        """One END CHECK finished (once per window)."""
        self.btc_price_var.set(f"BTC/USDT: ${price:,.2f}")
        pred = self.btc.last_prediction
        call = pred["guess"]
        node = pred.get("node")
        node_txt = "" if not node else f"  [depth {node[0]}: {node[1]} up / {node[2]} down]"
        self.btc_call_var.set(f"Tree call: {call}{node_txt}")
        self.ctl["btc_last_call"] = call
        self.ctl["btc_last_price"] = price
        self._maybe_inject("tree", call.lower())

        checks = self.btc.last_checks
        a, b = checks["driver_a"], checks["driver_b"]
        change = "price changed" if changed else "price unchanged (tie, not scored)"
        who = "manual" if (result and result.get("manual")) else "tree"

        # THE RULE: release only if the end check is factual AND the previous window's call matched
        arm = bool(factual and result is not None and result["correct"])
        self.ctl["btc_release_armed"] = arm

        if arm:
            self.btc_result_var.set(
                f"{who.upper()} GUESSED RIGHT ({result['guess']} -> {result['actual_direction']}); "
                f"release armed for the next even frame. Next call: {call}.")
            self._log(f"BTC HIGH/LOW ORACLE END CHECK: ${price:,.2f} ({change}); {who} call "
                      f"{result['guess']} was CORRECT; Driver A={a['driver_a']}, "
                      f"Driver B={b['driver_b']}; release armed for next even frame.")
        elif factual:
            if result is None:
                reason = "no scored call (tree had no call, or the price was unchanged)"
            else:
                reason = f"{who} guessed wrong ({result['guess']} -> {result['actual_direction']})"
            self.btc_result_var.set(f"FACTUAL ({change}) but {reason}; no release. Next call: {call}.")
            self._log(f"BTC HIGH/LOW ORACLE END CHECK: ${price:,.2f} ({change}); {reason}; "
                      f"no release armed; next call={call}.")
        else:
            self.btc_result_var.set(
                f"NOT FACTUAL ({change}); Driver B={b['driver_b']}, "
                f"Driver A={a['driver_a']}; no oracle release armed.")
            self._log(f"BTC HIGH/LOW ORACLE REJECTED: ${price:,.2f} ({change}); "
                      f"Driver A={a['driver_a']}, Driver B={b['driver_b']}; no release armed.")

        if result:
            outcome = "CORRECT" if result["correct"] else "WRONG"
            self._log(f"BTC HIGH/LOW ORACLE RESULT: {who} call {result['guess']}; "
                      f"market went {result['actual_direction']}; "
                      f"${result['previous']:,.2f} -> ${result['actual']:,.2f}; "
                      f"{outcome}; record {self.btc.correct}/{self.btc.total}")
        self._btc_show_stats()

    def _miner_panel(self):
        panel = ttk.LabelFrame(self.root, text="Time-code sources")
        panel.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 6))

        left = ttk.Frame(panel, width=380, height=270)
        left.pack_propagate(False)
        left.pack(side="left", anchor="n", padx=8, pady=8)

        self.l_miner = ttk.Label(
            left,
            text="Time codes added: 0 (pending 0)\n"
                 "Armed frames (2 evens in a row): 0\n"
                 "Now: waiting for the circuit",
            font=("Consolas", 9), justify="left", anchor="nw", width=52)
        self.l_miner.pack(anchor="w", pady=(4, 0))

        right = ttk.Frame(panel)
        right.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=8)
        logf = ttk.Frame(panel)
        logf.pack(side="left", fill="y", padx=(0, 8), pady=8)
        ttk.Label(logf, text="Log", font=("Arial", 9, "bold")).pack(anchor="w")
        self.log = tk.Text(logf, height=14, width=58, font=("Consolas", 8), state="disabled",
                           wrap="none")
        self.log.pack(fill="y", expand=True)
        self._hidden_panel(right)
        self._search_panel(right)
        self._factor_panel(right)
        right.pack_forget()                   # oracle-exclusive: these sources are inert, hide them

    def _hidden_panel(self, parent):
        box = ttk.LabelFrame(parent, text="Hidden number (spot check after 2 sequential evens)")
        box.pack(fill="x", pady=(8, 0))
        self.hidden_box = box
        row = ttk.Frame(box)
        row.pack(anchor="w", padx=6, pady=(4, 0))
        self.v_hidden_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Enabled", variable=self.v_hidden_on,
                        command=lambda: self.ctl.update(hidden_on=self.v_hidden_on.get())
                        ).pack(side="left")
        ttk.Label(row, text="Range 0 to").pack(side="left", padx=(12, 2))
        self.v_hrange = tk.StringVar(value=str(HIDDEN_RANGE_DEFAULT))
        self.s_hrange = ttk.Spinbox(row, from_=2, to=HIDDEN_RANGE_MAX, increment=100, width=9,
                                    textvariable=self.v_hrange, command=self._hidden_range_changed)
        self.s_hrange.pack(side="left")
        for ev in ("<Return>", "<FocusOut>", "<<Increment>>", "<<Decrement>>"):
            self.s_hrange.bind(ev, lambda e: self.root.after_idle(self._hidden_range_changed), add="+")
        ttk.Button(row, text="New secret",
                   command=self._hidden_new_secret).pack(side="left", padx=(12, 2))
        ttk.Button(row, text="Reveal", command=self._hidden_reveal).pack(side="left", padx=2)
        ttk.Label(row, text="Freq Hz").pack(side="left", padx=(12, 2))
        self.v_hhz = tk.StringVar(value=f"{GUESS_HZ_DEFAULT:g}")
        self.s_hhz = ttk.Spinbox(row, from_=GUESS_HZ_LIMITS[0], to=GUESS_HZ_LIMITS[1],
                                 increment=1, width=7, textvariable=self.v_hhz,
                                 command=self._hidden_hz_changed)
        self.s_hhz.pack(side="left")
        for ev in ("<Return>", "<FocusOut>", "<<Increment>>", "<<Decrement>>"):
            self.s_hhz.bind(ev, lambda e: self.root.after_idle(self._hidden_hz_changed), add="+")
        hbox, self.l_hidden = fixed_label(box, 480, 92, text="", font=("Consolas", 9),
                                          justify="left", wraplength=480)
        hbox.pack(anchor="w", padx=6, pady=(2, 4))
        set_text(self.l_hidden, "clock idle")

    def _search_panel(self, parent):
        box = ttk.LabelFrame(parent, text="Higher/lower search (binary, 1 guess per 2 sequential evens)")
        box.pack(fill="x", pady=(8, 0))
        self.search_box = box
        row = ttk.Frame(box)
        row.pack(anchor="w", padx=6, pady=(4, 0))
        self.v_search_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Enabled", variable=self.v_search_on,
                        command=lambda: self.ctl.update(search_on=self.v_search_on.get())
                        ).pack(side="left")
        ttk.Label(row, text="Secret bits").pack(side="left", padx=(12, 2))
        self.v_sbits = tk.StringVar(value=str(SEARCH_BITS_DEFAULT))
        self.s_sbits = ttk.Spinbox(row, from_=2, to=SEARCH_BITS_MAX, increment=8, width=6,
                                   textvariable=self.v_sbits, command=self._search_bits_changed)
        self.s_sbits.pack(side="left")
        for ev in ("<Return>", "<FocusOut>", "<<Increment>>", "<<Decrement>>"):
            self.s_sbits.bind(ev, lambda e: self.root.after_idle(self._search_bits_changed), add="+")
        ttk.Button(row, text="New secret", command=self._search_new_secret).pack(side="left", padx=(12, 2))
        ttk.Button(row, text="Reveal", command=self._search_reveal).pack(side="left", padx=2)
        sbox, self.l_search = fixed_label(box, 480, 106, text="", font=("Consolas", 9),
                                          justify="left", wraplength=480)
        sbox.pack(anchor="w", padx=6, pady=(2, 4))
        set_text(self.l_search, "waiting for two sequential even states")

    def _search_bits_changed(self, _e=None):
        try:
            n = int(self.v_sbits.get().strip())
        except (ValueError, tk.TclError):
            return
        n = max(2, min(SEARCH_BITS_MAX, n))
        if n != self.search.bits:
            self.search.set_bits(n)
            self._show_search(self.search.snapshot())
            self._log(f"# search: {n}-bit secret, new secret drawn, counts reset")

    def _search_new_secret(self):
        self.search.new_secret()
        self._show_search(self.search.snapshot())
        self._log("# search: new secret drawn, counts reset")

    def _search_reveal(self):
        v = self.search.reveal()
        self._log(f"# search secret revealed: {short_int(v, 40)}")
        self._show_search(self.search.snapshot())

    def _show_search(self, h):
        if not h:
            return
        last = "none yet" if h["last"] is None else f"{h['last'][0]}"
        avg = "n/a" if h["avg"] is None else f"{h['avg']:.1f}"
        sec = "hidden" if h["secret"] is None else h["secret"]
        lf = h["last_found"]
        found = ("none yet" if lf is None else
                 f"{short_int(lf[0], 16)} in {lf[1]} guesses")
        set_text(self.l_search, (
            f"secret: {h['bits']} bits   blind guessing would need ~2^{h['bits']} tries\n"
            f"guesses: {h['guesses']} total, {h['round_guesses']} on this secret   "
            f"found: {h['solves']}   avg guesses/found: {avg}\n"
            f"pooled: {h['pooled_n']}/{h['bits']} bits   final checks: {h['final_checks']}   last: {last}\n"
            f"last FOUND number: {found}\n"
            f"current secret: {sec}"))

    def _factor_panel(self, parent):
        box = ttk.LabelFrame(parent, text="Factoriser (higher/lower bit search; time code only after a verified factorisation)")
        box.pack(fill="x", pady=(8, 0))
        row = ttk.Frame(box)
        row.pack(anchor="w", padx=6, pady=(4, 0))
        self.v_fs_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Enabled", variable=self.v_fs_on,
                        command=lambda: self.ctl.update(fsearch_on=self.v_fs_on.get())).pack(side="left")
        ttk.Label(row, text="N =").pack(side="left", padx=(12, 2))
        self.e_fn = ttk.Entry(row, width=46)
        self.e_fn.pack(side="left")
        self.e_fn.bind("<Return>", lambda e: self._factor_go())
        ttk.Button(row, text="Factor N", command=self._factor_go).pack(side="left", padx=(8, 2))
        self.v_fhl = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="high/low only", variable=self.v_fhl).pack(side="left", padx=(4, 2))
        ttk.Button(row, text="Cancel", command=self._factor_cancel).pack(side="left", padx=2)
        ttk.Button(row, text="Save numbers", command=self._factor_save).pack(side="left", padx=2)
        row2 = ttk.Frame(box)
        row2.pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Label(row2, text="or random semiprime, bits").pack(side="left")
        self.v_fbits = tk.StringVar(value=str(FACTOR_BITS_DEFAULT))
        ttk.Spinbox(row2, from_=16, to=FACTOR_BITS_MAX, increment=64, width=6,
                    textvariable=self.v_fbits).pack(side="left", padx=(4, 6))
        ttk.Button(row2, text="Generate", command=self._factor_generate).pack(side="left", padx=2)
        ttk.Button(row2, text="Reveal factor", command=self._factor_reveal).pack(side="left", padx=2)
        fbox, self.l_factor = fixed_label(box, 480, 118, text="", font=("Consolas", 9),
                                          justify="left", wraplength=480)
        fbox.pack(anchor="w", padx=6, pady=(2, 4))
        set_text(self.l_factor, "type N and press Factor N, or generate a semiprime")

    def _factor_go(self):
        try:
            n = parse_number(self.e_fn.get())
        except ValueError as e:
            messagebox.showerror("Number", str(e))
            return
        if self.v_fhl.get():
            self.fsearch.search_number(n)
        else:
            self.fsearch.factor_number(n)

    def _factor_generate(self):
        try:
            bits = int(self.v_fbits.get().strip())
        except ValueError:
            return
        self.fsearch.generate(bits)

    def _factor_save(self):
        data = self.fsearch.export()
        if not data["history"] and not data["current"]["n"]:
            messagebox.showinfo("Save numbers", "No numbers yet: type N and press Factor N, or Generate.")
            return
        path = filedialog.asksaveasfilename(
            title="Save numbers", defaultextension=".json", initialfile="factor_numbers.json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            messagebox.showerror("Save numbers", str(e))
            return
        self._log(f"# saved {len(data['history'])} number record(s) to {path}")

    def _factor_cancel(self):
        self.fsearch.cancel()

    def _factor_reveal(self):
        self.fsearch.reveal()
        v = self.fsearch.snapshot()["secret"]
        self._log(f"# factor revealed: {v}" if v else "# factor not known yet")

    def _factor_tick(self):
        for s in self.fsearch.drain():
            self._log(s)
        self._show_factor(self.fsearch.snapshot())
        self.root.after(300, self._factor_tick)

    def _show_factor(self, h):
        st = {"idle": "idle", "generating": "building a semiprime", "factoring": "ENGINE RUNNING: oracle not ready",
              "ready": "READY: pooling bits on armed frames", "solved": "SOLVED and verified",
              "prime": "N is PRIME: nothing to factor, no oracle (type a composite)", "failed": "failed", "cancelled": "cancelled"}.get(h["state"], h["state"])
        nn = "-" if not h["n"] else short_dec(h["n"], 12) + f", {h['n'].bit_length()} bit"
        lines = [f"state: {st}",
                 f"N: {nn}   mode: {h['mode'] or '-'}",
                 f"pooled {h['pooled_n']}/{h['width']} bits   queries {h['guesses']}   "
                 f"final checks {h['final_checks']}   verified solves {h['solves']}"]
        if h["progress"]:
            lines.append("engine: " + h["progress"])
        elif h["last"]:
            lines.append("last: " + h["last"])
        lf = h["last_found"]
        if lf:
            lines.append(f"p = {short_dec(lf[0], 12)}   q = {short_dec(lf[1], 12)}   ({lf[2]} queries)")
        if h["secret"]:
            lines.append("oracle factor: " + h["secret"])
        set_text(self.l_factor, "\n".join(lines))

    def _hidden_range_changed(self, _e=None):
        try:
            n = int(self.v_hrange.get().strip())
        except (ValueError, tk.TclError):
            return
        n = max(2, min(HIDDEN_RANGE_MAX, n))
        if n != self.hidden.size:
            self.hidden.set_range(n)
            self._show_hidden(self.hidden.snapshot())
            self._log(f"# hidden number: range 0..{n - 1}, new secret drawn, counts reset")

    def _hidden_hz_changed(self, _e=None):
        try:
            hz = float(self.v_hhz.get().strip())
        except (ValueError, tk.TclError):
            return
        lo, hi = GUESS_HZ_LIMITS
        hz = max(lo, min(hi, hz))
        if hz != self.hidden.target_hz:
            self.hidden.set_hz(hz)
            self._log(f"# hidden number: clock frequency set to {hz:g} Hz")

    def _hidden_new_secret(self):
        self.hidden.new_secret()
        self._show_hidden(self.hidden.snapshot())      # show the zeroed counts at once
        self._log("# hidden number: new secret drawn, counts and luck reset")

    def _hidden_reveal(self):
        v = self.hidden.reveal()
        self._log(f"# hidden number revealed: {v}")

    def _show_hidden(self, h):
        if not h:
            return
        lc = h["last_check"]
        last = "none yet" if lc is None else f"candidate {lc[0]} -> {'HIT' if lc[1] else 'miss'}"
        sec = "hidden" if h["secret"] is None else str(h["secret"])
        if h["luck"] is None:
            luck = "luck factor: n/a (no spot checks yet)"
        else:
            luck = (f"luck factor: {h['luck']:.0f}%  (100% = average)   "
                    f"P(>= {h['hits']} hits by chance) = {h['luck_tail']:.1%}")
        lf = h["last_found"]
        found = "none yet" if lf is None else f"{lf[0]} (guessed on check #{lf[1]})"
        set_text(self.l_hidden, (
            f"clock: candidate {h['candidate']:>7d} of 0..{h['size'] - 1}   {h['hz']:7.1f} Hz (set)\n"
            f"spot checks: {h['checks']}   hits: {h['hits']}   expected ~{h['expected']:.2f}\n"
            f"{luck}\n"
            f"last check: {last}   secret: {sec}\n"
            f"last FOUND number: {found}"))

    def _show_miner(self, m):
        self._show_hidden(m.get("hidden"))
        self._show_search(m.get("search"))
        set_text(self.l_miner, (
            f"Time codes added: {m['timecodes_added']} (hidden-number: {m['hidden_timecodes']}, "
            f"search: {m['search_timecodes']}, factor: {m['factor_timecodes']}, "
            f"pending {m['timecode_pending']})\n"
            f"Armed frames (2 evens in a row): {m['armed_frames']}\n"
            f"{'ARMED' if m['last_armed'] else 'not armed'}"
            f"{'' if m['circuit'] else '  [circuit OFF: press Start circuit]'}"))

    def _y_changed(self, _e=None):
        try:
            y, info = parse_code(self.e_y.get())
        except ValueError as err:
            self.l_y.config(text=f"invalid: {err}", foreground="#c33")
            self.y_ok = False
            return
        self.y_ok = True
        self.ctl["y"] = y
        self.code_info = info
        note = ""
        if info["carries"]:
            note = f"  ({info['carries']} carries" + (", top wrapped)" if info["wrapped"] else ")")
        self.l_y.config(text=f"= {y}{note}", foreground="#222")
        self._refresh_analysis()

    def _open_analysis(self):
        if self.an_win is not None:
            self.an_win.lift()
            return
        w = tk.Toplevel(self.root)
        w.title("Driver analysis: predictive properties of A, B and Q")
        self.an_label = ttk.Label(w, text="", font=("Consolas", 9), justify="left")
        self.an_label.pack(padx=10, pady=8, anchor="w")
        ttk.Button(w, text="Reset counters",
                   command=lambda: self.ctl.update(reset_an=True)).pack(pady=(0, 8))
        self.an_win = w
        w.protocol("WM_DELETE_WINDOW", self._close_analysis)
        self._show_analysis()

    def _close_analysis(self):
        if self.an_win is not None:
            self.an_win.destroy()
        self.an_win = None

    def _show_analysis(self):
        if self.an_win is not None and self.an_win.winfo_exists():
            set_text(self.an_label, "\n".join(self.an_lines))

    def _refresh_analysis(self, S=None):
        """Show what the time code does to the error bound (Thm 1-4)."""
        if not hasattr(self, "analysis"):
            return
        k = self.ctl["k"]
        Y0 = self.ctl["y"]
        Y = effective_code(self.ctl)
        m, ones = len(Y), Y.count("1")
        L = [f"Time code Y = {Y0}   open {ones}/{m}  (density {ones / m:.2f})"]
        if self.ctl.get("clock") == "ms":
            L.append(f"bins of {self.ctl['bin_ms']:g} ms  (~{m // max(1, len(Y0))} frames/bin at "
                     f"{(self.ctl.get('fps_est') or 30):.0f} fps)")
        elif self.ctl.get("bin_w", 1) > 1:
            L.append(f"each bin lasts {self.ctl['bin_w']} frames")
        c_ = self.ctl
        if c_.get("gate", "AND") != "AND" or c_.get("b_min_x", 2) != 2 or not c_.get("b_odd", True):
            L.append("NOTE: gate / Driver B rule changed: Thm 1-4 no longer apply as stated")
        if c_.get("a_rounds", 1) > 1:
            L.append(f"A uses {c_['a_rounds']} witnesses/frame: effective p = 1-(1-p)^{c_['a_rounds']}")
        if self.ctl.get("phase") or self.ctl.get("invert"):
            L.append(f"phase {self.ctl.get('phase', 0)}" + ("  inverted" if self.ctl.get("invert") else ""))
        ci = self.code_info
        if ci["carries"]:
            L.append(f"integer add: {ci['carries']} carry(ies), each loses one open bin (Thm 3c)")
        if self.ctl.get("force"):
            L += ["FORCE mode: D = (A.B) OR Y.", "A forced bin gives Q=1 for ANY x, so x not in L",
                  "is accepted: the one-sided guarantee is lost (Thm 4).",
                  "Watch 'false accepts' in the stats."]
        else:
            nmin, nmax = window_stats(Y, k)
            L.append(f"window k={k}: open bins worst N={nmin}, best {nmax}")
            L.append("false accept: impossible for any Y (Thm 1)")
            if nmin == 0:
                L.append("false reject: some window has NO open bin, no bound")
            else:
                L.append(f"false reject <= (1-p)^N; p=1/2: 2^-{nmin} = {0.5 ** nmin:.2g}")
                if S and S["open_inL"] >= 20:
                    ph = S["acc_open_inL"] / S["open_inL"]
                    L.append(f"measured p={ph:.3f} (n={S['open_inL']}): (1-p)^N = {(1 - ph) ** nmin:.2g}")
            fn = frames_needed(Y, 20)
            L.append(f"frames for error <= 2^-20 at p=1/2: {fn if fn else 'never'}")
        set_text(self.analysis, "\n".join(L))

    # ---------------------------------------------------------- diagram
    def _box(self, key, x0, y0, x1, y1, title, sub=""):
        c = self.c
        r = c.create_rectangle(x0, y0, x1, y1, fill=OFF, outline="#666", width=1.5)
        c.create_text((x0 + x1) / 2, y0 + 11, text=title, font=("Arial", 9, "bold"))
        s = c.create_text((x0 + x1) / 2, y1 - 11, text=sub, font=("Arial", 8), fill="#444")
        self.boxes[key] = (r, s)

    def _wire(self, key, *pts, arrow=False):
        w = self.c.create_line(*pts, fill=W_OFF, width=2, arrow="last" if arrow else None)
        self.wires.setdefault(key, []).append(w)

    def _diagram(self):
        c = self.c
        c.create_text(250, 10, text="USB camera  \u2192  frame difference (motion)",
                      font=("Arial", 9, "bold"), fill="#a33")
        c.create_text(90, 24, text="Detector B: motion mask", font=("Arial", 8))
        c.create_text(250, 24, text="camera + motion", font=("Arial", 8))
        c.create_text(410, 24, text="Detector A: motion energy", font=("Arial", 8))
        self.img_b = c.create_image(26, 34, anchor="nw")
        self.img_c = c.create_image(186, 34, anchor="nw")
        self.img_a = c.create_image(346, 34, anchor="nw")
        c.create_line(186, 82, 154, 82, arrow="last", fill="#d9480f")
        c.create_line(314, 82, 346, 82, arrow="last", fill="#d9480f")

        self._box("drvB", 30, 180, 150, 230, "Driver B", "deterministic")
        self._box("drvA", 350, 180, 470, 230, "Driver A", "randomized")
        self.x_text = c.create_text(250, 198, text="x = 0", font=("Arial", 12, "bold"))
        c.create_text(250, 218, text="moving pixels", font=("Arial", 7), fill="#666")
        c.create_line(170, 205, 150, 205, arrow="last")
        c.create_line(330, 205, 350, 205, arrow="last")
        c.create_line(90, 130, 90, 180, arrow="last")
        c.create_line(410, 130, 410, 180, arrow="last")

        self._box("link", 30, 262, 190, 332, "IF/THEN  (C + NPN)", "V(C)=0.00")
        self._wire("B", 90, 230, 90, 262, arrow=True)
        self._box("or", 300, 262, 370, 296, "OR")
        self._box("and", 300, 310, 370, 344, "AND")
        self._wire("A", 410, 230, 410, 247, 285, 247, 285, 319)
        self._wire("A", 285, 272, 300, 272)
        self._wire("A", 285, 319, 300, 319)
        self._wire("B", 190, 297, 240, 297, 240, 286, 300, 286)
        self._wire("B", 240, 297, 240, 336, 300, 336)
        self._wire("gate", 370, 279, 392, 279, 392, 327, 370, 327)
        self._wire("gate", 392, 303, 415, 303)
        self._box("node", 415, 280, 515, 330, "10\u00b5F node", "0.00")
        self._wire("node", 465, 330, 465, 352, arrow=True)
        self._box("Y", 415, 352, 515, 400, "Y time code", "")
        self._wire("D", 515, 376, 540, 376, 540, 262, 560, 262, arrow=True)
        self._box("ff", 560, 232, 660, 332, "74HC74", "")
        c.create_text(575, 262, text="D", font=("Arial", 8))
        c.create_text(575, 312, text="CLK", font=("Arial", 8))
        self._box("clk", 560, 352, 660, 400, "Clock", "= camera frame")
        c.create_line(610, 352, 610, 332, arrow="last")
        c.create_text(676, 247, text="SELECT", font=("Arial", 8, "bold"))
        self.lamp = c.create_oval(668, 258, 686, 276, fill="#ddd", outline="#555")
        c.create_text(676, 290, text="Q", font=("Arial", 8))
        c.create_text(676, 312, text="Q\u0304", font=("Arial", 8), fill="#666")

    def _lit(self, key, on, sub=None, color=ON):
        r, s = self.boxes[key]
        fill = color if on else OFF
        if self._photo_cache.get(("fill", key)) != fill:
            self._photo_cache[("fill", key)] = fill
            self.c.itemconfig(r, fill=fill)
        if sub is not None and self._photo_cache.get(("sub", key)) != sub:
            self._photo_cache[("sub", key)] = sub
            self.c.itemconfig(s, text=sub)

    def _wire_state(self, key, on):
        on = bool(on)
        if self._photo_cache.get(("wire", key)) is on:
            return
        self._photo_cache[("wire", key)] = on
        for w in self.wires.get(key, []):
            self.c.itemconfig(w, fill=W_ON if on else W_OFF)

    # ---------------------------------------------------------- right panel
    def _right_panel(self):
        f = ttk.Frame(self.root)
        f.grid(row=1, column=1, sticky="n", padx=6, pady=4)
        self.right_col = f
        self.plot1_label = ttk.Label(f, text="IF/THEN link during the latest frame",
                                     font=("Arial", 9, "bold"))
        self.plot1_label.pack(anchor="w")
        self.plot1 = tk.Canvas(f, width=420, height=170, bg="white",
                               highlightthickness=1, highlightbackground="#bbb")
        self.plot1.pack()
        self.plot2_label = ttk.Label(f, text="Moving pixels per frame (x)",
                                     font=("Arial", 9, "bold"))
        self.plot2_label.pack(anchor="w", pady=(6, 0))
        self.plot2_label_anchor = self.plot2_label
        self.plot2 = tk.Canvas(f, width=420, height=90, bg="white",
                               highlightthickness=1, highlightbackground="#bbb")
        self.plot2.pack()
        ttk.Label(f, text="Q per frame: green=1, grey=0, hollow=no motion, red edge=Y blocked",
                  font=("Arial", 8, "bold")).pack(anchor="w", pady=(6, 0))
        self.strip = tk.Canvas(f, width=420, height=34, bg="white",
                               highlightthickness=1, highlightbackground="#bbb")
        self.strip.pack()
        vbox, self.verdict = fixed_label(f, 420, 44, text="Idle", font=("Arial", 11, "bold"),
                                         wraplength=420)
        vbox.pack(anchor="w", pady=(6, 0))
        sbox, self.stats = fixed_label(f, 420, 96, text="", font=("Consolas", 9),
                                       justify="left", wraplength=420)
        sbox.pack(anchor="w")

    def _draw_traj(self):
        p = self.plot1
        key = ("traj", id(self.last_traj), len(self.last_traj), round(self.ctl["min_px"]),
               self.ctl["vs"], self.ctl["vbe"], self.ctl["vih"], self.ctl["rc_ms"])
        if self._photo_cache.get("traj_key") == key:
            return
        self._photo_cache["traj_key"] = key
        p.delete("all")
        W, H, m = 420, 170, 28
        top, bot = 8, H - 20
        vs_, vbe_, vih_ = self.ctl["vs"], self.ctl["vbe"], self.ctl["vih"]
        vmax = max(vs_, 5.0) * 1.06
        sy = lambda v: bot - v / vmax * (bot - top)
        for v in range(int(vmax) + 1):
            p.create_line(m, sy(v), W - 6, sy(v), fill="#eee")
            p.create_text(14, sy(v), text=str(v), font=("Arial", 7))
        for v, col, lab in ((vbe_, "#888", f"{vbe_:g} V NPN on"), (vih_, "#c33", f"{vih_:g} V logic HIGH")):
            p.create_line(m, sy(v), W - 6, sy(v), fill=col, dash=(4, 3))
            p.create_text(W - 8, sy(v) - 6, text=lab, anchor="e", font=("Arial", 7), fill=col)
        tr = self.last_traj
        if len(tr) > 1:
            tmax = tr[-1][0]
            head = vs_ - vih_ - vbe_
            lt_txt = (f"{self.ctl['rc_ms'] * math.log(vs_ / head):.1f} ms" if head > 0
                      else "never (Vs - V_BE < V_IH)")
            sx = lambda t: m + t / tmax * (W - m - 6)
            for k, col, lab in ((1, "#1f6fb5", "V(C)"), (2, "#2a9d3a", "V(B)")):
                pts = []
                for row in tr:
                    pts += [sx(row[0]), sy(row[k])]
                p.create_line(*pts, fill=col, width=2)
            p.create_text(m + 6, H - 8, text="\u25cf V(C)", anchor="w", fill="#1f6fb5", font=("Arial", 8))
            p.create_text(m + 70, H - 8, text="\u25cf V(B)", anchor="w", fill="#2a9d3a", font=("Arial", 8))
            p.create_text(W - 8, H - 8, text=f"frame period {tmax:.0f} ms; time to logic HIGH = {lt_txt}",
                          anchor="e", font=("Arial", 8), fill="#555")

    def _draw_x(self):
        p = self.plot2
        p.delete("all")
        W, H, m = 420, 90, 6
        if len(self.x_hist) < 2:
            return
        ymax = max(100, max(self.x_hist) * 1.1)
        sy = lambda v: H - 6 - v / ymax * (H - 12)
        step = (W - 2 * m) / (self.x_hist.maxlen - 1)
        thr = self.ctl["min_px"]
        p.create_line(m, sy(thr), W - m, sy(thr), fill="#c33", dash=(4, 3))
        p.create_text(W - m, sy(thr) - 6, text=f"min {thr}", anchor="e", fill="#c33", font=("Arial", 7))
        pts = []
        for i, v in enumerate(self.x_hist):
            pts += [m + i * step, sy(v)]
        p.create_line(*pts, fill="#d9480f", width=2)
        p.create_text(m + 2, 8, text=f"max {int(ymax)}", anchor="w", font=("Arial", 7), fill="#666")

    def _draw_strip(self):
        s = self.strip
        key = tuple(self.q_hist)
        if self._photo_cache.get("strip_key") == key:
            return
        self._photo_cache["strip_key"] = key
        s.delete("all")
        for i, (q, y, mo) in enumerate(self.q_hist):
            row, col = divmod(i, 42)
            x0, y0 = 2 + col * 10, 4 + row * 14
            if not mo:
                s.create_rectangle(x0, y0, x0 + 8, y0 + 8, outline="#aaa")
                continue
            s.create_rectangle(x0, y0, x0 + 8, y0 + 8,
                               fill="#3cb043" if q else "#d8d8d8",
                               outline="#c33" if y == 0 else "#888")

    # ---------------------------------------------------------- run control
    def start(self, circuit=True):
        self._y_changed()
        if not self.y_ok:
            messagebox.showerror("Time code", "Fix the time code first (bits 0/1 with | & + only).")
            return
        self.ctl.update(stop=False, circuit=circuit, timecode_pending=0)
        self.q = queue.Queue()
        self.q_hist.clear()
        self.x_hist.clear()
        self.last_traj = []
        self.err = False
        self._log_clear()
        set_text(self.verdict, "Starting ...")
        self.thread = SimThread(int(self.s_cam.get()), self.ctl, self.q)
        self.thread.start()
        self.b_start.config(state="disabled")
        self.b_prev.config(state="disabled")
        self.b_stop.config(state="normal")

    def stop(self):
        self.ctl["stop"] = True

    def _close(self):
        self.ctl["stop"] = True
        self.fsearch.cancel()
        self.hidden.stop()
        self.root.after(150, self.root.destroy)

    # ---------------------------------------------------------- events
    def _poll(self):
        last = None
        try:
            for _ in range(400):                       # bounded drain per tick
                ev = self.q.get_nowait()
                kind = ev[0]
                if kind == "frame":
                    d = ev[1]
                    self.x_hist.append(d["x"])
                    if d["circuit"]:
                        self.q_hist.append((d["Q"], d["y_bin"], d["motion"]))
                        self.last_traj = d["traj"] or self.last_traj
                        if d["motion"]:
                            self._frame_log(d)
                    last = d                           # older frames are skipped, not drawn
                elif kind == "codes":
                    self._pending_codes = ev[1]        # keep only the newest
                elif kind == "hidden_hit":
                    hh = ev[1]
                    self._log(f"HIDDEN NUMBER GUESSED frame {hh['frame']:5d} x={hh['x']} "
                              f"candidate={hh['snap']['last_check'][0]} "
                              f"after {hh['snap']['checks']} checks; time code added")
                elif kind == "search_hit":
                    sh = ev[1]
                    avg = "n/a" if sh["avg"] is None else f"{sh['avg']:.1f}"
                    val, n = sh["found"]
                    self._log(f"SEARCH FOUND frame {sh['frame']:5d} x={sh['x']} "
                              f"{sh['bits']}-bit secret found in {n} guesses "
                              f"(avg {avg}); time code added")
                    self._log(f"  number (hex, {val.bit_length()} bit): {val:x}")
                elif kind == "factor_hit":
                    fh = ev[1]
                    p, q, n = fh["found"]
                    self._log(f"FACTORISATION VERIFIED frame {fh['frame']:5d} x={fh['x']} "
                              f"p found in {n} queries; time code added")
                    self._log(f"  p = {short_dec(p, 30)}")
                    self._log(f"  q = {short_dec(q, 30)}")
                elif kind == "analysis":
                    self.an_lines = ev[1]
                    self._show_analysis()
                elif kind == "error":
                    self.err = True
                    self._log("# ERROR: " + ev[1])
                    messagebox.showerror("Camera", ev[1])
                elif kind == "info":
                    self._log("# " + ev[1])
                elif kind == "done":
                    self.b_start.config(state="normal")
                    self.b_prev.config(state="normal")
                    self.b_stop.config(state="disabled")
                    set_text(self.verdict, "Camera not available" if self.err else "Stopped")
        except queue.Empty:
            pass

        now = time.perf_counter()
        if last is not None:
            self._latest = last
        # redraw at most ~30 times per second, always from the newest frame
        if getattr(self, "_latest", None) is not None and now - self._last_render >= FRAME_INTERVAL / 1000:
            d, self._latest = self._latest, None
            self._last_render = now
            self._render(d)
            self._draw_traj()
            self._draw_x()
            self._draw_strip()
        if self._pending_codes is not None and now - self._last_codes >= MINER_INTERVAL:
            self._show_miner(self._pending_codes)
            self._pending_codes = None
            self._last_codes = now
        self._flush_log()
        self.root.after(FRAME_INTERVAL // 2, self._poll)

    def _render(self, d):
        g = d["small"]
        prev_rgb = np.stack([g, g, g], axis=-1).copy()
        prev_rgb[d["mask"]] = (255, 40, 40)
        m = d["mask"].astype(np.uint8)
        mask_rgb = np.stack([m * 255, m * 115, m * 40], axis=-1)
        e = np.clip(d["energy"].astype(np.int16) * 4, 0, 255).astype(np.uint8)
        energy_rgb = np.stack([e, e, e], axis=-1)
        self.photos = [ppm_photo(mask_rgb), ppm_photo(prev_rgb), ppm_photo(energy_rgb)]
        self.c.itemconfig(self.img_b, image=self.photos[0])
        self.c.itemconfig(self.img_c, image=self.photos[1])
        self.c.itemconfig(self.img_a, image=self.photos[2])
        xt = f"x = {d['x']}" if d["motion"] else "no motion"
        if self._photo_cache.get("x_text") != xt:
            self._photo_cache["x_text"] = xt
            self.c.itemconfig(self.x_text, text=xt)

        self.cam_photo = ppm_photo(d["preview"])
        self.cam.itemconfig(self.cam_img, image=self.cam_photo)
        # fps is smoothed so the number does not flicker every frame
        fps = 1 / d["dt"]
        self._fps = fps if getattr(self, "_fps", None) is None else 10.9 * self._fps + 10.1 * fps
        ct = f"{self._fps:.0f} fps   " + (f"moving px: {d['x']}" if d["motion"] else "no motion")
        if self._photo_cache.get("cam_text") != ct:
            self._photo_cache["cam_text"] = ct
            self.cam.itemconfig(self.cam_text, text=ct)
        if not d["circuit"]:
            set_text(self.verdict, "Preview only: circuit idle. Adjust the sliders, then Start circuit.")
            return

        self._lit("drvA", d["A"], f"A={d['A']}   r={d['r']}")
        self._lit("drvB", d["c_of_x"], f"c(x) = {d['c_of_x']}")
        self._lit("link", d["B"], f"V(C)={d['vC']:.2f}  V(B)={d['vB']:.2f}", GOOD)
        self._lit("or", d["orv"])
        self._lit("and", d["andv"])
        self._lit("node", d["node"] > self.ctl.get("node_thr", 0.5), f"{d['node']:.2f}")
        self._lit("Y", d["y_bin"],
                  f"[{self.ctl['y']}]  bin {d['y_idx']}={d['y_bin']}"
                  + (" +HIDDEN" if d.get("hidden_hit") else "")
                  + (" +SEARCH" if d.get("search_hit") else "")
                  + (" +FACTOR" if d.get("factor_hit") else ""))
        ecc_note = f"  ECC {self.ctl.get('q_ecc', 'none')} / flips {d.get('q_flip', 0)}"
        self._lit("ff", d["Q"], f"Q = {d['Q']}  raw {d.get('q_raw', d['Q'])}{ecc_note}", GOOD)
        self._lit("clk", False, f"frame {d['idx']}  {round(d['dt'] * 1000, -1):.0f} ms")
        lamp = "#3cb043" if d["select"] else "#ddd"
        if self._photo_cache.get("lamp") != lamp:
            self._photo_cache["lamp"] = lamp
            self.c.itemconfig(self.lamp, fill=lamp)
        self._wire_state("A", d["A"])
        self._wire_state("B", d["B"])
        self._wire_state("gate", d["gate"])
        self._wire_state("node", d["node"] > self.ctl.get("node_thr", 0.5))
        self._wire_state("D", d["node"] > self.ctl.get("node_thr", 0.5) and d["y_bin"])

        S = d["stats"]
        if d["select"]:
            set_text(self.verdict, f"SELECT = 1  ACCEPT  (witness found within last {self.ctl['k']} frames)")
        elif d["motion"]:
            set_text(self.verdict, f"SELECT = 0  (x = {d['x']}: no witness in last {self.ctl['k']} frames)")
        else:
            set_text(self.verdict, "SELECT = 0  (no motion, no instance)")
        self._refresh_analysis(S)
        rate = S["acc_in_L"] / S["in_L"] if S["in_L"] else 0.0
        set_text(self.stats, (
            f"frames {S['frames']}  with motion {S['motion']}\n"
            f"x in L (odd composite): {S['in_L']}  accepted: {S['acc_in_L']}  ({rate:.0%})\n"
            f"x not in L accepted (false accepts): {S['false_acc']}\n"
            f"time codes added: {S['timecodes_added']} (hidden {S['hidden_timecodes']}, "
            f"search {S['search_timecodes']}, factor {S['factor_timecodes']})"))

    def _frame_log(self, d):
        kind = "odd composite" if d["in_L"] else ("even" if d["x"] % 2 == 0 else "prime/small")
        self._log(f"frame {d['idx']:5d} x={d['x']:<6d} ({kind:<13s}) r={d['r']:<6d} "
                  f"A={d['A']} B={d['B']} node={d['node']:.2f} Y={d['y_bin']} -> Q={d['Q']}")

    # ---------------------------------------------------------- log helpers
    def _log(self, s):
        self._log_buf.append(s)

    def _flush_log(self):
        if not self._log_buf:
            return
        lines, self._log_buf = self._log_buf[-200:], []
        self.log.config(state="normal")
        at_bottom = self.log.yview()[1] >= 0.999
        self.log.insert("end", "\n".join(lines) + "\n")
        n = int(self.log.index("end-1c").split(".")[0])
        if n > 400:
            self.log.delete("1.0", f"{n - 300}.0")
        if at_bottom:
            self.log.see("end")
        self.log.config(state="disabled")

    def _log_clear(self):
        self._log_buf = []
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")


def fit_window(win, content_w=1180, content_h=1180):
    """Open at the content size, but never larger than the usable screen."""
    win.update_idletasks()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    w = min(content_w, sw - 40)
    h = min(content_h, sh - 110)              # room for title bar + taskbar
    x = max(0, (sw - w) // 2)
    win.geometry(f"{w}x{h}+{x}+0")
    win.minsize(640, 420)


if __name__ == "__main__":
    mp.freeze_support()
    root = tk.Tk()
    app = App(root)
    root.update_idletasks()
    fit_window(root, app.root.winfo_reqwidth() + 24, app.root.winfo_reqheight() + 8)
    root.mainloop()
