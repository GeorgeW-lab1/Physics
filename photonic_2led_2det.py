#!/usr/bin/env python3
"""
Photonic AND-gate factoring bench: 2 emitters, static, no sweep, no clock.

Hardware model (matches the tweet: IR LEDs + patterned card + photodetectors)
  * Emitter A lights the TRUE rail, Emitter B lights the FALSE rail (dual-rail).
    A signal x is a pair of light intensities (p, q): 1 = (bright, dark),
    0 = (dark, bright). NOT is free: swap the rails.
  * Constants (N's bits, the bits of every candidate a and b) are printed on
    the card: a cell open to A is a 1, open to B is a 0.
  * AND gate = two beams through two masks in series (intensities multiply)
    on the TRUE rail, and a beam merge (intensities add) on the FALSE rail.
    OR is the same thing with the rails swapped. Every gate has insertion loss.
  * A grid of cells, one per candidate pair (a, b), each holds an AND/XOR
    multiplier and a comparator against N. All cells run in the same instant.
  * Priority chain picks the smallest factor; photodetectors read its bits.
  * Readout is differential: bit = 1 if the TRUE-rail detector beats the FALSE-rail one.

Usage:
    python photonic_and_gate_sim.py 143
    python photonic_and_gate_sim.py 221 --loss 0.99 --leak 0.002 --noise 0.05
    python photonic_and_gate_sim.py --selftest
"""
import argparse
import math
import random


class Optics:
    def __init__(self, loss=1.0, leak=0.0, noise=0.0, seed=None):
        self.loss, self.leak, self.noise = loss, leak, noise
        self.rng = random.Random(seed)
        self.gates = 0

    def _fix(self, p, q):
        self.gates += 1
        p = min(1.0, p * self.loss + self.leak)
        q = min(1.0, q * self.loss + self.leak)
        return (p, q)

    def AND(self, *xs):
        p = 1.0
        for x in xs:
            p *= x[0]                      # series masks: intensities multiply
        q = sum(x[1] for x in xs)          # merged beams: intensities add
        return self._fix(p, min(1.0, q))

    def OR(self, *xs):
        q = 1.0
        for x in xs:
            q *= x[1]
        p = sum(x[0] for x in xs)
        return self._fix(min(1.0, p), q)

    @staticmethod
    def NOT(x):
        return (x[1], x[0])

    def XOR(self, x, y):
        return self.OR(self.AND(x, self.NOT(y)), self.AND(self.NOT(x), y))

    def XNOR(self, x, y):
        return self.NOT(self.XOR(x, y))

    def full_add(self, a, b, c):
        t = self.XOR(a, b)
        return self.XOR(t, c), self.OR(self.AND(a, b), self.AND(c, t))


ONE, ZERO = (1.0, 0.0), (0.0, 1.0)         # printed on the card: open to A / open to B


def bits_of(v, w):
    return [ONE if (v >> k) & 1 else ZERO for k in range(w)]


def multiply(o, a, b):
    """Array multiplier from AND partial products + full adders."""
    W = len(a) + len(b)
    acc = [ZERO] * W
    for i, ai in enumerate(a):
        row = [o.AND(ai, bj) for bj in b]
        c = ZERO
        for k in range(i, i + len(b) + 1):
            y = row[k - i] if k - i < len(row) else ZERO
            acc[k], c = o.full_add(acc[k], y, c)
    return acc


def build_and_run(N, nbits=8, loss=1.0, leak=0.0, noise=0.0, seed=None):
    o = Optics(loss, leak, noise, seed)
    wa, wb = (nbits + 1) // 2, nbits - 1
    amax, bmax = (1 << wa) - 1, (1 << wb) - 1
    W = wa + wb
    Nbits = bits_of(N, W)

    # one comparator cell per candidate pair (a, b)
    cell = {}
    matchA = {}
    for a in range(2, amax + 1):
        abits = bits_of(a, wa)
        row = []
        for b in range(2, bmax + 1):
            prod = multiply(o, abits, bits_of(b, wb))
            eq = [o.XNOR(prod[k], Nbits[k]) for k in range(W)]
            m = o.AND(*eq)
            cell[(a, b)] = m
            row.append(m)
        matchA[a] = o.OR(*row)             # beams of one a-row merge

    # priority chain: smallest factor wins
    seen, win = ZERO, {}
    for a in range(2, amax + 1):
        win[a] = o.AND(matchA[a], o.NOT(seen))
        seen = o.OR(seen, matchA[a])
    is_prime = o.NOT(seen)

    # output detectors
    det = []
    for k in range(wa):
        det.append(("a%d" % k, o.OR(*[win[a] for a in range(2, amax + 1) if (a >> k) & 1] or [ZERO])))
    for k in range(wb):
        terms = [o.AND(win[a], cell[(a, b)]) for a in range(2, amax + 1)
                 for b in range(2, bmax + 1) if (b >> k) & 1]
        det.append(("b%d" % k, o.OR(*terms) if terms else ZERO))
    det.append(("prime", is_prime))

    # photodetectors: add noise, differential readout
    out, margin = {}, 1.0
    for name, (p, q) in det:
        p += o.rng.gauss(0, noise)
        q += o.rng.gauss(0, noise)
        out[name] = (round(p, 3), round(q, 3))
        margin = min(margin, abs(p - q))
    bit = {n: int(p > q) for n, (p, q) in out.items()}
    a = sum(bit["a%d" % k] << k for k in range(wa))
    b = sum(bit["b%d" % k] << k for k in range(wb))
    return dict(a=a, b=b, prime=bool(bit["prime"]), out=out, margin=margin,
                gates=o.gates, cells=len(cell), detectors=len(det), emitters=2,
                wa=wa, wb=wb)


def factor(N, **kw):
    """Repeat the static card on the cofactor to get the full prime list."""
    fs, m = [], N
    while m > 3:
        r = build_and_run(m, **kw)
        if r["prime"] or r["a"] < 2 or r["a"] * r["b"] != m:
            break
        fs.append(r["a"])
        m = r["b"]
    fs.append(m)
    return fs


def selftest():
    bad = 0
    for N in range(4, 256):
        r = build_and_run(N)
        truth = next((d for d in range(2, math.isqrt(N) + 1) if N % d == 0), None)
        ok = (r["prime"] and truth is None) or (not r["prime"] and r["a"] == truth and r["a"] * r["b"] == N)
        bad += not ok
    print(f"Ideal optics, all N=4..255: {255 - 3 - bad}/{255 - 3} correct")




"""
Two LEDs, two photodetectors, AND-gate logic. Static, one exposure, no sweep.

Keep this file next to photonic_and_gate_sim.py (it reuses the optical gate model).

Same card logic as before: dual-rail AND/OR gates, a multiplier + comparator for
every candidate pair (a, b), and a priority chain that picks the smallest factor.
What changes is the readout. Two detectors can only give two numbers, so they
read ANALOG values:

  * Detector 1 collects the beams of the 'a' output bits, each passed through an
    attenuator of weight 2^k / (2^wa - 1), so total intensity = a / a_max.
  * Detector 2 does the same for 'b': intensity = b / b_max.
  * The Arduino UNO's 10-bit ADC digitizes each detector and divides back out.
  * Prime: no cell matched, both detectors stay dark.

Price of 2 detectors: no differential readout, so loss, leak and noise turn into
analog errors in the decoded factor (b's top bit is worth only 1/255 of full scale).

Usage:
    python photonic_2led_2det.py 143
    python photonic_2led_2det.py 221 --loss 0.9995 --noise 0.001
    python photonic_2led_2det.py --selftest
"""
import argparse
import math
import random


def run(N, nbits=8, loss=1.0, leak=0.0, noise=0.0, adc_bits=10, seed=None):
    o = Optics(loss, leak, noise, seed)
    wa, wb = (nbits + 1) // 2, nbits - 1
    amax, bmax = (1 << wa) - 1, (1 << wb) - 1
    W = wa + wb
    Nbits = bits_of(N, W)

    cell, matchA = {}, {}
    for a in range(2, amax + 1):
        ab = bits_of(a, wa)
        row = []
        for b in range(2, bmax + 1):
            prod = multiply(o, ab, bits_of(b, wb))
            m = o.AND(*[o.XNOR(prod[k], Nbits[k]) for k in range(W)])
            cell[(a, b)] = m
            row.append(m)
        matchA[a] = o.OR(*row)

    seen, win = ZERO, {}
    for a in range(2, amax + 1):
        win[a] = o.AND(matchA[a], o.NOT(seen))
        seen = o.OR(seen, matchA[a])

    # output bit beams (TRUE rail only goes to the detectors)
    a_beams = [o.OR(*[win[a] for a in range(2, amax + 1) if (a >> k) & 1])[0] for k in range(wa)]
    b_beams = []
    for k in range(wb):
        terms = [o.AND(win[a], cell[(a, b)]) for a in range(2, amax + 1)
                 for b in range(2, bmax + 1) if (b >> k) & 1]
        b_beams.append(o.OR(*terms)[0])

    # two photodetectors: weighted sum of beams, plus noise, then the ADC
    full = (1 << adc_bits) - 1
    def detect(beams, wmax):
        r = sum(beams[k] * (1 << k) / wmax for k in range(len(beams))) + o.rng.gauss(0, noise)
        code = max(0, min(full, round(r * full)))
        return r, code, round(code / full * wmax)
    ra, ca, a = detect(a_beams, amax)
    rb, cb, b = detect(b_beams, bmax)
    prime = a < 2 or b < 2
    return dict(a=a, b=b, prime=prime, ra=ra, rb=rb, ca=ca, cb=cb, gates=o.gates, cells=len(cell))


def factor(N, **kw):
    fs, m = [], N
    while m > 3:
        r = run(m, **kw)
        if r["prime"] or r["a"] * r["b"] != m:
            break
        fs.append(r["a"])
        m = r["b"]
    fs.append(m)
    return fs


def selftest():
    bad = 0
    for N in range(4, 256):
        r = run(N)
        t = next((d for d in range(2, math.isqrt(N) + 1) if N % d == 0), None)
        ok = r["prime"] if t is None else (not r["prime"] and r["a"] == t and r["a"] * r["b"] == N)
        bad += not ok
    print(f"Ideal optics, 2 detectors, all N=4..255: {252 - bad}/252 correct")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", type=int, nargs="?")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--loss", type=float, default=1.0)
    ap.add_argument("--leak", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--adc", type=int, default=10, help="ADC bits (UNO = 10)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    x = ap.parse_args()
    if x.selftest:
        return selftest()
    if x.n is None or not 4 <= x.n < (1 << x.bits):
        ap.error(f"give N with 4 <= N < {1 << x.bits}")
    kw = dict(nbits=x.bits, loss=x.loss, leak=x.leak, noise=x.noise, adc_bits=x.adc, seed=x.seed)
    r = run(x.n, **kw)
    print(f"N = {x.n}   2 LEDs, 2 photodetectors, 1 exposure, 0 sweeps")
    print(f"Candidate cells: {r['cells']}   Optical gates: {r['gates']:,}")
    print(f"Detector 1 (a): {r['ra']:.4f} of full scale -> ADC {r['ca']} -> a = {r['a']}")
    print(f"Detector 2 (b): {r['rb']:.4f} of full scale -> ADC {r['cb']} -> b = {r['b']}")
    if r["prime"]:
        print(f"\nBoth detectors dark: {x.n} reported prime")
    else:
        fs = factor(x.n, **kw)
        ok = r["a"] * r["b"] == x.n and math.prod(fs) == x.n
        print(f"\n{x.n} = {r['a']} x {r['b']}" + ("" if ok else "   READOUT ERROR"))
        print(f"Prime factors: {' x '.join(map(str, fs))}" + ("" if ok else "   READOUT ERROR"))


if __name__ == "__main__":
    main()
