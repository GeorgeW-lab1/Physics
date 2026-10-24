#!/usr/bin/env python3
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

from photonic_and_gate_sim import Optics, ONE, ZERO, bits_of, multiply


def run(N, nbits=256, loss=1.0, leak=0.0, noise=0.0, adc_bits=10, seed=None):
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
    ap.add_argument("--bits", type=int, default=256)
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
