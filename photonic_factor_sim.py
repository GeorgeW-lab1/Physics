#!/usr/bin/env python3
"""
Static photonic factoring bench (pure Python, no dependencies).

No sweep, no clock. One exposure:

  * Every IR LED row is switched on at the same time.
  * The patterned detection card is a printed multiplication table.
    Cell (i, j) is open only where i * j == N (i, j >= 2).
  * Light reaches the photodetector for column j only through an open cell,
    so after a single exposure the lit detectors ARE the divisors of N.

The full prime factorization is read from that one frame: take the smallest
lit detector (always prime), divide it out, repeat using the same frame.

Cost of doing it in one shot: ~N/2 emitters, ~N/2 detectors, ~N^2/4 card cells.

Usage:
    python photonic_factor_sim.py 8633
    python photonic_factor_sim.py 8633 --leak 0.00005 --noise 0.3
    python photonic_factor_sim.py 60 --show
"""
import argparse
import math
import random


def expose(n, leak=0.0, noise=0.0, rng=None):
    """One static exposure. Returns detector readings for columns j = 2..n//2."""
    rng = rng or random.Random()
    cols = range(2, n // 2 + 1)
    rows = n // 2 - 1                      # every row emitter is on at once
    frame = {}
    for j in cols:
        light = leak * rows                # closed cells all leak a little
        light += rng.gauss(0, noise) * 0.5  # detector noise
        if n % j == 0 and 2 <= n // j <= n // 2:
            light += 1.0                   # the one open cell in this column
        frame[j] = light
    return frame


def read_frame(n, frame, threshold=0.5, verify=True):
    """Fired detectors -> divisors. Optional Arduino check: does j divide N?"""
    fired = [j for j, v in frame.items() if v > threshold]
    if not verify:
        return fired, 0
    good = [j for j in fired if n % j == 0]
    return good, len(fired) - len(good)


def factor_from_divisors(n, divisors):
    """Prime factors from the single frame (smallest lit divisor is prime)."""
    ds, out, m = sorted(divisors), [], n
    while m > 1:
        d = next((d for d in ds if m % d == 0 and d <= m), None)
        if d is None or d == m:
            out.append(m)
            break
        out.append(d)
        m //= d
    return out


def show_card(n):
    """ASCII card with all rows lit. '#' = open cell, '.' = closed."""
    top = n // 2
    print(f"Card for N={n}  (rows i and columns j both run 2..{top})")
    print("      " + "".join(str(j % 10) for j in range(2, top + 1)))
    for i in range(2, top + 1):
        line = "".join("#" if i * j == n else "." for j in range(2, top + 1))
        print(f"{i:>4}  {line}")
    print("\nLit columns after one exposure = divisors:",
          [j for j in range(2, top + 1) if n % j == 0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", type=int)
    ap.add_argument("--leak", type=float, default=0.0, help="light through each closed cell")
    ap.add_argument("--noise", type=float, default=0.0, help="detector noise std-dev")
    ap.add_argument("--no-verify", action="store_true", help="skip the Arduino divisibility check")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--show", action="store_true", help="ASCII card (N <= 120)")
    a = ap.parse_args()
    n = a.n
    if n < 4:
        ap.error("n must be >= 4")
    if a.show:
        if n > 120:
            ap.error("--show only for N <= 120")
        show_card(n)
        print()

    rng = random.Random(a.seed)
    frame = expose(n, a.leak, a.noise, rng)
    divs, false_alarms = read_frame(n, frame, verify=not a.no_verify)
    raw = [j for j, v in frame.items() if v > 0.5]
    true_divs = [j for j in range(2, n // 2 + 1) if n % j == 0]
    false_alarms = len([j for j in raw if n % j])   # fired on closed columns
    missed = [j for j in true_divs if j not in divs]
    primes = factor_from_divisors(n, divs)

    print(f"N = {n}   exposures: 1   time bins: 0")
    print(f"Emitters: {n//2-1:,}  Detectors: {n//2-1:,}  Card cells: {(n//2-1)**2:,}")
    print(f"Detectors fired: {len(raw)}   false alarms: {false_alarms}   missed divisors: {len(missed)}")
    shown = divs if len(divs) <= 20 else divs[:20] + ["..."]
    print(f"Lit divisors: {shown}")
    ok = math.prod(primes) == n
    tag = " (reported prime)" if len(primes) == 1 else ""
    print(f"\n{n} = {' x '.join(map(str, primes))}{tag}" + ("" if ok else "   READOUT ERROR"))
    if false_alarms:
        print("Closed cells leaked enough light to fire detectors; the check filtered them.")
    print(f"\nScaling: a {n.bit_length()}-bit input needs ~N/2 emitters and detectors. "
          f"A 2048-bit RSA modulus would need ~10^{int(2048*0.301)} of each.")


if __name__ == "__main__":
    main()
