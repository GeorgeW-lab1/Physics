#!/usr/bin/env python3
"""
QR-style engraving bitmap for the 2-LED / 2-detector photonic factoring card.

Keep next to photonic_engrave.py (reuses its netlist builder).

Black module = engraved defect, white = left clear. The card is packed as a dense
square of tiles (one tile per candidate pair a, b), with QR-style corner finders,
timing lines and an alignment mark so the engraver can register to the pattern.

Inside a tile every logic site takes two modules side by side:
    AND gate            black white
    OR gate             white black
    constant = 1        black black   (open to LED A)
    constant = 0        white white   (open to LED B)
The first sites of each tile are the 23 constants (a bits, b bits, N bits, zero),
then the gates in netlist order. Leftover modules are an alternating filler.

Wiring is NOT drawn here. Gate inputs are listed in qr_template.csv.
This looks like a QR code but no phone can read it.

Outputs (in --out): engrave_qr.png, engrave_qr_key.png, qr_template.csv, qr_tiles.csv

Usage:
    python photonic_engrave_qr.py 143
    python photonic_engrave_qr.py 221 --module-px 4 --pitch-um 25 --out card221
"""
import argparse
import csv
import math
import os

from PIL import Image, ImageDraw, ImageFont

from photonic_engrave import build_tile, const_bit

BORDER, QUIET = 8, 4


def finder(m, x0, y0):
    for dy in range(7):
        for dx in range(7):
            ring = dx in (0, 6) or dy in (0, 6)
            core = 2 <= dx <= 4 and 2 <= dy <= 4
            m[y0 + dy][x0 + dx] = 1 if (ring or core) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", type=int)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--module-px", type=int, default=3)
    ap.add_argument("--pitch-um", type=float, default=25.0, help="module pitch in micrometres")
    ap.add_argument("--out", default=".")
    x = ap.parse_args()
    N, nbits, P = x.n, x.bits, x.module_px
    wa, wb = (nbits + 1) // 2, nbits - 1
    amax, bmax = (1 << wa) - 1, (1 << wb) - 1
    if not 4 <= N < (1 << nbits):
        ap.error(f"N must satisfy 4 <= N < {1 << nbits}")
    if nbits > 12:
        ap.error("use --bits <= 12 (a 256-bit card would be ~10^115 tiles)")
    os.makedirs(x.out, exist_ok=True)

    net, _ = build_tile(wa, wb)
    consts = [n for n in net.nodes if n.kind == "C"]
    gates = [n for n in net.nodes if n.kind != "C"]
    sites = consts + gates
    S = len(sites)
    T = math.ceil(math.sqrt(2 * S))                       # tile is T x T modules

    gate_bits = []
    for n in gates:
        gate_bits += [1, 0] if n.kind == "AND" else [0, 1]

    def tile_bits(a, b):
        bits = []
        for n in consts:
            v = const_bit(n.name, a, b, N)
            bits += [v, v]
        bits += gate_bits
        bits += [(p + 1) % 2 for p in range(T * T - len(bits))]   # alternating filler
        return bits

    pairs = [(a, b) for a in range(2, amax + 1) for b in range(2, bmax + 1)]
    cols = math.ceil(math.sqrt(len(pairs)))
    rows = math.ceil(len(pairs) / cols)
    DW, DH = cols * T, rows * T
    MW, MH = DW + 2 * BORDER, DH + 2 * BORDER
    m = [[0] * MW for _ in range(MH)]
    for idx, (a, b) in enumerate(pairs):
        tr, tc = divmod(idx, cols)
        bits = tile_bits(a, b)
        for p, v in enumerate(bits):
            m[BORDER + tr * T + p // T][BORDER + tc * T + p % T] = v
    finder(m, 0, 0); finder(m, MW - 7, 0); finder(m, 0, MH - 7)
    for i in range(8, MW - 8):
        m[6][i] = 1 if i % 2 == 0 else 0
    for i in range(8, MH - 8):
        m[i][6] = 1 if i % 2 == 0 else 0
    for dy in range(5):                                    # alignment mark, bottom right
        for dx in range(5):
            m[MH - 7 + dy][MW - 7 + dx] = 1 if (dx in (0, 4) or dy in (0, 4) or (dx == 2 and dy == 2)) else 0

    # --- verify by reading the bitmap back: which tiles satisfy a*b == N? ---
    found = []
    for idx, (a, b) in enumerate(pairs):
        tr, tc = divmod(idx, cols)
        def bit(site):
            p = 2 * site
            return m[BORDER + tr * T + p // T][BORDER + tc * T + p % T]
        val = {n.name: bit(i) for i, n in enumerate(consts)}
        ra = sum(val[f"a{k}"] << k for k in range(wa))
        rb = sum(val[f"b{k}"] << k for k in range(wb))
        rn = sum(val[f"N{k}"] << k for k in range(wa + wb))
        assert (ra, rb, rn) == (a, b, N)
        if ra * rb == rn:
            found.append((a, b, tr, tc))
    truth = [(a, N // a) for a in range(2, amax + 1) if N % a == 0 and 2 <= N // a <= bmax]
    assert [(a, b) for a, b, _, _ in found] == truth

    # --- render ---
    W, H = (MW + 2 * QUIET) * P, (MH + 2 * QUIET) * P
    img = Image.new("1", (W, H), 1)
    d = ImageDraw.Draw(img)
    for y in range(MH):
        for xx in range(MW):
            if m[y][xx]:
                px, py = (xx + QUIET) * P, (y + QUIET) * P
                d.rectangle([px, py, px + P - 1, py + P - 1], fill=0)
    img.save(os.path.join(x.out, "engrave_qr.png"))

    key = img.convert("RGB")
    k = ImageDraw.Draw(key)
    for a, b, tr, tc in found:
        x0 = (QUIET + BORDER + tc * T) * P
        y0 = (QUIET + BORDER + tr * T) * P
        for o in range(3):
            k.rectangle([x0 - o, y0 - o, x0 + T * P + o, y0 + T * P + o], outline=(220, 30, 30))
    try:
        f = ImageFont.load_default(size=16)
    except TypeError:
        f = ImageFont.load_default()
    band = Image.new("RGB", (W, 54), (255, 255, 255))
    ImageDraw.Draw(band).text((10, 6), f"N={N} ({nbits}-bit)  red box = tile where a x b = N: " +
                              (", ".join(f"{a}x{b}" for a, b, *_ in found) or "none (prime)"), fill=(0, 0, 0), font=f)
    ImageDraw.Draw(band).text((10, 28), "black = engrave, white = leave clear. Corner squares + timing + small mark = registration.", fill=(80, 80, 80), font=f)
    out = Image.new("RGB", (W, H + 54), (255, 255, 255))
    out.paste(key, (0, 0)); out.paste(band, (0, H))
    out.save(os.path.join(x.out, "engrave_qr_key.png"))

    pitch = x.pitch_um / 1000.0
    with open(os.path.join(x.out, "qr_template.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["site", "node_id", "kind", "name", "module_row", "module_col_pair", "inputs(node_id:rail_swap)"])
        for i, n in enumerate(sites):
            p = 2 * i
            w.writerow([i, n.id, "CONST" if n.kind == "C" else n.kind, n.name or "", p // T, p % T,
                        " ".join(f"{s.id}:{int(v)}" for s, v in n.ins)])
    with open(os.path.join(x.out, "qr_tiles.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "tile_row", "tile_col", "origin_x_mm", "origin_y_mm", "is_match"])
        for idx, (a, b) in enumerate(pairs):
            tr, tc = divmod(idx, cols)
            w.writerow([a, b, tr, tc, round((BORDER + tc * T) * pitch, 4), round((BORDER + tr * T) * pitch, 4), int(a * b == N)])

    print(f"N={N} {nbits}-bit: {len(pairs):,} tiles in {cols}x{rows}, tile {T}x{T} modules ({len(gates)} gates + {len(consts)} constants)")
    print(f"Bitmap {MW}x{MH} modules; card {MW * pitch:.1f} x {MH * pitch:.1f} mm at {x.pitch_um:g} um/module")
    print("Read-back check passed. Match tiles:", [(a, b) for a, b, *_ in found] or "none (prime)")


if __name__ == "__main__":
    main()
