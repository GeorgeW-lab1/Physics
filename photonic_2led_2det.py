#!/usr/bin/env python3
"""
Engraving map generator for the 2-LED / 2-detector photonic factoring card.

Builds the same gate netlist the simulator uses (AND/XOR multiplier + comparator
for every candidate pair (a, b), priority chain, weighted detector merge) and
lays it out on a grid of engraving sites.

Outputs (in --out):
  engrave_map.png          whole card: every gate site, every constant, match tile outlined
  engrave_tile_detail.png  one tile zoomed, with wires, gate types, constant values
  engrave_template.csv     gate/constant sites of ONE tile (the stamp), in mm from tile origin
  engrave_tiles.csv        origin of every tile + which constants are open to LED A / LED B

Site meanings
  AND site     two masks in series on the TRUE rail, beam merge on the FALSE rail
  OR site      beam merge on the TRUE rail, two masks in series on the FALSE rail
  const site   orange = open to LED A (logic 1), blue = open to LED B (logic 0)
  inverted wire (rail swap) is drawn brown; plain wire grey
NOT gates are free: they are just a rail swap in the wiring.

This map comes from the logic simulation. It does not validate the physical
gate implementation, loss or crosstalk of a real engraved card.

Usage:
    python photonic_engrave.py 143
    python photonic_engrave.py 221 --bits 8 --pitch-um 50 --out card221
"""
import argparse
import csv
import math
import os

from PIL import Image, ImageDraw, ImageFont

C = dict(bg=(14, 16, 20), grid=(30, 34, 42), AND=(79, 209, 197), OR=(214, 96, 196),
         A=(255, 122, 61), B=(70, 130, 255), wire=(78, 84, 98), winv=(176, 112, 60),
         text=(232, 228, 220), mute=(140, 144, 154), hit=(255, 255, 255))


def font(sz):
    try:
        return ImageFont.load_default(size=sz)
    except TypeError:
        return ImageFont.load_default()


class Node:
    __slots__ = ("id", "kind", "ins", "level", "name")


class Net:
    def __init__(self):
        self.nodes, self.consts = [], {}

    def _new(self, kind, ins, name=None):
        n = Node()
        n.id, n.kind, n.ins, n.name = len(self.nodes), kind, ins, name
        n.level = 0 if kind == "C" else 1 + max(x[0].level for x in ins)
        self.nodes.append(n)
        return (n, False)

    def const(self, name):
        if name not in self.consts:
            self.consts[name] = self._new("C", [], name)
        return self.consts[name]

    def AND(self, *xs): return self._new("AND", list(xs))
    def OR(self, *xs): return self._new("OR", list(xs))
    @staticmethod
    def NOT(x): return (x[0], not x[1])
    def XOR(self, x, y): return self.OR(self.AND(x, self.NOT(y)), self.AND(self.NOT(x), y))
    def XNOR(self, x, y): return self.NOT(self.XOR(x, y))

    def full_add(self, a, b, c):
        t = self.XOR(a, b)
        return self.XOR(t, c), self.OR(self.AND(a, b), self.AND(c, t))


def build_tile(wa, wb):
    net, W = Net(), wa + wb
    a = [net.const(f"a{k}") for k in range(wa)]
    b = [net.const(f"b{k}") for k in range(wb)]
    N = [net.const(f"N{k}") for k in range(W)]
    Z = net.const("zero")
    acc = [Z] * W
    for i, ai in enumerate(a):
        row = [net.AND(ai, bj) for bj in b]
        c = Z
        for k in range(i, i + wb + 1):
            y = row[k - i] if k - i < wb else Z
            acc[k], c = net.full_add(acc[k], y, c)
    match = net.AND(*[net.XNOR(acc[k], N[k]) for k in range(W)])
    return net, match


def layout(net):
    pos, per = {}, {}
    consts = [n for n in net.nodes if n.kind == "C"]
    for j, n in enumerate(consts):
        pos[n.id] = (0, j)
    for n in net.nodes:
        if n.kind != "C":
            per[n.level] = per.get(n.level, 0) + 1
            pos[n.id] = (n.level, per[n.level] - 1)
    TW = max(per) + 1
    TH = max(len(consts), max(per.values()))
    return pos, TW, TH, consts


def const_bit(name, a, b, N):
    if name == "zero":
        return 0
    k = int(name[1:])
    return ((a if name[0] == "a" else b if name[0] == "b" else N) >> k) & 1


def draw_foot(d, x0, y0, wa, wb, amax, bmax, F, width):
    d.text((x0, y0), "Output merge (what the two photodetectors collect)", fill=C["text"], font=F[1])
    d.text((x0, y0 + 26), f"Priority chain: {amax - 1} stages, each 1 AND + 1 OR, smallest factor a wins.", fill=C["mute"], font=F[0])
    for col, (label, w, mx) in enumerate([("Detector 1 (a)", wa, amax), ("Detector 2 (b)", wb, bmax)]):
        bx = x0 + col * min(560, width // 2)
        d.text((bx, y0 + 56), f"{label}: bit attenuator transmission = 2^k / {mx}", fill=C["text"], font=F[0])
        for k in range(w):
            t = (1 << k) / mx
            yy = y0 + 80 + k * 18
            d.text((bx, yy), f"bit {k}", fill=C["mute"], font=F[0])
            d.rectangle([bx + 50, yy + 2, bx + 50 + max(1, int(300 * t)), yy + 12], fill=C["A"])
            d.text((bx + 360, yy), f"{t * 100:.3f}%", fill=C["text"], font=F[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", type=int)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--pitch-um", type=float, default=50.0, help="site pitch in micrometres")
    ap.add_argument("--out", default=".")
    x = ap.parse_args()
    N, nbits = x.n, x.bits
    wa, wb = (nbits + 1) // 2, nbits - 1
    amax, bmax = (1 << wa) - 1, (1 << wb) - 1
    na, nb = amax - 1, bmax - 1
    if not 4 <= N < (1 << nbits):
        ap.error(f"N must satisfy 4 <= N < {1 << nbits}")
    if nb * 4 > 14000:
        ap.error(f"{nbits} bits means {na:,} x {nb:,} tiles, too many to draw "
                 f"(a 256-bit card would be ~10^{int(383 * 0.30103)} tiles). Use --bits <= 12.")
    os.makedirs(x.out, exist_ok=True)

    net, match = build_tile(wa, wb)
    pos, TW, TH, consts = layout(net)
    gates = [n for n in net.nodes if n.kind != "C"]
    pitch = x.pitch_um / 1000.0
    matches = [(a, N // a) for a in range(2, amax + 1) if N % a == 0 and 2 <= N // a <= bmax]

    # ---- full card -------------------------------------------------------
    s = max(0, min(3, 14000 // (nb * TW)))
    detail = s >= 1
    s = s if detail else max(2, min(8, 14000 // nb))
    tw, th = (TW * s, TH * s) if detail else (s, s)
    cw, ch = nb * tw, na * th
    LM, HEAD, FOOT = 150, 130, 40 + 80 + 18 * wb + 20
    W = max(cw + LM + 20, 1300)
    F = (font(14), font(20), font(26))
    img = Image.new("RGB", (W, HEAD + ch + FOOT), C["bg"])
    d = ImageDraw.Draw(img)

    # gate-layer stamp
    if detail:
        stamp = Image.new("RGB", (tw, th), C["bg"])
        sd = ImageDraw.Draw(stamp)
        g = 1 if s >= 3 else 0
        for n in gates:
            px, py = pos[n.id]
            sd.rectangle([px * s, py * s, px * s + s - 1 - g, py * s + s - 1 - g], fill=C[n.kind])
    for a in range(2, amax + 1):
        for b in range(2, bmax + 1):
            ox, oy = LM + (b - 2) * tw, HEAD + (a - 2) * th
            if detail:
                img.paste(stamp, (ox, oy))
                for j, n in enumerate(consts):
                    col = C["A"] if const_bit(n.name, a, b, N) else C["B"]
                    d.rectangle([ox, oy + j * s, ox + s - 1, oy + j * s + s - 1], fill=col)
                d.rectangle([ox, oy, ox + tw - 1, oy + th - 1], outline=C["grid"])
            else:
                d.rectangle([ox, oy, ox + tw - 2, oy + th - 2], fill=C["grid"])
    for a, b in matches:
        ox, oy = LM + (b - 2) * tw, HEAD + (a - 2) * th
        for k in range(3):
            d.rectangle([ox - k, oy - k, ox + tw + k - 1, oy + th + k - 1], outline=C["hit"])
    # header
    d.text((16, 10), f"Photonic factoring card engraving map: N = {N} ({nbits}-bit)", fill=C["text"], font=F[2])
    d.text((16, 46), f"{na}x{nb} = {na * nb:,} tiles   {len(gates)} gate sites + {len(consts)} constants per tile   "
           f"total gate sites {na * nb * len(gates):,}   card {cw / tw * TW * pitch:.1f} x {na * TH * pitch:.1f} mm at {x.pitch_um:g} um pitch",
           fill=C["mute"], font=F[0])
    lg = [(C["AND"], "AND site"), (C["OR"], "OR site"), (C["A"], "constant open to LED A (1)"), (C["B"], "constant open to LED B (0)")]
    lx = 16
    for col, t in lg:
        d.rectangle([lx, 76, lx + 14, 90], fill=col)
        d.text((lx + 20, 74), t, fill=C["text"], font=F[0])
        lx += 40 + 9 * len(t)
    d.rectangle([lx, 76, lx + 14, 90], outline=C["hit"], width=2)
    d.text((lx + 20, 74), "tile where a x b = N (light reaches detectors)" if matches else "no match: N is prime, detectors stay dark", fill=C["text"], font=F[0])
    d.text((16, 100), "Rows: a = 2.." + str(amax) + " (top to bottom)    Columns: b = 2.." + str(bmax) + " (left to right)"
           + ("" if detail else "    OVERVIEW: tile interior is the same stamp, see engrave_tile_detail.png"), fill=C["mute"], font=F[0])
    # LEDs
    for k, (col, t) in enumerate([(C["A"], "LED A"), (C["B"], "LED B")]):
        cy = HEAD + 40 + k * 70
        d.ellipse([20, cy, 50, cy + 30], fill=col)
        d.text((58, cy + 6), t, fill=C["text"], font=F[0])
        d.line([(50, cy + 15), (LM - 4, cy + 15)], fill=col, width=2)
    # foot
    fy = HEAD + ch + 14
    draw_foot(d, 16, fy, wa, wb, amax, bmax, F, W)
    for k, t in enumerate(["Detector 1", "Detector 2"]):
        cx = W - 190 + k * 95
        d.ellipse([cx, fy + 6, cx + 26, fy + 32], outline=C["text"], width=2)
        d.text((cx - 6, fy + 38), t, fill=C["text"], font=F[0])
    img.save(os.path.join(x.out, "engrave_map.png"))

    # ---- tile detail -----------------------------------------------------
    S, M = 16, 100
    ta, tb = matches[0] if matches else (2, 2)
    dw, dh = M + TW * S + 20, 60 + TH * S + 20
    im2 = Image.new("RGB", (dw, dh), C["bg"])
    e = ImageDraw.Draw(im2)
    ctr = lambda i: (M + pos[i][0] * S + S // 2, 60 + pos[i][1] * S + S // 2)
    for n in gates:
        for src, inv in n.ins:
            e.line([ctr(src.id), ctr(n.id)], fill=C["winv"] if inv else C["wire"], width=1)
    for j, n in enumerate(consts):
        bit = const_bit(n.name, ta, tb, N)
        px, py = pos[n.id]
        e.rectangle([M + px * S + 2, 60 + py * S + 2, M + px * S + S - 3, 60 + py * S + S - 3], fill=C["A"] if bit else C["B"])
        e.text((6, 60 + py * S + 1), f"{n.name}={bit}", fill=C["mute"], font=F[0])
    for n in gates:
        px, py = pos[n.id]
        e.rounded_rectangle([M + px * S + 2, 60 + py * S + 2, M + px * S + S - 3, 60 + py * S + S - 3], radius=3, fill=C[n.kind])
        e.text((M + px * S + 4, 60 + py * S + 1), n.kind[0], fill=C["bg"], font=F[0])
    mx_, my_ = pos[match[0].id]
    e.rectangle([M + mx_ * S, 60 + my_ * S, M + mx_ * S + S - 1, 60 + my_ * S + S - 1], outline=C["hit"], width=2)
    e.text((6, 8), f"Tile detail: a = {ta}, b = {tb}" + ("  (match tile)" if matches else "  (N prime: no match tile, shown for a=b=2)"), fill=C["text"], font=F[1])
    e.text((6, 34), "Left column: constants.  Right edge: white box = match output.  Brown wires swap rails (NOT).", fill=C["mute"], font=F[0])
    im2.save(os.path.join(x.out, "engrave_tile_detail.png"))

    # ---- CSVs ------------------------------------------------------------
    with open(os.path.join(x.out, "engrave_template.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "name", "site_x", "site_y", "x_mm", "y_mm", "inputs(id:inv)"])
        for n in net.nodes:
            px, py = pos[n.id]
            w.writerow([n.id, n.kind if n.kind != "C" else "CONST", n.name or "", px, py,
                        round(px * pitch, 4), round(py * pitch, 4),
                        " ".join(f"{s.id}:{int(i)}" for s, i in n.ins)])
    with open(os.path.join(x.out, "engrave_tiles.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "origin_x_mm", "origin_y_mm", "open_to_LED_A", "open_to_LED_B", "is_match"])
        for a in range(2, amax + 1):
            for b in range(2, bmax + 1):
                ones = [n.name for n in consts if const_bit(n.name, a, b, N)]
                zeros = [n.name for n in consts if not const_bit(n.name, a, b, N)]
                w.writerow([a, b, round((b - 2) * TW * pitch, 4), round((a - 2) * TH * pitch, 4),
                            " ".join(ones), " ".join(zeros), int(a * b == N)])

    print(f"N={N}  {nbits}-bit  tiles {na}x{nb}={na * nb:,}  per tile: {len(gates)} gates + {len(consts)} constants "
          f"(stamp {TW}x{TH} sites)")
    print(f"Card size ~{nb * TW * pitch:.1f} x {na * TH * pitch:.1f} mm at {x.pitch_um:g} um pitch; "
          f"full gate-level map: {detail}")
    print("Match tiles (a, b):", matches or "none (prime)")
    print("Wrote engrave_map.png, engrave_tile_detail.png, engrave_template.csv, engrave_tiles.csv in", os.path.abspath(x.out))


if __name__ == "__main__":
    main()
