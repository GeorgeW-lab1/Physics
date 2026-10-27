"""
kaspa_gatec.py - compile a kHeavyHash bench job into a logic-gate circuit for the optical bench.

What it builds
  stage "heavy": inputs = 256 bits of pow_hash, outputs = 256 bits of the XOR-transformed hash.
                 The 64x64 nibble matrix is a CONSTANT of the job, so every "AND with a matrix
                 bit" disappears (AND with 0 -> const 0, AND with 1 -> wire). What is left is a
                 weighted sum of input bits: a column-compression adder tree of full/half adders.
  stage "job":   input = 64 nonce bits, outputs = 256 hash bits + 1 "hit" bit (hash <= target).
                 cSHAKE256 -> heavy stage -> cSHAKE256 -> reverse -> comparator. pre-PoW hash,
                 timestamp, target, matrix and the cSHAKE prefix states are all constant-folded.

Compiler pipeline
  front end (builders) -> hash-consing + constant folding while building -> dead-gate removal
  -> NOT fusion (NOT(AND)->NAND, ...) -> levelise -> pack into waves of 8 lanes (one lane per
  ITO pixel / photodiode pair) -> verify by bit-parallel simulation against the reference
  hash -> emit netlist stats / JSON / C header for the Pico gate VM.

Optical gate model (what one lane does in one wave)
  Slot A: pixel k driven at input bit a. Slot B: pixel k driven at input bit b (dark for 1-input
  gates). Photodiode k integrates, giving s = a + b in {0,1,2}. Output = truth table over s:
      AND {0,0,1}  OR {0,1,1}  XOR {0,1,0}  NAND {1,1,0}  NOR {1,0,0}  XNOR {1,0,1}
  NOT(a) is NOR(a, 0).

Usage
  python kaspa_gatec.py --stage heavy --verify --emit-c gate_program.h
  python kaspa_gatec.py --stage job --verify --leading-zero-bits 4
  python kaspa_gatec.py --stage heavy --optical-sim --noise 0.08

Needs: pip install numpy pycryptodome
"""
from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import time

import numpy as np
from Crypto.Hash import cSHAKE256

AND, OR, XOR, NAND, NOR, XNOR = range(6)
OPNAME = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR"]
TT = {AND: (0, 0, 1), OR: (0, 1, 1), XOR: (0, 1, 0), NAND: (1, 1, 0), NOR: (1, 0, 0), XNOR: (1, 0, 1)}
INVERT = {AND: NAND, OR: NOR, XOR: XNOR}
LANES = 8
RATE = 136
MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# Netlist builder with folding + hash-consing
# ---------------------------------------------------------------------------

class Circuit:
    """Wires: 0 = const 0, 1 = const 1, then inputs, then gates (in creation order)."""

    def __init__(self, n_inputs: int):
        self.n_in = n_inputs
        self.ops: list[int] = []
        self.ga: list[int] = []
        self.gb: list[int] = []
        self.cache: dict[tuple, int] = {}
        self.not_of: dict[int, int] = {}
        self.base = 2 + n_inputs

    def inputs(self) -> list[int]:
        return list(range(2, 2 + self.n_in))

    def _new(self, op: int, a: int, b: int) -> int:
        self.ops.append(op)
        self.ga.append(a)
        self.gb.append(b)
        return self.base + len(self.ops) - 1

    def not_(self, a: int) -> int:
        if a == 0:
            return 1
        if a == 1:
            return 0
        if a in self.not_of:
            return self.not_of[a]
        w = self._new(NOR, a, 0)
        self.not_of[a] = w
        self.not_of[w] = a
        return w

    def _bin(self, op: int, a: int, b: int) -> int:
        if a > b:
            a, b = b, a
        key = (op, a, b)
        w = self.cache.get(key)
        if w is None:
            w = self._new(op, a, b)
            self.cache[key] = w
        return w

    def and_(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        if a == 1:
            return b
        if b == 1:
            return a
        if a == b:
            return a
        if self.not_of.get(a) == b:
            return 0
        return self._bin(AND, a, b)

    def or_(self, a: int, b: int) -> int:
        if a == 1 or b == 1:
            return 1
        if a == 0:
            return b
        if b == 0:
            return a
        if a == b:
            return a
        if self.not_of.get(a) == b:
            return 1
        return self._bin(OR, a, b)

    def xor(self, a: int, b: int) -> int:
        if a == 0:
            return b
        if b == 0:
            return a
        if a == 1:
            return self.not_(b)
        if b == 1:
            return self.not_(a)
        if a == b:
            return 0
        if self.not_of.get(a) == b:
            return 1
        return self._bin(XOR, a, b)

    def xor_n(self, ws: list[int]) -> int:
        ws = list(ws)
        while len(ws) > 1:  # balanced tree for low depth
            nxt = [self.xor(ws[i], ws[i + 1]) for i in range(0, len(ws) - 1, 2)]
            if len(ws) & 1:
                nxt.append(ws[-1])
            ws = nxt
        return ws[0]

    # full / half adders
    def fa(self, a: int, b: int, c: int, need_sum: bool = True):
        p = self.xor(a, b)
        carry = self.or_(self.and_(a, b), self.and_(c, p))
        s = self.xor(p, c) if need_sum else None
        return s, carry

    def ha(self, a: int, b: int, need_sum: bool = True):
        return (self.xor(a, b) if need_sum else None), self.and_(a, b)


# ---------------------------------------------------------------------------
# Finalise: DCE, NOT fusion, levelise, wave packing
# ---------------------------------------------------------------------------

class Program:
    def __init__(self):
        self.n_in = 0
        self.ops: list[int] = []
        self.a: list[int] = []
        self.b: list[int] = []
        self.wave_end: list[int] = []     # cumulative gate count at the end of each wave
        self.outputs: dict[str, list[int]] = {}
        self.depth = 0
        self.stats: dict = {}

    @property
    def n_gates(self) -> int:
        return len(self.ops)

    @property
    def n_wires(self) -> int:
        return 2 + self.n_in + self.n_gates


def finalise(c: Circuit, outputs: dict[str, list[int]]) -> Program:
    n_in, base = c.n_in, c.base
    ops, ga, gb = c.ops[:], c.ga[:], c.gb[:]
    n = len(ops)

    def live_set(outs):
        live = [False] * n
        stack = [w for ws in outs.values() for w in ws if w >= base]
        while stack:
            w = stack.pop()
            i = w - base
            if live[i]:
                continue
            live[i] = True
            for x in (ga[i], gb[i]):
                if x >= base and not live[x - base]:
                    stack.append(x)
        return live

    raw_before = n
    live = live_set(outputs)

    # fanout among live gates and outputs
    fan = [0] * n
    for i in range(n):
        if live[i]:
            for x in (ga[i], gb[i]):
                if x >= base:
                    fan[x - base] += 1
    out_wires = {w for ws in outputs.values() for w in ws}
    for w in out_wires:
        if w >= base:
            fan[w - base] += 1

    # NOT fusion: NOT(g) where g = AND/OR/XOR with fanout 1 -> inverted gate
    fused = 0
    for i in range(n):
        if live[i] and ops[i] == NOR and gb[i] == 0 and ga[i] >= base:
            j = ga[i] - base
            if ops[j] in INVERT and fan[j] == 1 and (ga[i] not in out_wires):
                ops[i], ga[i], gb[i] = INVERT[ops[j]], ga[j], gb[j]
                live[j] = False
                fused += 1
    live2 = live_set(outputs) if fused == 0 else None
    if fused:
        # recompute liveness after rewrites (outputs may now bypass removed gates)
        live = [False] * n
        stack = [w for ws in outputs.values() for w in ws if w >= base]
        while stack:
            w = stack.pop()
            i = w - base
            if live[i]:
                continue
            live[i] = True
            for x in (ga[i], gb[i]):
                if x >= base and not live[x - base]:
                    stack.append(x)

    # levelise
    level = [0] * n
    for i in range(n):
        if not live[i]:
            continue
        la = level[ga[i] - base] if ga[i] >= base else 0
        lb = level[gb[i] - base] if gb[i] >= base else 0
        level[i] = 1 + max(la, lb)
    order = sorted((i for i in range(n) if live[i]), key=lambda i: (level[i], i))

    # renumber in schedule order; waves of <= LANES gates within a level
    newid = {}
    prog = Program()
    prog.n_in = n_in
    for k, i in enumerate(order):
        newid[base + i] = 2 + n_in + k

    def remap(w):
        return newid[w] if w >= base else w

    prev_level, in_wave = None, 0
    for k, i in enumerate(order):
        if prev_level is not None and (level[i] != prev_level or in_wave == LANES):
            prog.wave_end.append(k)
            in_wave = 0
        prog.ops.append(ops[i])
        prog.a.append(remap(ga[i]))
        prog.b.append(remap(gb[i]))
        prev_level = level[i]
        in_wave += 1
    if order:
        prog.wave_end.append(len(order))
    prog.outputs = {k: [remap(w) for w in ws] for k, ws in outputs.items()}
    prog.depth = max(level) if n else 0
    hist = {OPNAME[o]: prog.ops.count(o) for o in range(6)}
    prog.stats = {
        "gates_built": raw_before,
        "gates_after_dce": int(sum(live)) + fused,
        "not_fused": fused,
        "gates": prog.n_gates,
        "depth": prog.depth,
        "waves": len(prog.wave_end),
        "lanes": LANES,
        "lane_utilisation": prog.n_gates / max(1, len(prog.wave_end) * LANES),
        "by_type": hist,
        "flash_bytes_c_header": prog.n_gates * 8 + len(prog.wave_end) * 4,
    }
    return prog


# ---------------------------------------------------------------------------
# Keccak front end
# ---------------------------------------------------------------------------

def _rho_rc():
    rot = [[0] * 5 for _ in range(5)]
    x, y = 1, 0
    for t in range(24):
        rot[x][y] = ((t + 1) * (t + 2) // 2) % 64
        x, y = y, (2 * x + 3 * y) % 5

    def lfsr(t):
        r = 1
        for _ in range(t % 255):
            r <<= 1
            if r & 0x100:
                r ^= 0x171
        return r & 1

    rc = []
    for i in range(24):
        v = 0
        for j in range(7):
            if lfsr(j + 7 * i):
                v |= 1 << ((1 << j) - 1)
        rc.append(v)
    return rot, rc


ROT, RC = _rho_rc()


def rotl(v, n):
    n %= 64
    return ((v << n) | (v >> (64 - n))) & MASK64 if n else v


def keccak_f_ref(A):
    A = list(A)
    for r in range(24):
        C = [A[x] ^ A[x + 5] ^ A[x + 10] ^ A[x + 15] ^ A[x + 20] for x in range(5)]
        D = [C[(x + 4) % 5] ^ rotl(C[(x + 1) % 5], 1) for x in range(5)]
        A = [A[i] ^ D[i % 5] for i in range(25)]
        B = [0] * 25
        for x in range(5):
            for y in range(5):
                B[y + 5 * ((2 * x + 3 * y) % 5)] = rotl(A[x + 5 * y], ROT[x][y])
        A = [B[x + 5 * y] ^ ((~B[(x + 1) % 5 + 5 * y]) & B[(x + 2) % 5 + 5 * y] & MASK64)
             for y in range(5) for x in range(5)]
        A[0] ^= RC[r]
    return A


def left_encode(x: int) -> bytes:
    n = max(1, (x.bit_length() + 7) // 8)
    return bytes([n]) + x.to_bytes(n, "big")


def cshake_init_state(custom: bytes):
    prefix = left_encode(RATE) + left_encode(0) + left_encode(len(custom) * 8) + custom
    prefix = prefix.ljust(RATE, b"\0")
    st = [0] * 25
    for i in range(RATE // 8):
        st[i] ^= int.from_bytes(prefix[8 * i:8 * i + 8], "little")
    return keccak_f_ref(st)


def keccak_f_circuit(c: Circuit, lanes):
    """lanes: 25 lists of 64 wires (bit i of the lane at index i)."""
    A = [list(l) for l in lanes]

    def rot_lane(l, n):
        return [l[(i - n) % 64] for i in range(64)]

    for r in range(24):
        C = [[c.xor_n([A[x + 5 * y][i] for y in range(5)]) for i in range(64)] for x in range(5)]
        D = []
        for x in range(5):
            rl = rot_lane(C[(x + 1) % 5], 1)
            D.append([c.xor(C[(x + 4) % 5][i], rl[i]) for i in range(64)])
        A = [[c.xor(A[i][b], D[i % 5][b]) for b in range(64)] for i in range(25)]
        B = [None] * 25
        for x in range(5):
            for y in range(5):
                B[y + 5 * ((2 * x + 3 * y) % 5)] = rot_lane(A[x + 5 * y], ROT[x][y])
        A = [None] * 25
        for y in range(5):
            for x in range(5):
                n1 = B[(x + 1) % 5 + 5 * y]
                n2 = B[(x + 2) % 5 + 5 * y]
                b0 = B[x + 5 * y]
                A[x + 5 * y] = [c.xor(b0[i], c.and_(c.not_(n1[i]), n2[i])) for i in range(64)]
        rc = RC[r]
        A[0] = [c.not_(A[0][i]) if (rc >> i) & 1 else A[0][i] for i in range(64)]
    return A


def absorb_block_lanes(c: Circuit, st0, block_bits):
    """st0: 25 const lane ints. block_bits: dict byte_index -> list of 8 wires (const wires 0/1 allowed);
    bytes not present are zero. Returns lanes (25 x 64 wires) = st0 XOR block."""
    lanes = []
    for L in range(25):
        bits = []
        for bit in range(64):
            base = (st0[L] >> bit) & 1
            byte = 8 * L + bit // 8
            w = block_bits.get(byte, [0] * 8)[bit % 8] if byte < RATE else 0
            bits.append(c.xor(w, 1 if base else 0))
        lanes.append(bits)
    return lanes


def byte_wires_const(v: int):
    return [(v >> k) & 1 for k in range(8)]


def state_bytes_wires(lanes, n_bytes):
    out = []
    for i in range(n_bytes):
        L, off = divmod(i, 8)
        out.append([lanes[L][8 * off + k] for k in range(8)])
    return out


# ---------------------------------------------------------------------------
# kHeavyHash reference + job (matrix is the bench matrix from kaspa_bench_offload_pico.py)
# ---------------------------------------------------------------------------

def cshake256(data: bytes, custom: bytes, length: int = 32) -> bytes:
    h = cSHAKE256.new(custom=custom)
    h.update(data)
    return h.read(length)


def gf2_rank_64(m: np.ndarray) -> int:
    rows = [sum(int(m[i][j] & 1) << j for j in range(64)) for i in range(64)]
    rank = 0
    for col in range(64):
        p = next((r for r in range(rank, 64) if (rows[r] >> col) & 1), None)
        if p is None:
            continue
        rows[rank], rows[p] = rows[p], rows[rank]
        for r in range(64):
            if r != rank and (rows[r] >> col) & 1:
                rows[r] ^= rows[rank]
        rank += 1
    return rank


def generate_matrix(seed32: bytes, max_attempts: int = 1000) -> np.ndarray:
    state = list(struct.unpack("<4Q", seed32))
    if not any(state):
        raise ValueError("all-zero xoshiro seed")
    def next64():
        a, b, c, d = state
        result = (a + rotl((a + d) & MASK64, 23)) & MASK64
        t = (b << 17) & MASK64
        c ^= a
        d ^= b
        b ^= c
        a ^= d
        c ^= t
        d = rotl(d, 45)
        state[:] = [a, b, c, d]
        return result
    def rank(m):
        a = m.astype(np.float64).copy()
        selected = [False] * 64
        count = 0
        for i in range(64):
            j = next((j for j in range(64) if not selected[j] and abs(a[j,i]) > 1e-9), None)
            if j is None:
                continue
            count += 1
            selected[j] = True
            a[j,i+1:] /= a[j,i]
            for k in range(64):
                if k != j and abs(a[k,i]) > 1e-9:
                    a[k,i+1:] -= a[j,i+1:] * a[k,i]
        return count
    for _ in range(max_attempts):
        values = []
        for _ in range(256):
            v = next64()
            values.extend((v >> (4*k)) & 15 for k in range(16))
        m = np.array(values, dtype=np.uint64).reshape(64,64)
        if rank(m) == 64:
            return m
    raise RuntimeError("no full-rank matrix")


def heavy_ref(pow_hash: bytes, matrix: np.ndarray) -> bytes:
    v = np.array([x for b in pow_hash for x in (b >> 4, b & 15)], dtype=np.uint64)
    p = [int((matrix[i] * v).sum()) >> 10 for i in range(64)]
    prod = bytes((((p[2 * i] & 15) << 4) | (p[2 * i + 1] & 15)) for i in range(32))
    return bytes(a ^ b for a, b in zip(pow_hash, prod))


def kheavy_ref(pre_pow: bytes, ts: int, nonce: int, matrix: np.ndarray) -> bytes:
    pw = cshake256(pre_pow + struct.pack("<Q", ts) + bytes(32) + struct.pack("<Q", nonce), b"ProofOfWorkHash")
    return cshake256(heavy_ref(pw, matrix), b"HeavyHash")[::-1]


def target_from_bits(bits: int) -> bytes:
    return ((1 << (256 - bits)) - 1).to_bytes(32, "big")


# ---------------------------------------------------------------------------
# Stage builders
# ---------------------------------------------------------------------------

def build_matvec_xor(c: Circuit, matrix: np.ndarray, pow_bits):
    """pow_bits[byte][k] wires -> transformed bits[byte][k] wires. Matrix is constant-folded."""
    VW = 14  # sum < 2^14, only bits 10..13 are used
    nib = []
    for i in range(64):
        cols = [[] for _ in range(VW)]
        for j in range(64):
            m = int(matrix[i][j]) & 15
            byte, hi = j // 2, (j % 2 == 0)
            for a in range(4):
                if (m >> a) & 1:
                    for b in range(4):
                        cols[a + b].append(pow_bits[byte][(4 if hi else 0) + b])
        # Wallace layers (3:2 compressors) until every column has <= 2 bits
        while max(len(x) for x in cols) > 2:
            new = [[] for _ in range(VW)]
            for col in range(VW):
                bits = cols[col]
                k = 0
                while len(bits) - k >= 3:
                    s, cy = c.fa(bits[k], bits[k + 1], bits[k + 2])
                    new[col].append(s)
                    if col + 1 < VW:
                        new[col + 1].append(cy)
                    k += 3
                new[col].extend(bits[k:])
            cols = new
        # final ripple add of two rows; sum bits below 10 are not needed, only carries
        carry, out = 0, [0] * VW
        for col in range(VW):
            bits = [w for w in cols[col] if w != 0] + ([carry] if carry != 0 else [])
            need = col >= 10
            if len(bits) == 0:
                out[col], carry = 0, 0
            elif len(bits) == 1:
                out[col], carry = bits[0], 0
            elif len(bits) == 2:
                s, carry = c.ha(bits[0], bits[1], need)
                out[col] = s
            else:
                s, carry = c.fa(bits[0], bits[1], bits[2], need)
                out[col] = s
        nib.append(out[10:14])  # nibble bit0..bit3
    res = []
    for i in range(32):
        row = []
        for k in range(8):
            src = nib[2 * i][k - 4] if k >= 4 else nib[2 * i + 1][k]
            row.append(c.xor(pow_bits[i][k], src))
        res.append(row)
    return res


def build_compare_le(c: Circuit, hash_bytes_bits, target: bytes):
    """hit = (hash <= target), big-endian byte compare. hash_bytes_bits[byte][k] wires."""
    le = 1  # equal -> true
    for byte in range(31, -1, -1):
        for k in range(8):  # LSB first
            h = hash_bytes_bits[byte][k]
            t = (target[byte] >> k) & 1
            if t:
                le = c.or_(c.not_(h), le)
            else:
                le = c.and_(c.not_(h), le)
    return le


def compile_heavy(matrix: np.ndarray) -> Program:
    c = Circuit(256)
    ins = c.inputs()
    pow_bits = [[ins[8 * i + k] for k in range(8)] for i in range(32)]
    out = build_matvec_xor(c, matrix, pow_bits)
    return finalise(c, {"transformed": [out[i][k] for i in range(32) for k in range(8)]})


def compile_job(pre_pow: bytes, ts: int, target: bytes, matrix: np.ndarray) -> Program:
    c = Circuit(64)
    nonce = c.inputs()
    st_pow0 = cshake_init_state(b"ProofOfWorkHash")
    st_heavy0 = cshake_init_state(b"HeavyHash")

    blk = {i: byte_wires_const(pre_pow[i]) for i in range(32)}
    for i in range(8):
        blk[32 + i] = byte_wires_const((ts >> (8 * i)) & 0xFF)
        blk[72 + i] = [nonce[8 * i + k] for k in range(8)]
    blk[80] = byte_wires_const(0x04)
    blk[RATE - 1] = byte_wires_const(0x80)
    lanes = keccak_f_circuit(c, absorb_block_lanes(c, st_pow0, blk))
    pow_bits = state_bytes_wires(lanes, 32)

    transformed = build_matvec_xor(c, matrix, pow_bits)

    blk2 = {i: transformed[i] for i in range(32)}
    blk2[32] = byte_wires_const(0x04)
    blk2[RATE - 1] = byte_wires_const(0x80)
    lanes2 = keccak_f_circuit(c, absorb_block_lanes(c, st_heavy0, blk2))
    fin = state_bytes_wires(lanes2, 32)
    h = [fin[31 - i] for i in range(32)]  # Kaspa compares the byte-reversed hash
    hit = build_compare_le(c, h, target)
    return finalise(c, {"hash": [h[i][k] for i in range(32) for k in range(8)], "hit": [hit]})


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------

def simulate_bits(prog: Program, input_words: list[int], nvec: int) -> list[int]:
    """Bit-parallel digital simulation: each wire is an int whose bit n is the value in vector n."""
    mask = (1 << nvec) - 1
    w = [0, mask] + list(input_words) + [0] * prog.n_gates
    base = 2 + prog.n_in
    ops, A, B = prog.ops, prog.a, prog.b
    for i in range(prog.n_gates):
        x, y, op = w[A[i]], w[B[i]], ops[i]
        if op == AND:
            r = x & y
        elif op == OR:
            r = x | y
        elif op == XOR:
            r = x ^ y
        elif op == NAND:
            r = ~(x & y) & mask
        elif op == NOR:
            r = ~(x | y) & mask
        else:
            r = ~(x ^ y) & mask
        w[base + i] = r
    return w


def simulate_optical(prog: Program, input_bits: list[int], sigma: float, rng) -> tuple[list[int], int]:
    """Analog lane model: s = a + b + N(0, sigma) classified into 0/1/2 by thresholds 0.5 / 1.5 on
    the calibrated scale. Returns (wire values, number of gate decisions that differed from ideal)."""
    base = 2 + prog.n_in
    w = [0, 1] + list(input_bits) + [0] * prog.n_gates
    wrong = 0
    tt = {op: TT[op] for op in TT}
    for i in range(prog.n_gates):
        a, b, op = w[prog.a[i]], w[prog.b[i]], prog.ops[i]
        s = a + b + rng.gauss(0.0, sigma)
        cls = 0 if s < 0.5 else (1 if s < 1.5 else 2)
        out = tt[op][cls]
        ideal = tt[op][a + b]
        if out != ideal:
            wrong += 1
        w[base + i] = out
    return w, wrong


def pack_inputs(values: list[int], nbits: int, per_vector_bits):
    words = [0] * nbits
    for n, v in enumerate(values):
        bits = per_vector_bits(v)
        for k in range(nbits):
            if bits[k]:
                words[k] |= 1 << n
    return words


def read_bytes(w: list[int], wires: list[int], nvec: int):
    out = []
    for n in range(nvec):
        bs = bytearray(len(wires) // 8)
        for idx, wire in enumerate(wires):
            if (w[wire] >> n) & 1:
                bs[idx // 8] |= 1 << (idx % 8)
        out.append(bytes(bs))
    return out


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------

def matrix_checksum(matrix: np.ndarray) -> int:
    h = 2166136261
    for i in range(64):
        for j in range(64):
            h = ((h ^ (int(matrix[i][j]) & 15)) * 16777619) & 0xFFFFFFFF
    return h


def emit_c_header(prog: Program, path: str, pre_pow: bytes, matrix: np.ndarray):
    assert "transformed" in prog.outputs and prog.n_in == 256, "C header is for the heavy stage"
    lines = ["// Generated by kaspa_gatec.py - heavy stage gate program (matrix x vector + XOR).",
             "// Inputs: wires 2..257 = pow_hash bits, wire 2+8*byte+bit.  Wires 0/1 = const 0/1.",
             "// Gate g produces wire GP_FIRST_GATE_WIRE+g. word0 = (op<<28)|a, word1 = b.",
             "// op: 0 AND 1 OR 2 XOR 3 NAND 4 NOR 5 XNOR. Gates are in wave order; a wave is <= 8 lanes.",
             "#pragma once", "#include <stdint.h>",
             f"#define GP_N_GATES {prog.n_gates}u",
             f"#define GP_N_WAVES {len(prog.wave_end)}u",
             f"#define GP_N_WIRES {prog.n_wires}u",
             f"#define GP_FIRST_GATE_WIRE {2 + prog.n_in}u",
             f"#define GP_MATRIX_FNV 0x{matrix_checksum(matrix):08x}u",
             "static const uint8_t GP_PREPOW[32] = {" + ",".join(str(b) for b in pre_pow) + "};"]
    words = []
    for i in range(prog.n_gates):
        words.append((prog.ops[i] << 28) | prog.a[i])
        words.append(prog.b[i])
    lines.append("static const uint32_t GP_GATES[%d] = {" % len(words))
    for k in range(0, len(words), 12):
        lines.append(",".join(f"0x{x:08x}" for x in words[k:k + 12]) + ",")
    lines.append("};")
    lines.append("static const uint32_t GP_WAVE_END[%d] = {" % len(prog.wave_end))
    for k in range(0, len(prog.wave_end), 16):
        lines.append(",".join(str(x) for x in prog.wave_end[k:k + 16]) + ",")
    lines.append("};")
    outs = prog.outputs["transformed"]
    lines.append("static const uint32_t GP_OUT[256] = {")
    for k in range(0, 256, 16):
        lines.append(",".join(str(x) for x in outs[k:k + 16]) + ",")
    lines.append("};")
    open(path, "w").write("\n".join(lines) + "\n")


def emit_json(prog: Program, path: str):
    json.dump({"n_in": prog.n_in, "ops": prog.ops, "a": prog.a, "b": prog.b,
               "wave_end": prog.wave_end, "outputs": prog.outputs, "stats": prog.stats}, open(path, "w"))


def estimate_time(prog: Program, slot_us: float, discharge_us: float) -> float:
    per_wave = 2 * slot_us + discharge_us + LANES * 12  # two slots, reset, 8 ADC reads
    return len(prog.wave_end) * per_wave / 1e6


# ---------------------------------------------------------------------------
# Hashes-per-second statistics
# ---------------------------------------------------------------------------

ADC_READ_US = 12.0  # per lane: mux settle (8 us) + conversion, as used by estimate_time()


def hashrate_report(prog: Program, stage: str, slot_us: float, discharge_us: float,
                    matrix, pre_pow: bytes, ts: int):
    unit = "hash" if stage == "job" else "matvec+xor stage (not a full hash)"
    waves = len(prog.wave_end)
    print(f"\n=== throughput stats ({unit}) ===")
    print(f"circuit: {prog.n_gates} gates, depth {prog.depth}, {waves} waves x {LANES} lanes")

    scenarios = [
        ("your settings", slot_us, discharge_us),
        ("fast LED + photodiode (1 us slot, 10 us discharge)", 1.0, 10.0),
        ("LCD/ITO-class (40 us slot, 2 ms discharge)", 40.0, 2000.0),
    ]
    print(f"{'scenario':58s} {'s/unit':>12s} {'units/s':>12s}")
    for name, sl, di in scenarios:
        t = estimate_time(prog, sl, di)
        print(f"{name:58s} {t:12.4g} {1.0 / t:12.4g}")
    floor = waves * LANES * ADC_READ_US / 1e6
    print(f"readout floor (waves x {LANES} ADC reads x {ADC_READ_US:g} us, even with zero optical time): "
          f"{floor:.3g} s/unit = {1.0 / floor:.3g} units/s")
    if slot_us < 0.1 or discharge_us < 0.1:
        print("note: your slot/discharge times are far below what any pixel, photodiode or ADC can do; "
              "the readout floor is what actually limits you.")
    print(f"scaling: N independent benches multiply the rate by N (e.g. 100 benches -> "
          f"{100.0 / estimate_time(prog, slot_us, discharge_us):.3g} units/s at your settings)")

    # measured: software simulation of this circuit, and the reference implementation
    nvec = 256
    rng = random.Random(3)
    if stage == "job":
        vals = [rng.getrandbits(64) for _ in range(nvec)]
        words = pack_inputs(vals, 64, lambda v: [(v >> k) & 1 for k in range(64)])
    else:
        vals = [bytes(rng.randrange(256) for _ in range(32)) for _ in range(nvec)]
        words = pack_inputs(vals, 256, lambda v: [(v[k // 8] >> (k % 8)) & 1 for k in range(256)])
    t0 = time.perf_counter()
    simulate_bits(prog, words, nvec)
    dt = time.perf_counter() - t0
    print(f"\nmeasured on this PC: bit-parallel gate simulation, {nvec} vectors in {dt:.3f} s "
          f"= {nvec / dt:,.0f} {'hashes' if stage == 'job' else 'stage evals'}/s")
    t0 = time.perf_counter()
    n_ref = 20
    for i in range(n_ref):
        if stage == "job":
            kheavy_ref(pre_pow, ts, i, matrix)
        else:
            heavy_ref(vals[i], matrix)
    dt = time.perf_counter() - t0
    print(f"measured on this PC: Python reference (pycryptodome + numpy), {n_ref / dt:,.0f} "
          f"{'hashes' if stage == 'job' else 'stage evals'}/s")
    print("Pico digital rate is not measured here: run RUN on the board and read bench_hashrate_approx.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Compile a kHeavyHash job to an optical-bench logic-gate circuit")
    ap.add_argument("--stage", choices=["heavy", "job"], default="heavy")
    ap.add_argument("--pre-pow-hash", help="64 hex chars; default is the bench demo template")
    ap.add_argument("--timestamp", type=lambda x: int(x, 0), default=1700000000)
    ap.add_argument("--leading-zero-bits", type=int, default=12)
    ap.add_argument("--target-hex")
    ap.add_argument("--verify", action="store_true", help="bit-parallel check against the reference hash")
    ap.add_argument("--vectors", type=int, default=64)
    ap.add_argument("--optical-sim", action="store_true", help="analog lane model with Gaussian noise")
    ap.add_argument("--noise", type=float, default=0.08, help="sigma in units of one pixel level (0..2 scale)")
    ap.add_argument("--slot-us", type=float, default=1.0)
    ap.add_argument("--discharge-us", type=float, default=10.0)
    ap.add_argument("--stats", action="store_true", help="hashes-per-second report (optical estimate + measured software)")
    ap.add_argument("--emit-c", help="write Pico gate-program header (heavy stage)")
    ap.add_argument("--emit-json")
    args = ap.parse_args()

    pre_pow = (bytes.fromhex(args.pre_pow_hash) if args.pre_pow_hash
               else cshake256(b"kaspa-optical-bench-demo-template", b"ProofOfWorkHash"))
    target = bytes.fromhex(args.target_hex) if args.target_hex else target_from_bits(args.leading_zero_bits)
    matrix = generate_matrix(pre_pow)

    t0 = time.time()
    prog = compile_heavy(matrix) if args.stage == "heavy" else compile_job(pre_pow, args.timestamp, target, matrix)
    print(f"compiled stage '{args.stage}' in {time.time() - t0:.1f}s")
    print(json.dumps(prog.stats, indent=2))
    print(f"estimated optical time per {'matvec+xor' if args.stage == 'heavy' else 'hash'}: "
          f"{estimate_time(prog, args.slot_us, args.discharge_us):.2f} s "
          f"(slot {args.slot_us:g} us, discharge {args.discharge_us:g} us, {LANES} lanes)")

    if args.stats:
        hashrate_report(prog, args.stage, args.slot_us, args.discharge_us, matrix, pre_pow, args.timestamp)

    ok = True
    if args.verify:
        n = args.vectors
        rng = random.Random(1)
        if args.stage == "heavy":
            vecs = [bytes(rng.randrange(256) for _ in range(32)) for _ in range(n)]
            words = pack_inputs(vecs, 256, lambda v: [(v[k // 8] >> (k % 8)) & 1 for k in range(256)])
            w = simulate_bits(prog, words, n)
            got = read_bytes(w, prog.outputs["transformed"], n)
            bad = sum(1 for v, g in zip(vecs, got) if g != heavy_ref(v, matrix))
        else:
            nonces = [rng.getrandbits(64) for _ in range(n)]
            words = pack_inputs(nonces, 64, lambda v: [(v >> k) & 1 for k in range(64)])
            w = simulate_bits(prog, words, n)
            hashes = read_bytes(w, prog.outputs["hash"], n)
            bad = hits = 0
            for i, nn in enumerate(nonces):
                ref = kheavy_ref(pre_pow, args.timestamp, nn, matrix)
                ref_hit = int.from_bytes(ref, "big") <= int.from_bytes(target, "big")
                hit = (w[prog.outputs["hit"][0]] >> i) & 1
                hits += hit
                if hashes[i] != ref or bool(hit) != ref_hit:
                    bad += 1
            print(f"hit bits set in {hits}/{n} vectors")
        ok = bad == 0
        print(f"VERIFY: {n - bad}/{n} vectors match the reference ->", "PASS" if ok else "FAIL")

    if args.optical_sim:
        rng = random.Random(7)
        trials, tot_wrong, bad_out = 8, 0, 0
        for _ in range(trials):
            if args.stage == "heavy":
                v = bytes(rng.randrange(256) for _ in range(32))
                bits = [(v[k // 8] >> (k % 8)) & 1 for k in range(256)]
                w, wrong = simulate_optical(prog, bits, args.noise, rng)
                got = bytes(sum(w[prog.outputs["transformed"][8 * i + k]] << k for k in range(8)) for i in range(32))
                bad_out += got != heavy_ref(v, matrix)
            else:
                nn = rng.getrandbits(64)
                bits = [(nn >> k) & 1 for k in range(64)]
                w, wrong = simulate_optical(prog, bits, args.noise, rng)
                got = bytes(sum(w[prog.outputs["hash"][8 * i + k]] << k for k in range(8)) for i in range(32))
                bad_out += got != kheavy_ref(pre_pow, args.timestamp, nn, matrix)
            tot_wrong += wrong
        print(f"OPTICAL SIM sigma={args.noise}: gate error rate {tot_wrong / (trials * prog.n_gates):.2e}, "
              f"wrong final outputs {bad_out}/{trials}")

    if args.emit_c:
        emit_c_header(prog, args.emit_c, pre_pow, matrix)
        print(f"wrote {args.emit_c}")
    if args.emit_json:
        emit_json(prog, args.emit_json)
        print(f"wrote {args.emit_json}")
    return 0 if ok else 1



# Live mining uses the node's unmodified block and compact network target.
def header_digest(h, *, pre_pow=False):
    import hashlib
    u64 = lambda v: struct.pack("<Q", int(v))
    parents = h["parentsByLevel"]
    data = struct.pack("<H", int(h["version"])) + u64(len(parents))
    for level in parents:
        if isinstance(level, dict):
            level = level["parentHashes"]
        data += u64(len(level)) + b"".join(bytes.fromhex(x) for x in level)
    for key in ("hashMerkleRoot", "acceptedIdMerkleRoot", "utxoCommitment"):
        data += bytes.fromhex(h[key])
    data += u64(0 if pre_pow else h["timestamp"])
    data += struct.pack("<I", int(h["bits"]))
    data += u64(0 if pre_pow else h["nonce"])
    data += u64(h["daaScore"]) + u64(h["blueScore"])
    work = int(h["blueWork"] or "0", 16)
    wb = work.to_bytes((work.bit_length()+7)//8, "big")
    data += u64(len(wb)) + wb + bytes.fromhex(h["pruningPoint"])
    return hashlib.blake2b(data, digest_size=32, key=b"BlockHash").digest()


def compact_target(bits):
    bits = int(bits)
    exponent, mantissa = bits >> 24, bits & 0x7fffff
    if bits & 0x800000:
        raise ValueError("negative compact target")
    target = mantissa >> (8*(3-exponent)) if exponent <= 3 else mantissa << (8*(exponent-3))
    if not 0 < target < 1 << 256:
        raise ValueError("invalid compact target")
    return target


def scan_batch(pre, ts, matrix, target, nonce, count, prog):
    nonces = [(nonce+i) & MASK64 for i in range(count)]
    if prog is None:
        hashes = [kheavy_ref(pre, ts, n, matrix) for n in nonces]
    else:
        words = pack_inputs(nonces, 64, lambda n: [(n >> k)&1 for k in range(64)])
        wires = simulate_bits(prog, words, count)
        hashes = read_bytes(wires, prog.outputs["hash"], count)
    values = [int.from_bytes(h, "big") for h in hashes]
    hits = [(n, v) for n,v in zip(nonces,values) if v <= target]
    return hits, min(values)


async def live_main(a):
    import asyncio, copy, csv, math, secrets
    try:
        from kaspa import RpcClient, Resolver
    except ImportError:
        raise SystemExit("Install dependencies: pip install kaspa numpy pycryptodome")
    client = (RpcClient(url=a.rpc_url, network_id=a.network) if a.rpc_url
              else RpcClient(resolver=Resolver(), network_id=a.network))
    await client.connect()
    started = time.perf_counter()
    total = jobs = accepted = rejected = candidates = 0
    hash_seconds = compile_seconds = rpc_seconds = expected_hits = 0.0
    best_ratio = 0.0
    last_report = started
    job = None
    nonce = secrets.randbits(64)
    rows = []
    print(f"LIVE {a.network} | engine={a.engine} | submit={a.submit}", flush=True)
    try:
        print("Node:", await client.get_server_info(), flush=True)
        while not a.seconds or time.perf_counter()-started < a.seconds:
            now = time.perf_counter()
            if job is None or now >= refresh_at:
                t = time.perf_counter()
                response = await client.get_block_template({"payAddress": a.pay_address,
                                                            "extraData": list(b"kaspa-gatec-live")})
                rpc_seconds += time.perf_counter()-t
                if not response.get("isSynced", False):
                    raise RuntimeError("Node is not synced; refusing to mine")
                block = response["block"]
                h = block["header"]
                pre = header_digest(h, pre_pow=True)
                ts, target = int(h["timestamp"]), compact_target(h["bits"])
                key = (pre, ts, target)
                if key != job:
                    t = time.perf_counter()
                    matrix = generate_matrix(pre)
                    prog = compile_job(pre, ts, target.to_bytes(32,"big"), matrix) if a.engine == "gates" else None
                    if prog:
                        check = [0, 1, MASK64]
                        words = pack_inputs(check,64,lambda n:[(n>>k)&1 for k in range(64)])
                        w = simulate_bits(prog,words,len(check))
                        got = read_bytes(w,prog.outputs["hash"],len(check))
                        if got != [kheavy_ref(pre,ts,n,matrix) for n in check]:
                            raise RuntimeError("Gate/reference mismatch; mining stopped")
                        print("CIRCUIT", json.dumps(prog.stats), flush=True)
                        optical = estimate_time(prog,a.slot_us,a.discharge_us)
                        print(f"OPTICAL ESTIMATE ONLY: {optical:.6g} s/hash, {1/optical:.6g} H/s",flush=True)
                    compile_seconds += time.perf_counter()-t
                    jobs += 1
                    job = key
                    nonce = secrets.randbits(64)
                    print(f"JOB {jobs} prePoW={pre.hex()} timestamp_ms={ts} "
                          f"bits=0x{int(h['bits']):08x} target={target:064x} "
                          f"transactions={len(block['transactions'])}", flush=True)
                refresh_at = time.perf_counter()+a.refresh
            t = time.perf_counter()
            hits, best = await asyncio.to_thread(scan_batch,pre,ts,matrix,target,nonce,a.batch,prog)
            dt = time.perf_counter()-t
            hash_seconds += dt
            total += a.batch
            expected_hits += a.batch*(target+1)/(1 << 256)
            best_ratio = max(best_ratio,target/max(1,best))
            nonce = (nonce+a.batch)&MASK64
            for nn, value in hits:
                if int.from_bytes(kheavy_ref(pre,ts,nn,matrix),"big") != value:
                    raise RuntimeError("Candidate reference verification failed")
                candidates += 1
                candidate = copy.deepcopy(block)
                candidate["header"]["nonce"] = nn
                candidate["header"].pop("hash",None)
                bh = header_digest(candidate["header"]).hex()
                path = f"candidate-{bh}.json"
                with open(path,"w") as f:
                    json.dump(candidate,f,indent=2)
                print(f"NETWORK TARGET MET nonce={nn} block_hash={bh} saved={path}",flush=True)
                if a.submit:
                    result = await client.submit_block({"block":candidate,"allowNonDAABlocks":False})
                    print("SUBMIT RESPONSE",json.dumps(result),flush=True)
                    report = result.get("report")
                    if report == "success" or report == "Success":
                        accepted += 1
                        print("NODE ACCEPTED",bh,flush=True)
                    else:
                        rejected += 1
                        print("NOT CONFIRMED ACCEPTED: inspect response",flush=True)
                job = None
            now = time.perf_counter()
            if now-last_report >= a.stats_interval:
                wall = now-started
                rate = total/wall
                eta = (1 << 256)/(target+1)/rate if rate else float("inf")
                row = dict(elapsed_s=wall,hashes=total,wall_hps=rate,
                           hashing_hps=total/max(hash_seconds,1e-12),last_batch_hps=a.batch/dt,
                           jobs=jobs,compile_s=compile_seconds,rpc_s=rpc_seconds,
                           best_target_ratio=best_ratio,candidates=candidates,
                           node_accepted=accepted,unconfirmed_or_rejected=rejected,
                           expected_seconds_per_block=eta,
                           probability_at_least_one= -math.expm1(-expected_hits))
                rows.append(row)
                print("PERF",json.dumps(row),flush=True)
                last_report = now
    finally:
        await client.disconnect()
        wall = time.perf_counter()-started
        summary = dict(network=a.network,engine=a.engine,elapsed_s=wall,hashes=total,
                       wall_hps=total/max(wall,1e-12),hashing_hps=total/max(hash_seconds,1e-12),
                       jobs=jobs,compile_s=compile_seconds,rpc_s=rpc_seconds,
                       candidates=candidates,node_accepted=accepted,
                       unconfirmed_or_rejected=rejected,best_target_ratio=best_ratio)
        with open(a.report,"w") as f:
            json.dump({"summary":summary,"samples":rows},f,indent=2)
        if rows:
            with open(a.csv,"w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        print("FINAL",json.dumps(summary),flush=True)


def entry():
    if "--mine" not in sys.argv:
        return main()
    import asyncio
    ap=argparse.ArgumentParser(description="Live Kaspa solo miner using real node templates and network difficulty")
    ap.add_argument("--mine",action="store_true")
    ap.add_argument("--pay-address",required=True)
    ap.add_argument("--rpc-url")
    ap.add_argument("--network",default="mainnet")
    ap.add_argument("--engine",choices=["reference","gates"],default="reference")
    ap.add_argument("--submit",action="store_true",help="Explicitly permit submitting network-target candidates")
    ap.add_argument("--seconds",type=float,default=60000,help="0 = run until interrupted")
    ap.add_argument("--refresh",type=float,default=0.25)
    ap.add_argument("--batch",type=int,default=256)
    ap.add_argument("--stats-interval",type=float,default=2)
    ap.add_argument("--slot-us",type=float,default=1.0)
    ap.add_argument("--discharge-us",type=float,default=10.0)
    ap.add_argument("--report",default="mining-stats.json")
    ap.add_argument("--csv",default="mining-stats.csv")
    a=ap.parse_args()
    if a.batch < 1 or a.seconds < 0 or a.refresh <= 0 or a.stats_interval <= 0:
        ap.error("batch/refresh/stats-interval must be positive; seconds must be >= 0")
    try:
        asyncio.run(live_main(a))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(entry())