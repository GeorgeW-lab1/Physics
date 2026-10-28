#!/usr/bin/env python3
"""Photonic Pearlhash-style PoUW simulator (software model, not a live miner).

MAC and adder exposure counters are driven by configurable simulated cell banks.
They count modeled cell activations, not measured photons or physical energy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    import blake3  # type: ignore
    HAVE_BLAKE3 = True
except ImportError:
    HAVE_BLAKE3 = False

MASK32 = 0xFFFFFFFF
MASK512 = (1 << 512) - 1

def digest32(data: bytes) -> bytes:
    return blake3.blake3(data).digest(32) if HAVE_BLAKE3 else hashlib.blake2b(data, digest_size=32).digest()

def leading_zero_bits(d: bytes) -> int:
    n = 0
    for byte in d:
        if byte:
            return n + (8 - byte.bit_length())
        n += 8
    return n

@dataclass
class CellBank:
    """A configurable pool of simulated cells with activation/exposure accounting."""
    cell_count: int
    exposure_cost: float = 1.0
    name: str = "cell"
    activations: int = 0
    weighted_exposures: float = 0.0

    def __post_init__(self):
        if self.cell_count < 1:
            raise ValueError(f"{self.name} cell count must be at least 1")
        if self.exposure_cost < 0:
            raise ValueError("exposure costs cannot be negative")

    def activate(self, operations: int = 1) -> None:
        if operations < 0:
            raise ValueError("operations cannot be negative")
        # Each operation is assigned to a cell in the configured pool. Reuse is
        # allowed; activations count work, while cell_count describes capacity.
        self.activations += operations
        self.weighted_exposures += operations * self.exposure_cost

    def report(self) -> dict:
        return {
            "cell_count": self.cell_count,
            "activations": self.activations,
            "exposure_cost_per_activation": self.exposure_cost,
            "weighted_exposures": round(self.weighted_exposures, 6),
        }

class PhotonicLogicCell:
    @staticmethod
    def xor(a: int, b: int) -> int: return (a ^ b) & 1
    @staticmethod
    def and_(a: int, b: int) -> int: return a & b
    @staticmethod
    def or_(a: int, b: int) -> int: return a | b

class PhotonicFullAdder:
    @staticmethod
    def add(a: int, b: int, carry_in: int) -> tuple[int, int]:
        if not all(x in (0, 1) for x in (a, b, carry_in)):
            raise ValueError("full-adder inputs must be binary")
        axb = PhotonicLogicCell.xor(a, b)
        s = PhotonicLogicCell.xor(axb, carry_in)
        carry_out = PhotonicLogicCell.or_(PhotonicLogicCell.and_(a, b), PhotonicLogicCell.and_(carry_in, axb))
        return s, carry_out

class PhotonicAdderArray:
    """Width-wide ripple adder; every bit stage activates configured gate cells."""
    def __init__(self, width: int = 32, cell_count: int = 32, exposure_cost: float = 1.0):
        if width < 1: raise ValueError("adder width must be positive")
        self.width = width
        # One full-adder bit stage uses 2 XOR, 2 AND, and 1 OR gate activations.
        self.cells = CellBank(cell_count, exposure_cost, "adder")
        self.bit_activations = 0
        self.gate_activations = {"xor": 0, "and": 0, "or": 0}

    @property
    def exposures(self) -> int:
        """Compatibility alias: number of full-adder bit stages activated."""
        return self.bit_activations

    def add(self, x: int, y: int, wrap: bool = False) -> int:
        if x < 0 or y < 0: raise ValueError("adder operands must be non-negative")
        if not wrap and (x >= (1 << self.width) or y >= (1 << self.width)):
            raise ValueError("operand does not fit adder width")
        result, carry = 0, 0
        for bit in range(self.width):
            a, b = (x >> bit) & 1, (y >> bit) & 1
            s, carry = PhotonicFullAdder.add(a, b, carry)
            result |= s << bit
            self.bit_activations += 1
            self.cells.activate(1)
            self.gate_activations["xor"] += 2
            self.gate_activations["and"] += 2
            self.gate_activations["or"] += 1
        if carry and not wrap: raise OverflowError(f"{self.width}-bit optical addition overflow")
        return result

    def report(self) -> dict:
        return {"width_bits": self.width, "bit_stage_activations": self.bit_activations,
                "gate_activations": dict(self.gate_activations), "cells": self.cells.report()}

class PhotonicInt8Multiplier:
    def __init__(self, adder: PhotonicAdderArray, mac_cells: CellBank):
        self.adder, self.mac_cells = adder, mac_cells

    def multiply(self, a: int, b: int) -> int:
        if not -128 <= a <= 127 or not -128 <= b <= 127: raise ValueError("INT8 operands required")
        negative = (a < 0) != (b < 0)
        ua, ub, product = abs(a), abs(b), 0
        # One MAC cell activation models each signed INT8 product operation.
        self.mac_cells.activate(1)
        for bit in range(8):
            if (ub >> bit) & 1: product = self.adder.add(product, ua << bit)
        return -product if negative else product

class PhotonicMacArray:
    def __init__(self, lanes: int = 64, mac_cells: int | None = None,
                 mac_exposure_cost: float = 1.0, adder_width: int = 32,
                 adder_cells: int = 32, adder_exposure_cost: float = 1.0):
        if lanes < 1: raise ValueError("MAC lanes must be positive")
        self.lanes = lanes
        self.mac_cells = CellBank(mac_cells if mac_cells is not None else lanes, mac_exposure_cost, "MAC")
        self.adder = PhotonicAdderArray(adder_width, adder_cells, adder_exposure_cost)
        self.multipliers = [PhotonicInt8Multiplier(self.adder, self.mac_cells) for _ in range(lanes)]
        self.accumulator = 0
        self.mac_operations = 0

    def reset(self) -> None: self.accumulator = 0

    def mac(self, a_values: Iterable[int], b_values: Iterable[int]) -> int:
        aa, bb = list(a_values), list(b_values)
        if len(aa) != self.lanes or len(bb) != self.lanes:
            raise ValueError(f"MAC array requires exactly {self.lanes} lane pairs")
        for a, b in zip(aa, bb):
            product = self.multipliers[self.mac_operations % self.lanes].multiply(a, b)
            self.accumulator = self.adder.add(self.accumulator, product & MASK32, wrap=True)
            self.mac_operations += 1
        return self.accumulator

    def report(self) -> dict:
        return {"lanes": self.lanes, "mac_operations": self.mac_operations,
                "mac_cells": self.mac_cells.report(), "adder": self.adder.report()}

def to_int8(values: list[int]) -> list[int]:
    if any(not -128 <= v <= 127 for v in values): raise ValueError("matrix entries must be INT8")
    return values

@dataclass
class PhotonicTileEngine:
    rows: int = 4
    cols: int = 4
    inner: int = 64
    lanes: int = 64
    seed: int = 1
    mac_cells: int | None = None
    mac_exposure_cost: float = 1.0
    adder_width: int = 32
    adder_cells: int = 32
    adder_exposure_cost: float = 1.0

    def __post_init__(self):
        if self.inner % self.lanes: raise ValueError("inner dimension must be divisible by MAC lanes")
        self.mac = PhotonicMacArray(self.lanes, self.mac_cells, self.mac_exposure_cost,
                                    self.adder_width, self.adder_cells, self.adder_exposure_cost)
        self.rng = random.Random(self.seed)

    def random_matrix(self, rows: int, cols: int) -> list[list[int]]:
        return [[self.rng.randint(-128, 127) for _ in range(cols)] for _ in range(rows)]

    def compute_tile(self, a_tile: list[list[int]], b_tile: list[list[int]]) -> list[list[int]]:
        if len(a_tile) != self.rows or len(b_tile) != self.inner: raise ValueError("tile row count mismatch")
        if any(len(r) != self.inner for r in a_tile): raise ValueError("A tile inner dimension mismatch")
        if any(len(r) != self.cols for r in b_tile): raise ValueError("B tile column dimension mismatch")
        result = [[0] * self.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                self.mac.reset()
                for start in range(0, self.inner, self.lanes):
                    av = to_int8(a_tile[i][start:start+self.lanes])
                    bv = to_int8([b_tile[k][j] for k in range(start, start+self.lanes)])
                    result[i][j] = self.mac.mac(av, bv)
        return result

    @staticmethod
    def tile_state(tile: list[list[int]]) -> int:
        state = 0
        for row in tile:
            for value in row:
                state ^= value & MASK32
                state = ((state << 13) | (state >> (512 - 13))) & MASK512
        return state

@dataclass
class PearlJob:
    header: bytes
    matrix_rows: int = 16
    matrix_cols: int = 16
    inner: int = 64
    tile_rows: int = 4
    tile_cols: int = 4
    rank: int = 1
    difficulty_bits: int = 12
    nonce: int = 0

    def __post_init__(self):
        if self.matrix_rows % self.tile_rows or self.matrix_cols % self.tile_cols:
            raise ValueError("matrix dimensions must be divisible by tile dimensions")
        if not 0 <= self.difficulty_bits <= 256: raise ValueError("difficulty must be 0..256")
        if self.rank < 1: raise ValueError("rank must be at least 1")

    @property
    def target(self) -> int: return (1 << (256 - self.difficulty_bits)) - 1
    def encode(self) -> bytes:
        return self.header + struct.pack("<QQQQI", self.nonce, self.matrix_rows, self.matrix_cols, self.inner, self.rank)

@dataclass
class PearlShare:
    header_hash: str
    nonce: int
    difficulty_bits: int
    opened_tiles: list[dict] = field(default_factory=list)
    photonics: dict = field(default_factory=dict)
    def to_json(self) -> str: return json.dumps(self.__dict__, indent=2, sort_keys=True)

class PhotonicPearlMiner:
    def __init__(self, job: PearlJob, lanes: int = 64, seed: int = 1,
                 mac_cells: int | None = None, mac_exposure_cost: float = 1.0,
                 adder_width: int = 32, adder_cells: int = 32, adder_exposure_cost: float = 1.0):
        self.job = job
        self.engine = PhotonicTileEngine(job.tile_rows, job.tile_cols, job.inner, lanes, seed,
                                         mac_cells, mac_exposure_cost, adder_width, adder_cells, adder_exposure_cost)
        self.matrix_a = self.engine.random_matrix(job.matrix_rows, job.inner)
        self.matrix_b = self.engine.random_matrix(job.inner, job.matrix_cols)
        self.tile_states: list[int] = []
        self.tile_digests: list[bytes] = []

    def build_tile_commitments(self):
        self.tile_states.clear(); self.tile_digests.clear()
        for i in range(0, self.job.matrix_rows, self.job.tile_rows):
            for j in range(0, self.job.matrix_cols, self.job.tile_cols):
                a_tile = [row[:] for row in self.matrix_a[i:i+self.job.tile_rows]]
                b_tile = [[self.matrix_b[k][c] for c in range(j, j+self.job.tile_cols)] for k in range(self.job.inner)]
                state = self.engine.tile_state(self.engine.compute_tile(a_tile, b_tile))
                self.tile_states.append(state)
                self.tile_digests.append(digest32(state.to_bytes(64, "little")))

    def merkle_root(self) -> bytes:
        if not self.tile_digests: raise RuntimeError("no tile commitments")
        level = list(self.tile_digests)
        while len(level) > 1:
            if len(level) % 2: level.append(level[-1])
            level = [digest32(level[i] + level[i+1]) for i in range(0, len(level), 2)]
        return level[0]

    def mine(self, max_nonces: int = 1_000_000, verbose: bool = True) -> PearlShare | None:
        started = time.perf_counter()
        for _ in range(max_nonces):
            self.job.nonce = (self.job.nonce + 1) & ((1 << 64) - 1)
            self.build_tile_commitments()
            header_hash = digest32(self.job.encode() + self.merkle_root())
            if int.from_bytes(header_hash, "big") <= self.job.target:
                opened, threshold = [], (1 << 256) // max(1, self.job.rank)
                for idx, digest in enumerate(self.tile_digests):
                    if int.from_bytes(digest, "big") <= threshold:
                        opened.append({"tile": idx, "state": f"{self.tile_states[idx]:0128x}", "digest": digest.hex()})
                report = self.engine.mac.report()
                share = PearlShare(header_hash.hex(), self.job.nonce, self.job.difficulty_bits, opened,
                                   {"mac_operations": report["mac_operations"],
                                    "mac_cells": report["mac_cells"],
                                    "adder": report["adder"], "tiles": len(self.tile_states),
                                    "hash": "BLAKE3" if HAVE_BLAKE3 else "BLAKE2b stand-in"})
                if verbose:
                    print(f"SIMULATED SHARE after {self.job.nonce} nonce value(s)")
                    print(f"header hash: {share.header_hash}")
                    print(f"opened tiles: {len(opened)}")
                    print("photonic cell report:")
                    print(json.dumps(share.photonics, indent=2))
                    print(f"elapsed: {time.perf_counter()-started:.3f}s")
                return share
        return None

def main():
    p = argparse.ArgumentParser(description="Photonic Pearlhash-style PoUW simulator")
    p.add_argument("--difficulty", type=int, default=12)
    p.add_argument("--lanes", type=int, default=64, help="parallel MAC lanes")
    p.add_argument("--mac-cells", type=int, default=None, help="simulated MAC cell pool size (default: lanes)")
    p.add_argument("--mac-exposure-cost", type=float, default=1.0, help="weighted exposure per MAC-cell activation")
    p.add_argument("--adder-width", type=int, default=32, help="adder width in bits")
    p.add_argument("--adder-cells", type=int, default=32, help="simulated adder cell pool size")
    p.add_argument("--adder-exposure-cost", type=float, default=1.0, help="weighted exposure per adder bit-stage activation")
    p.add_argument("--max-nonces", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save", default="output/pearl_share.json")
    args = p.parse_args()
    if args.max_nonces < 1: p.error("--max-nonces must be at least 1")
    try:
        header = digest32(os.urandom(64))
        job = PearlJob(header=header, difficulty_bits=args.difficulty)
        miner = PhotonicPearlMiner(job, args.lanes, args.seed, args.mac_cells,
                                   args.mac_exposure_cost, args.adder_width, args.adder_cells,
                                   args.adder_exposure_cost)
        share = miner.mine(args.max_nonces)
    except ValueError as exc:
        p.error(str(exc))
    if share is None:
        print("No share found in the requested nonce range.")
        return
    path = Path(args.save)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(share.to_json(), encoding="utf-8")
    print(f"Saved share to {path}")

if __name__ == "__main__": main()
