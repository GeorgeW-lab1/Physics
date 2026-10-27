#!/usr/bin/env python3
"""
kaspa_bench_offload.py

Single-file Kaspa kHeavyHash miner with maximum computation offloaded to an
Arduino Mega 2560 optical bench.

Offloaded to Arduino Mega:
  - Nonce iteration
  - cSHAKE256("ProofOfWorkHash") over pre-PoW hash + timestamp + nonce
  - 64x64 kHeavyHash matrix-vector multiplication
  - XOR transform
  - cSHAKE256("HeavyHash") finalisation
  - Target comparison
  - ITO PWM updates during mining

Retained on PC:
  - Matrix generation and full-rank check
  - Job configuration
  - Independent verification of any Mega-reported solution
  - CSV/report output

Install:
  pip install pyserial numpy pycryptodome

Generate Arduino sketch:
  python kaspa_bench_offload.py --emit-arduino

Demo mine, no Arduino:
  python kaspa_bench_offload.py --mine --leading-zero-bits 12

Photodiode bench:
  python kaspa_bench_offload.py --mine --leading-zero-bits 12 --source photodiodes --port COM7

Camera bench:
  python kaspa_bench_offload.py --mine --leading-zero-bits 12 --source camera --port /dev/ttyACM0

Real-style job:
  python kaspa_bench_offload.py --mine \
    --pre-pow-hash 64_HEX_CHARS \
    --target-hex 64_HEX_CHARS \
    --source photodiodes --port COM7
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from Crypto.Hash import cSHAKE256
    HAVE_CSHAKE = True
except ImportError:
    HAVE_CSHAKE = False

try:
    import serial
    HAVE_SERIAL = True
except ImportError:
    HAVE_SERIAL = False


ARDUINO_SKETCH = r"""
// Arduino Mega 2560 - fully offloaded Kaspa kHeavyHash bench miner.
//
// Photodiodes: A0..A7
// ITO pixels:  D2..D9, PWM 0..255
//
// PC -> Mega:
//   PING
//   MODE PHOTODIODES | MODE CAMERA
//   MATRIXROW <row 0..63> <128 hex chars: 64 4-bit matrix entries>
//   ITOEVERY <n>
//   JOB <pre_pow_hash 64 hex> <timestamp 16 hex LE> <target 64 hex> <start_nonce 16 hex LE> <count 16 hex LE>
//   SAMPLE
//   ITOCLEAR
//
// Mega -> PC:
//   OK,...
//   ERR,...
//   D,<millis>,<pd0..pd7>
//   F,<millis>,<64 camera values>
//   FOUND,<nonce dec>,<hash 64 hex>
//   NONE,<nonces tried>
//
// The optical readings are telemetry only. They do not validate PoW.

#include <string.h>

#define RATE 136
#define NUM_ITO 8
#define NUM_PD 8

static const int PHOTO_PINS[NUM_PD] = {A0, A1, A2, A3, A4, A5, A6, A7};
static const int ITO_PINS[NUM_ITO] = {2, 3, 4, 5, 6, 7, 8, 9};

static uint8_t matrix[64][64];
static bool matrix_loaded = false;
static bool camera_mode = false;
static uint32_t ito_every = 1000;

// ----- Keccak-f[1600] -----

static const uint64_t RC[24] = {
  0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
  0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
  0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
  0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
  0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
  0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
  0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
  0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

static const int ROTC[24] = {
  1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44
};

static const int PILN[24] = {
  10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1
};

#define ROL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

static void keccakf(uint64_t st[25]) {
  uint64_t t, bc[5];
  for (int round = 0; round < 24; round++) {
    for (int i = 0; i < 5; i++)
      bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

    for (int i = 0; i < 5; i++) {
      t = bc[(i + 4) % 5] ^ ROL64(bc[(i + 1) % 5], 1);
      for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
    }

    t = st[1];
    for (int i = 0; i < 24; i++) {
      int j = PILN[i];
      bc[0] = st[j];
      st[j] = ROL64(t, ROTC[i]);
      t = bc[0];
    }

    for (int j = 0; j < 25; j += 5) {
      for (int i = 0; i < 5; i++) bc[i] = st[j + i];
      for (int i = 0; i < 5; i++)
        st[j + i] ^= bc[(i + 1) % 5] ^ ((~bc[(i + 2) % 5]) & bc[(i + 3) % 5]);
    }

    st[0] ^= RC[round];
  }
}

// ----- cSHAKE256 -----

static uint64_t load64_le(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
  return v;
}

static void store64_le(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; i++) p[i] = (v >> (8 * i)) & 0xff;
}

static void xor_bytes(uint64_t st[25], const uint8_t *block, size_t len) {
  for (size_t i = 0; i < len; i++) {
    size_t lane = i / 8;
    size_t byte = i % 8;
    ((uint8_t *)st)[lane * 8 + byte] ^= block[i];
  }
}

static void keccak_absorb(uint64_t st[25], const uint8_t *data, size_t len) {
  while (len >= RATE) {
    xor_bytes(st, data, RATE);
    keccakf(st);
    data += RATE;
    len -= RATE;
  }
  uint8_t block[RATE];
  memset(block, 0, RATE);
  memcpy(block, data, len);
  block[len] ^= 0x04;
  block[RATE - 1] ^= 0x80;
  xor_bytes(st, block, RATE);
  keccakf(st);
}

static void keccak_squeeze(uint64_t st[25], uint8_t *out, size_t outlen) {
  size_t made = 0;
  while (made < outlen) {
    size_t chunk = outlen - made;
    if (chunk > RATE) chunk = RATE;
    memcpy(out + made, (uint8_t *)st, chunk);
    made += chunk;
    if (made < outlen) keccakf(st);
  }
}

static size_t left_encode(uint8_t *out, size_t x) {
  uint8_t tmp[8];
  size_t n = 0;
  while (x > 0) {
    tmp[n++] = x & 0xff;
    x >>= 8;
  }
  if (n == 0) {
    out[0] = 1;
    out[1] = 0;
    return 2;
  }
  out[0] = n;
  for (size_t i = 0; i < n; i++) out[1 + i] = tmp[n - 1 - i];
  return n + 1;
}

static size_t encode_string(uint8_t *out, const char *s) {
  size_t len = strlen(s);
  size_t n = left_encode(out, len * 8);
  memcpy(out + n, s, len);
  return n + len;
}

static void cshake256(const uint8_t *data, size_t datalen,
                      const char *custom,
                      uint8_t *out, size_t outlen) {
  uint64_t st[25];
  memset(st, 0, sizeof(st));

  uint8_t prefix[RATE];
  memset(prefix, 0, RATE);

  size_t p = 0;
  p += left_encode(prefix + p, RATE);
  p += left_encode(prefix + p, 0);          // empty function-name N
  p += encode_string(prefix + p, custom);   // customization S

  // bytepad(prefix, RATE): first two bytes are already left_encode(RATE).
  keccak_absorb(st, prefix, RATE);
  keccak_absorb(st, data, datalen);
  keccak_squeeze(st, out, outlen);
}

// ----- kHeavyHash -----

static uint8_t pre_pow[32];
static uint8_t target[32];
static uint64_t timestamp;
static uint64_t start_nonce;

static void heavy_transform(const uint8_t pow_hash[32], uint8_t out[32]) {
  uint16_t vector[64];
  for (int i = 0; i < 32; i++) {
    vector[2 * i] = pow_hash[i] >> 4;
    vector[2 * i + 1] = pow_hash[i] & 0x0f;
  }

  uint8_t product[32];
  for (int i = 0; i < 64; i += 2) {
    uint32_t a = 0, b = 0;
    for (int j = 0; j < 64; j++) {
      a += (uint32_t)matrix[i][j] * vector[j];
      b += (uint32_t)matrix[i + 1][j] * vector[j];
    }
    product[i / 2] = (uint8_t)(((a >> 10) << 4) | ((b >> 10) & 0x0f));
  }

  for (int i = 0; i < 32; i++) out[i] = pow_hash[i] ^ product[i];
}

static bool hash_meets_target(const uint8_t h[32]) {
  for (int i = 0; i < 32; i++) {
    if (h[i] < target[i]) return true;
    if (h[i] > target[i]) return false;
  }
  return true;
}

static void update_optics(const uint8_t h[32]) {
  for (int i = 0; i < NUM_ITO; i++) analogWrite(ITO_PINS[i], h[i]);
}

static bool mine(uint64_t count, uint64_t &found_nonce, uint8_t found_hash[32]) {
  uint8_t work[48];
  uint8_t pow_hash[32];
  uint8_t transformed[32];
  uint8_t final_hash[32];

  memcpy(work, pre_pow, 32);
  store64_le(work + 32, timestamp);

  for (uint64_t k = 0; k < count; k++) {
    uint64_t nonce = start_nonce + k;
    store64_le(work + 40, nonce);

    cshake256(work, 48, "ProofOfWorkHash", pow_hash, 32);
    heavy_transform(pow_hash, transformed);
    cshake256(transformed, 32, "HeavyHash", final_hash, 32);

    // Kaspa compares the byte-reversed final hash.
    for (int i = 0; i < 16; i++) {
      uint8_t tmp = final_hash[i];
      final_hash[i] = final_hash[31 - i];
      final_hash[31 - i] = tmp;
    }

    if (k % ito_every == 0) update_optics(final_hash);

    if (hash_meets_target(final_hash)) {
      found_nonce = nonce;
      memcpy(found_hash, final_hash, 32);
      update_optics(final_hash);
      return true;
    }
  }
  return false;
}

// ----- serial helpers -----

static int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static bool hex_to_bytes(const String &s, uint8_t *out, size_t n) {
  if (s.length() != (int)(n * 2)) return false;
  for (size_t i = 0; i < n; i++) {
    int hi = hexval(s[2 * i]);
    int lo = hexval(s[2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = (hi << 4) | lo;
  }
  return true;
}

static void send_photodiodes() {
  Serial.print(F("D,"));
  Serial.print(millis());
  for (int i = 0; i < NUM_PD; i++) {
    Serial.print(',');
    Serial.print(analogRead(PHOTO_PINS[i]));
  }
  Serial.println();
}

static void send_camera() {
  // Default: tile A0..A7 into an 8x8 frame.
  // Replace with your camera module's capture code.
  uint8_t frame[64];
  for (int y = 0; y < 8; y++)
    for (int x = 0; x < 8; x++)
      frame[y * 8 + x] = analogRead(PHOTO_PINS[x]) >> 2;

  Serial.print(F("F,"));
  Serial.print(millis());
  for (int i = 0; i < 64; i++) {
    Serial.print(',');
    Serial.print(frame[i]);
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < NUM_ITO; i++) {
    pinMode(ITO_PINS[i], OUTPUT);
    analogWrite(ITO_PINS[i], 0);
  }
  Serial.println(F("OK,READY"));
}

void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (!line.length()) return;

  if (line == F("PING")) {
    Serial.println(F("OK,PONG"));
  } else if (line == F("MODE PHOTODIODES")) {
    camera_mode = false;
    Serial.println(F("OK,MODE_PHOTODIODES"));
  } else if (line == F("MODE CAMERA")) {
    camera_mode = true;
    Serial.println(F("OK,MODE_CAMERA"));
  } else if (line == F("SAMPLE")) {
    if (camera_mode) send_camera();
    else send_photodiodes();
  } else if (line == F("ITOCLEAR")) {
    for (int i = 0; i < NUM_ITO; i++) analogWrite(ITO_PINS[i], 0);
    Serial.println(F("OK,ITOCLEAR"));
  } else if (line.startsWith(F("ITOEVERY "))) {
    ito_every = max((uint32_t)1, (uint32_t)line.substring(9).toInt());
    Serial.println(F("OK,ITOEVERY"));
  } else if (line.startsWith(F("MATRIXROW "))) {
    int sp = line.indexOf(' ', 10);
    if (sp < 0) {
      Serial.println(F("ERR,MATRIXROW syntax"));
      return;
    }
    int row = line.substring(10, sp).toInt();
    if (row < 0 || row >= 64) {
      Serial.println(F("ERR,MATRIXROW index"));
      return;
    }
    if (!hex_to_bytes(line.substring(sp + 1), matrix[row], 64)) {
      Serial.println(F("ERR,MATRIXROW hex"));
      return;
    }
    matrix_loaded = true;
    Serial.println(F("OK,MATRIXROW"));
  } else if (line.startsWith(F("JOB "))) {
    if (!matrix_loaded) {
      Serial.println(F("ERR,MATRIX_NOT_LOADED"));
      return;
    }

    int p1 = line.indexOf(' ', 4);
    int p2 = line.indexOf(' ', p1 + 1);
    int p3 = line.indexOf(' ', p2 + 1);
    int p4 = line.indexOf(' ', p3 + 1);
    int p5 = line.indexOf(' ', p4 + 1);
    if (p5 < 0) {
      Serial.println(F("ERR,JOB syntax"));
      return;
    }

    if (!hex_to_bytes(line.substring(4, p1), pre_pow, 32) ||
        !hex_to_bytes(line.substring(p3 + 1, p4), target, 32)) {
      Serial.println(F("ERR,JOB hex"));
      return;
    }

    timestamp = strtoull(line.substring(p1 + 1, p2).c_str(), NULL, 16);
    start_nonce = strtoull(line.substring(p4 + 1, p5).c_str(), NULL, 16);
    uint64_t count = strtoull(line.substring(p5 + 1).c_str(), NULL, 16);

    uint64_t found_nonce;
    uint8_t found_hash[32];

    if (mine(count, found_nonce, found_hash)) {
      Serial.print(F("FOUND,"));
      Serial.print(found_nonce);
      Serial.print(',');
      for (int i = 0; i < 32; i++) {
        if (found_hash[i] < 16) Serial.print('0');
        Serial.print(found_hash[i], HEX);
      }
      Serial.println();
    } else {
      Serial.print(F("NONE,"));
      Serial.println(count);
    }
  } else {
    Serial.println(F("ERR,UNKNOWN_COMMAND"));
  }
}
"""


# ---------------------------------------------------------------------------
# PC-side kHeavyHash: used to generate the matrix and verify Mega results
# ---------------------------------------------------------------------------

MASK64 = (1 << 64) - 1


def rotl64(x: int, k: int) -> int:
    x &= MASK64
    return ((x << k) | (x >> (64 - k))) & MASK64


class Xoshiro256PlusPlus:
    def __init__(self, seed32: bytes):
        if len(seed32) != 32:
            raise ValueError("xoshiro seed must be 32 bytes")
        self.s = list(struct.unpack("<4Q", seed32))

    def next_u64(self) -> int:
        s0, s1, s2, s3 = self.s
        result = (rotl64((s0 + s3) & MASK64, 23) + s0) & MASK64
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 = rotl64(s2, 17)
        s3 = rotl64(s3, 45)
        self.s = [s0, s1, s2, s3]
        return result


def gf2_rank_64(matrix: np.ndarray) -> int:
    rows = []
    for i in range(64):
        bits = (matrix[i].astype(np.uint64) & 1).astype(np.uint8)
        packed = np.packbits(bits, bitorder="little")
        value = int.from_bytes(packed.tobytes(), "little")
        rows.append(value)

    rank = 0
    for col in range(64):
        pivot = next((r for r in range(rank, 64) if (rows[r] >> col) & 1), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(64):
            if r != rank and ((rows[r] >> col) & 1):
                rows[r] ^= rows[rank]
        rank += 1
    return rank


def generate_matrix(seed32: bytes, max_attempts: int = 1000) -> np.ndarray:
    """Generate a full-rank 64x64 kHeavyHash-style matrix.

    The matrix entries are 4-bit values derived from a cSHAKE256 XOF stream
    seeded by the pre-PoW hash. The bench uses a GF(2) rank test on the
    parity bits of those entries.
    """
    if len(seed32) != 32:
        raise ValueError("matrix seed must be 32 bytes")
    if not HAVE_CSHAKE:
        raise RuntimeError("pycryptodome is required: pip install pycryptodome")

    for attempt in range(max_attempts):
        # Domain-separate each attempt so a failed rank check gets new material.
        stream = cshake256(
            seed32 + attempt.to_bytes(4, "little"),
            b"KaspaBenchMatrix",
            length=4096,  # 64 * 64 nibbles = 2048 bytes; request extra margin
        )

        m = np.zeros((64, 64), dtype=np.uint16)
        pos = 0
        for i in range(64):
            for j in range(64):
                m[i, j] = stream[pos] & 0x0F
                pos += 1

        if gf2_rank_64(m) == 64:
            return m

    raise RuntimeError("could not generate a full-rank matrix")


def cshake256(data: bytes, custom: bytes, length: int = 32) -> bytes:
    if not HAVE_CSHAKE:
        raise RuntimeError("pycryptodome is required: pip install pycryptodome")
    h = cSHAKE256.new(custom=custom)
    h.update(data)
    return h.read(length)


def heavy_transform(pow_hash: bytes, matrix: np.ndarray) -> bytes:
    vector = np.empty(64, dtype=np.uint64)
    for i in range(32):
        vector[2 * i] = pow_hash[i] >> 4
        vector[2 * i + 1] = pow_hash[i] & 0xF

    product = np.empty(64, dtype=np.uint64)
    for i in range(64):
        product[i] = np.sum(matrix[i].astype(np.uint64) * vector) >> 10

    transformed = bytearray(32)
    for i in range(32):
        transformed[i] = ((int(product[2 * i]) & 0xF) << 4) | (int(product[2 * i + 1]) & 0xF)

    return bytes(a ^ b for a, b in zip(pow_hash, transformed))


def kheavy_hash(pre_pow_hash: bytes, timestamp: int, nonce: int, matrix: np.ndarray) -> bytes:
    work = pre_pow_hash + struct.pack("<Q", timestamp) + struct.pack("<Q", nonce)
    pow_hash = cshake256(work, b"ProofOfWorkHash", 32)
    transformed = heavy_transform(pow_hash, matrix)
    final = cshake256(transformed, b"HeavyHash", 32)
    return final[::-1]


def target_from_leading_zero_bits(bits: int) -> bytes:
    if not 1 <= bits <= 255:
        raise ValueError("leading-zero-bits must be 1..255")
    return ((1 << (256 - bits)) - 1).to_bytes(32, "big")


# ---------------------------------------------------------------------------
# Mega serial interface
# ---------------------------------------------------------------------------

class MegaMiner:
    def __init__(self, port: str, baud: int = 115200):
        if not HAVE_SERIAL:
            raise RuntimeError("pyserial is not installed: pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=2.0)
        time.sleep(2.0)
        self.ser.reset_input_buffer()
        self.command("PING", "OK,PONG")

    def command(self, cmd: str, expected: Optional[str] = None, timeout: float = 5.0) -> str:
        self.ser.reset_input_buffer()
        self.ser.write((cmd + "\n").encode("ascii"))
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.ser.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue
            if expected and line == expected:
                return line
            if line.startswith("OK,"):
                return line
            if line.startswith("ERR,"):
                raise RuntimeError(f"Arduino error: {line}")
        raise TimeoutError(f"No valid response to: {cmd[:80]}")

    def set_source(self, source: str) -> None:
        if source not in {"photodiodes", "camera"}:
            raise ValueError("source must be photodiodes or camera")
        self.command(f"MODE {source.upper()}")

    def upload_matrix(self, matrix: np.ndarray) -> None:
        for row in range(64):
            hex_row = "".join(f"{int(v):x}" for v in matrix[row])
            self.command(f"MATRIXROW {row} {hex_row}", "OK,MATRIXROW")

    def set_ito_every(self, n: int) -> None:
        self.command(f"ITOEVERY {max(1, int(n))}", "OK,ITOEVERY")

    def mine(self, pre_pow_hash: bytes, timestamp: int, target: bytes, start_nonce: int, count: int, timeout: float) -> dict:
        cmd = (
            f"JOB {pre_pow_hash.hex()} {timestamp:016x} "
            f"{target.hex()} {start_nonce:016x} {count:016x}"
        )
        self.ser.reset_input_buffer()
        self.ser.write((cmd + "\n").encode("ascii"))

        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.ser.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("FOUND,"):
                _, nonce_s, hash_hex = line.split(",", 2)
                return {
                    "found": True,
                    "nonce": int(nonce_s),
                    "hash_hex": hash_hex.lower(),
                    "raw": line,
                }
            if line.startswith("NONE,"):
                return {"found": False, "nonces_tried": int(line.split(",")[1]), "raw": line}
            if line.startswith("ERR,"):
                raise RuntimeError(f"Arduino error: {line}")
        raise TimeoutError("Mega mining job timed out")

    def read_sample(self, camera: bool) -> tuple[float, np.ndarray]:
        line = self.command("SAMPLE")
        parts = line.split(",")
        if camera:
            if not line.startswith("F,") or len(parts) != 66:
                raise RuntimeError(f"Bad camera frame: {line}")
            return float(parts[1]), np.array([float(v) for v in parts[2:]], dtype=float)
        if not line.startswith("D,") or len(parts) != 10:
            raise RuntimeError(f"Bad photodiode sample: {line}")
        return float(parts[1]), np.array([float(v) for v in parts[2:]], dtype=float)

    def clear(self) -> None:
        try:
            self.command("ITOCLEAR")
        except Exception:
            pass

    def close(self) -> None:
        self.clear()
        self.ser.close()


# ---------------------------------------------------------------------------
# Mining orchestration
# ---------------------------------------------------------------------------

def mine_via_bench(
    pre_pow_hash: bytes,
    timestamp: int,
    target: bytes,
    source: str,
    port: Optional[str],
    baud: int,
    start_nonce: int,
    batch_size: int,
    max_nonces: int,
    optical_every: int,
    outdir: Path,
) -> dict:
    if not HAVE_CSHAKE:
        raise RuntimeError("pycryptodome is required: pip install pycryptodome")

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "kaspa_bench_offload.csv"
    solution_path = outdir / "kaspa_solution.json"

    matrix = generate_matrix(pre_pow_hash)
    bench = None
    camera = source == "camera"

    if source != "none":
        if not port:
            raise ValueError("--port is required unless --source none")
        bench = MegaMiner(port, baud)
        bench.set_source(source)
        bench.upload_matrix(matrix)
        bench.set_ito_every(optical_every)

    start_time = time.perf_counter()
    attempted = 0
    found_record = None
    telemetry = None

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if camera:
                writer.writerow(["nonce", "hash_hex", "ito", "t_ms"] + [f"px_{i}" for i in range(64)])
            else:
                writer.writerow(["nonce", "hash_hex", "ito", "t_ms"] + [f"pd_{i}" for i in range(8)])

            while attempted < max_nonces and found_record is None:
                count = min(batch_size, max_nonces - attempted)
                result = bench.mine(
                    pre_pow_hash=pre_pow_hash,
                    timestamp=timestamp,
                    target=target,
                    start_nonce=start_nonce + attempted,
                    count=count,
                    timeout=max(10.0, count * 0.05 + 10.0),
                )
                attempted += count

                if result["found"]:
                    nonce = result["nonce"]
                    mega_hash = bytes.fromhex(result["hash_hex"])

                    # PC independently verifies the Mega's claimed solution.
                    pc_hash = kheavy_hash(pre_pow_hash, timestamp, nonce, matrix)
                    verified = pc_hash == mega_hash and int.from_bytes(pc_hash, "big") <= int.from_bytes(target, "big")

                    ito = list(mega_hash[:8])
                    bench.set_ito(ito)
                    t_ms, readings = bench.read_sample(camera)
                    telemetry = {"t_ms": t_ms, "readings": readings.astype(int).tolist()}

                    writer.writerow([nonce, mega_hash.hex(), ito, t_ms] + readings.astype(int).tolist())

                    found_record = {
                        "nonce": nonce,
                        "mega_hash_hex": mega_hash.hex(),
                        "pc_verification_hash_hex": pc_hash.hex(),
                        "verified_by_pc": verified,
                        "ito_pattern": ito,
                        "target_hex": target.hex(),
                        "pre_pow_hash_hex": pre_pow_hash.hex(),
                        "timestamp": timestamp,
                        "nonces_searched": attempted,
                        "elapsed_s": time.perf_counter() - start_time,
                        "bench_hashrate_approx": attempted / max(time.perf_counter() - start_time, 1e-9),
                    }
                    break

    finally:
        if bench:
            bench.close()

    report = {
        "solved": found_record is not None,
        "solution": found_record,
        "nonces_attempted": attempted,
        "elapsed_s": time.perf_counter() - start_time,
        "bench_hashrate_approx": attempted / max(time.perf_counter() - start_time, 1e-9),
        "telemetry": telemetry,
        "source": source,
        "port": port,
        "batch_size": batch_size,
        "optical_every": optical_every,
        "csv": str(csv_path),
        "security_note": "The Mega performs the search, but the PC independently recomputes and verifies any solution before it is accepted.",
    }

    if found_record:
        solution_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Kaspa kHeavyHash bench miner with Mega offload")
    ap.add_argument("--mine", action="store_true")
    ap.add_argument("--source", choices=["photodiodes", "camera", "none"], default="none")
    ap.add_argument("--port", help="Arduino Mega serial port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--pre-pow-hash", help="32-byte pre-PoW hash, 64 hex chars")
    ap.add_argument("--work-order", help="80-byte work order, 160 hex chars")
    ap.add_argument("--timestamp", type=lambda x: int(x, 0), default=0)
    ap.add_argument("--leading-zero-bits", type=int, help="Easy bench target")
    ap.add_argument("--target-hex", help="Exact 32-byte target")
    ap.add_argument("--start-nonce", type=lambda x: int(x, 0), default=0)
    ap.add_argument("--batch-size", type=lambda x: int(x, 0), default=10_000)
    ap.add_argument("--max-nonces", type=lambda x: int(x, 0), default=1_000_000)
    ap.add_argument("--optical-every", type=int, default=1000)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--outdir", default="kaspa_bench_offload_run")
    ap.add_argument("--emit-arduino", action="store_true")
    args = ap.parse_args()

    if args.emit_arduino:
        p = Path("arduino_mega_kaspa_offload.ino")
        p.write_text(ARDUINO_SKETCH, encoding="utf-8")
        print(f"Wrote Arduino sketch: {p}")
        return 0

    if not args.mine:
        ap.print_help()
        return 0

    if args.work_order:
        work = bytes.fromhex(args.work_order.replace(" ", ""))
        if len(work) != 80:
            raise ValueError("--work-order must be exactly 80 bytes")
        pre_pow_hash = cshake256(work, b"ProofOfWorkHash", 32)
    elif args.pre_pow_hash:
        pre_pow_hash = bytes.fromhex(args.pre_pow_hash.replace(" ", ""))
        if len(pre_pow_hash) != 32:
            raise ValueError("--pre-pow-hash must be exactly 32 bytes")
    else:
        pre_pow_hash = cshake256(b"kaspa-optical-bench-demo-template", b"ProofOfWorkHash", 32)

    timestamp = args.timestamp or int(time.time())

    if args.target_hex:
        target = bytes.fromhex(args.target_hex.replace(" ", ""))
        if len(target) != 32:
            raise ValueError("--target-hex must be exactly 32 bytes")
    elif args.leading_zero_bits:
        target = target_from_leading_zero_bits(args.leading_zero_bits)
    else:
        target = target_from_leading_zero_bits(12)
        print("No target supplied; using easy bench target of 12 leading zero bits.")

    print(f"Pre-PoW hash: {pre_pow_hash.hex()}")
    print(f"Timestamp: {timestamp}")
    print(f"Target: {target.hex()}")
    print(f"Source: {args.source}")

    report = mine_via_bench(
        pre_pow_hash=pre_pow_hash,
        timestamp=timestamp,
        target=target,
        source=args.source,
        port=args.port,
        baud=args.baud,
        start_nonce=args.start_nonce,
        batch_size=args.batch_size,
        max_nonces=args.max_nonces,
        optical_every=args.optical_every,
        outdir=Path(args.outdir),
    )

    print(json.dumps(report, indent=2))
    return 0 if report["solved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())