#!/usr/bin/env python3
"""
RP-GHOST timestamp pipeline with OpenCV webcam whitening.

Implements the data path from the paper "Decoding the Timestamp from the
Optoelectronic Viewport":

  schedule T = t0 + 1 s  ->  c = floor(T/1e6) mod 60000
  -> whiten c' = c XOR h   (h = first 2 bytes of SHA-256 of grayscale webcam frame)
  -> frame [c' | delta | CRC-8]  (40 bits)
  -> viewport: 2000 windows x 40 Poisson slots with +-1/4 tilt and common mode
  -> window mean, sign decision, CRC check
  -> DE-WHITEN (the step missing in the reference program), unwrap to the
     representative nearest the receiver clock, print Time Difference.

Usage:
  pip install numpy opencv-python
  python rp_ghost_opencv.py                      # webcam 0, 5 cycles
  python rp_ghost_opencv.py --camera 1 --cycles 10
  python rp_ghost_opencv.py --no-camera          # h = 0 (no webcam)
  python rp_ghost_opencv.py --show               # preview window + slot display
  python rp_ghost_opencv.py --no-dewhiten        # reproduce the bug from Sec. 7
  python rp_ghost_opencv.py --selftest           # CRC check value, round trip
"""
import argparse
import hashlib
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:  # allow --no-camera / --selftest without OpenCV
    cv2 = None

# --------------------------------------------------------------------------
# Constants (Table 1 of the paper)
# --------------------------------------------------------------------------
LAMBDA_BASE = 1e4          # mean photon count per slot per window
SIGMA_C = 0.005            # relative common-mode jitter
N_SLOTS = 40               # bits per frame
N_WINDOWS = 2000           # windows averaged per decode
M_MS = 60_000              # residue modulus (ms)
CRC_POLY = 0x07            # x^8 + x^2 + x + 1
LEAD_NS = 1_000_000_000    # scheduled one second ahead


def tilt_amplitude(lam=LAMBDA_BASE, sigma_c=SIGMA_C):
    """s = max(0.02*sigma_z, 0.25), sigma_z^2 = 1 + lam*sigma_c^2."""
    sigma_z = np.sqrt(1.0 + lam * sigma_c ** 2)
    return max(0.02 * sigma_z, 0.25)


# --------------------------------------------------------------------------
# CRC-8
# --------------------------------------------------------------------------
def crc8(data: bytes, poly: int = CRC_POLY, init: int = 0) -> int:
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


# --------------------------------------------------------------------------
# Webcam hash via OpenCV
# --------------------------------------------------------------------------
class WebcamHasher:
    """Grabs frames with OpenCV and produces the 16-bit whitening word h."""

    def __init__(self, camera_index=0, enabled=True, show=False):
        self.cap = None
        self.show = show
        self.frames = 0
        self.latest_hash = 0
        self.last_gray = None
        if enabled:
            if cv2 is None:
                print("[warn] OpenCV not installed; whitening word h = 0", file=sys.stderr)
            else:
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    self.cap = cap
                else:
                    cap.release()
                    print(f"[warn] cannot open camera {camera_index}; h = 0", file=sys.stderr)

    @staticmethod
    def hash_frame(gray: np.ndarray) -> int:
        digest = hashlib.sha256(np.ascontiguousarray(gray).tobytes()).digest()
        return (digest[0] << 8) | digest[1]

    def grab(self) -> int:
        """Capture one frame, return h (0 if no camera)."""
        if self.cap is None:
            return self.latest_hash
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return self.latest_hash
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.last_gray = gray
        self.frames += 1
        self.latest_hash = self.hash_frame(gray)
        if self.show:
            cv2.imshow("RP-GHOST webcam (gray)", gray)
            cv2.waitKey(1)
        return self.latest_hash

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if cv2 is not None and self.show:
            cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------
def int_to_bits(value: int, width: int):
    return [(value >> (width - 1 - k)) & 1 for k in range(width)]


def bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def encode_payload_bits(code: int, delta_cents: int, h: int):
    """Whiten the residue, append signed 16-bit delta and CRC-8 -> 40 bits."""
    c_prime = (code ^ h) & 0xFFFF
    d = delta_cents & 0xFFFF  # two's complement
    body = bytes([c_prime >> 8, c_prime & 0xFF, d >> 8, d & 0xFF])
    crc = crc8(body)
    return int_to_bits(c_prime, 16) + int_to_bits(d, 16) + int_to_bits(crc, 8)


def decode_payload_bits(bits):
    """Returns (c_prime, delta_cents, crc_valid). c_prime is still whitened."""
    c_prime = bits_to_int(bits[0:16])
    d = bits_to_int(bits[16:32])
    rx_crc = bits_to_int(bits[32:40])
    body = bytes([c_prime >> 8, c_prime & 0xFF, d >> 8, d & 0xFF])
    delta = d - 0x10000 if d & 0x8000 else d
    return c_prime, delta, crc8(body) == rx_crc


# --------------------------------------------------------------------------
# Optoelectronic viewport (simulated)
# --------------------------------------------------------------------------
class OptoelectronicViewport:
    def __init__(self, lam=LAMBDA_BASE, sigma_c=SIGMA_C, n_slots=N_SLOTS, seed=None):
        self.lam = lam
        self.sigma_c = sigma_c
        self.n = n_slots
        self.s = tilt_amplitude(lam, sigma_c)
        self.rng = np.random.default_rng(seed)

    def window_block(self, bits, n_windows=N_WINDOWS):
        """Return Z[w, j] = (N - lam)/sqrt(lam) + s(2b_j - 1)."""
        t = self.s * (2 * np.asarray(bits, dtype=float) - 1)
        eps = self.rng.normal(0.0, self.sigma_c, size=(n_windows, 1))  # common mode
        rate = np.maximum(self.lam * (1.0 + eps), 1.0)                 # clip at 1
        counts = self.rng.poisson(rate, size=(n_windows, self.n))
        x = (counts - self.lam) / np.sqrt(self.lam)
        return x + t

    @staticmethod
    def decide(z_block):
        m = z_block.mean(axis=0)
        return (m > 0).astype(int).tolist(), m


# --------------------------------------------------------------------------
# Modular reconstruction (Theorem 3), integer arithmetic in ns
# --------------------------------------------------------------------------
def unwrap_to_clock(code_ms: int, receiver_ns: int, modulus_ms: int = M_MS) -> int:
    """Representative of code_ms + modulus*Z (in ns) nearest to receiver_ns."""
    mod_ns = modulus_ms * 1_000_000
    c_ns = code_ms * 1_000_000
    k = (receiver_ns - c_ns + mod_ns // 2) // mod_ns   # round to nearest
    return c_ns + k * mod_ns


# --------------------------------------------------------------------------
# One transmit/receive cycle
# --------------------------------------------------------------------------
def run_cycle(viewport, hasher, delta_cents=0, dewhiten=True, w=N_WINDOWS):
    t0 = time.time_ns()
    h = hasher.grab()
    scheduled_ns = t0 + LEAD_NS
    t_ms = scheduled_ns // 1_000_000
    code = t_ms % M_MS

    bits = encode_payload_bits(code, delta_cents, h)
    z = viewport.window_block(bits, w)
    rx_bits, m = viewport.decide(z)
    bit_errors = int(np.sum(np.array(bits) != np.array(rx_bits)))

    c_prime, delta_rx, crc_ok = decode_payload_bits(rx_bits)
    c_hat = (c_prime ^ h) & 0xFFFF if dewhiten else c_prime
    # Receiver clock R = t0 (as in the paper)
    t_hat_ns = unwrap_to_clock(c_hat, t0)
    diff_s = (t_hat_ns - t0) / 1e9
    return dict(h=h, code=code, bits=bits, rx_bits=rx_bits, m=m,
                bit_errors=bit_errors, crc_ok=crc_ok, delta=delta_rx,
                t_hat_ns=t_hat_ns, diff_s=diff_s)


def render_viewport(m, bits, s):
    """Small OpenCV picture of the 40 window means (cv2.imshow)."""
    img = np.full((140, 40 * 16, 3), 255, np.uint8)
    for j, mj in enumerate(m):
        x0 = j * 16
        y = int(70 - np.clip(mj / (1.5 * s), -1, 1) * 60)
        colour = (0, 160, 0) if (mj > 0) == bool(bits[j]) else (0, 0, 255)
        cv2.rectangle(img, (x0 + 2, min(y, 70)), (x0 + 13, max(y, 70)), colour, -1)
    cv2.line(img, (0, 70), (img.shape[1], 70), (0, 0, 0), 1)
    cv2.putText(img, "window means m_j (green = correct)", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return img


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def selftest():
    assert crc8(b"123456789") == 0xF4, "CRC-8 check value mismatch"
    rng = np.random.default_rng(1)
    for _ in range(1000):
        code = int(rng.integers(0, M_MS))
        h = int(rng.integers(0, 65536))
        delta = int(rng.integers(-32768, 32768))
        bits = encode_payload_bits(code, delta, h)
        c_prime, d, ok = decode_payload_bits(bits)
        assert ok and d == delta and (c_prime ^ h) == code
    # Theorem 3: reconstruction within +-30 s of the clock
    R = time.time_ns()
    for off_ms in (-29_999, -1, 0, 1_000, 29_999):
        T = R + off_ms * 1_000_000
        T_ms = T // 1_000_000
        assert unwrap_to_clock(T_ms % M_MS, R) // 1_000_000 == T_ms
    # Statistical check of Eq. (7): bit error rate at W = 200 ~ 7.8e-4
    vp = OptoelectronicViewport(seed=7)
    errs = 0
    trials = 4000
    for _ in range(trials):
        b = rng.integers(0, 2, N_SLOTS).tolist()
        rx, _ = vp.decide(vp.window_block(b, 200))
        errs += int(np.sum(np.array(b) != np.array(rx)))
    ber = errs / (trials * N_SLOTS)
    print(f"selftest OK  (BER at W=200: {ber:.2e}, theory 7.83e-04)")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--no-camera", action="store_true", help="do not open a camera (h = 0)")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--windows", type=int, default=N_WINDOWS)
    ap.add_argument("--delta", type=int, default=0, help="price delta in cents (signed 16-bit)")
    ap.add_argument("--no-dewhiten", action="store_true", help="reproduce the reference bug (Sec. 7)")
    ap.add_argument("--show", action="store_true", help="show webcam and viewport windows")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.show and cv2 is None:
        print("[warn] --show requires OpenCV", file=sys.stderr)
        args.show = False

    hasher = WebcamHasher(args.camera, enabled=not args.no_camera, show=args.show)
    viewport = OptoelectronicViewport(seed=args.seed)
    print(f"s = {viewport.s:.4f}, W = {args.windows}, de-whitening = {not args.no_dewhiten}")

    try:
        for i in range(1, args.cycles + 1):
            r = run_cycle(viewport, hasher, args.delta, not args.no_dewhiten, args.windows)
            print(f"[{i}] Frames: {hasher.frames}  h=0x{r['h']:04X}  "
                  f"bit errors: {r['bit_errors']}  CRC: {'OK' if r['crc_ok'] else 'FAIL'}  "
                  f"delta: {r['delta']:+d}c")
            if r["crc_ok"]:
                print(f"    Time Difference: {r['diff_s']:+.3f} s")
            else:
                print("    CRC mismatch, frame discarded (resend)")
            if args.show:
                cv2.imshow("RP-GHOST viewport", render_viewport(r["m"], r["bits"], viewport.s))
                cv2.waitKey(300)
    finally:
        hasher.release()


if __name__ == "__main__":
    main()
