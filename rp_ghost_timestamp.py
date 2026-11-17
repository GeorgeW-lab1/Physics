#!/usr/bin/env python3
"""
RP-GHOST timestamp viewport: standalone probabilistic encoder/decoder.

Implements the model described in "Decoding the Timestamp from the
Optoelectronic Viewport" (George Wagenknecht, 18 Nov 2026).

This is a software simulation of the paper's photon-count channel. It does
not claim to be a physical photon detector or a hardware timestamp receiver.

Install:
    pip install numpy matplotlib

Optional webcam hash:
    pip install opencv-python

Run:
    python rp_ghost_timestamp.py
    python rp_ghost_timestamp.py --windows 200 --plots
    python rp_ghost_timestamp.py --webcam 0 --benchmark --plots
"""

import argparse
import hashlib
import math
import time
from datetime import datetime, timezone

import numpy as np

# ---------------------------- Model parameters ----------------------------

LAMBDA_BASE = 10_000.0
SIGMA_C = 0.005
N_SLOTS = 40
DEFAULT_WINDOWS = 2000
TILT = 0.25
MODULUS_MS = 60_000
CRC_POLY = 0x07


# ------------------------------- CRC / bits -------------------------------

def crc8(data: bytes) -> int:
    """CRC-8, polynomial 0x07, init=0, no reflection, xorout=0."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def int_to_bits(value: int, width: int) -> np.ndarray:
    return np.array([(value >> (width - 1 - i)) & 1
                     for i in range(width)], dtype=np.uint8)


def bits_to_int(bits) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def encode_frame(timestamp_ms: int, price_delta_cents: int, frame_hash: int):
    """
    Frame layout:
        bits  0..15: timestamp residue XOR 16-bit frame hash
        bits 16..31: signed 16-bit price delta, two's complement
        bits 32..39: CRC-8 over the four preceding bytes
    """
    residue = int(timestamp_ms) % MODULUS_MS
    whitening = int(frame_hash) & 0xFFFF
    whitened = residue ^ whitening

    delta = int(price_delta_cents)
    if not -32768 <= delta <= 32767:
        raise ValueError("price_delta_cents must fit a signed 16-bit integer")
    delta_word = delta & 0xFFFF

    payload = (whitened.to_bytes(2, "big") +
               delta_word.to_bytes(2, "big"))
    check = crc8(payload)
    frame = payload + bytes([check])
    bits = np.unpackbits(np.frombuffer(frame, dtype=np.uint8))
    return bits.astype(np.uint8), residue, whitened, check


def decode_frame(bits):
    """Decode the 40-bit frame and verify its CRC."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if len(bits) != N_SLOTS:
        raise ValueError(f"Expected {N_SLOTS} bits, received {len(bits)}")

    raw = np.packbits(bits).tobytes()
    payload, received_crc = raw[:4], raw[4]
    calculated_crc = crc8(payload)
    crc_valid = calculated_crc == received_crc

    whitened = int.from_bytes(payload[:2], "big")
    delta_word = int.from_bytes(payload[2:4], "big")
    delta = delta_word - 65536 if delta_word & 0x8000 else delta_word

    return whitened, delta, crc_valid, received_crc, calculated_crc


# --------------------------- Webcam frame hash ----------------------------

def get_frame_hash(camera_index=None):
    """
    Return a 16-bit SHA-256-derived hash.

    With --webcam N, hash the current grayscale webcam frame. If OpenCV or
    the camera is unavailable, use a timestamp-seeded synthetic frame.
    The hash must be supplied to the receiver as side information, as
    explained in the paper.
    """
    image_bytes = None

    if camera_index is not None:
        try:
            import cv2
            camera = cv2.VideoCapture(camera_index)
            ok, frame = camera.read()
            camera.release()
            if ok and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image_bytes = gray.tobytes()
                print(f"[CAMERA] Captured grayscale frame "
                      f"{gray.shape[1]}x{gray.shape[0]}")
            else:
                print("[CAMERA] Capture failed; using synthetic frame hash.")
        except ImportError:
            print("[CAMERA] OpenCV unavailable; using synthetic frame hash.")

    if image_bytes is None:
        seed_material = f"{time.time_ns()}:{np.random.bytes(32).hex()}".encode()
        image_bytes = hashlib.sha256(seed_material).digest()

    digest = hashlib.sha256(image_bytes).digest()
    return int.from_bytes(digest[:2], "big")


# ------------------------- Optoelectronic channel -------------------------

def simulate_viewport(bits, windows=DEFAULT_WINDOWS, seed=None,
                      lambda_base=LAMBDA_BASE, sigma_c=SIGMA_C,
                      tilt=TILT, return_details=False):
    """
    Simulate W x 40 Poisson counts with a common Gaussian intensity
    fluctuation per window.

    For window w:
        epsilon_w ~ Normal(0, sigma_c^2)
        N_wj | epsilon_w ~ Poisson(lambda * (1 + epsilon_w))
        Z_wj = (N_wj - lambda) / sqrt(lambda) + tilt * (2*b_j - 1)

    Returns the 40 averaged slot measurements.
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if len(bits) != N_SLOTS:
        raise ValueError("Viewport requires exactly 40 input bits")
    if windows < 1:
        raise ValueError("windows must be at least 1")

    rng = np.random.default_rng(seed)
    common_mode = rng.normal(0.0, sigma_c, size=windows)
    rates = lambda_base * (1.0 + common_mode)
    rates = np.maximum(rates, 1.0)

    # Each row has one shared rate, so the 40 slots in that window
    # experience the same common-mode intensity fluctuation.
    counts = rng.poisson(rates[:, None], size=(windows, N_SLOTS))
    normalized = (counts - lambda_base) / math.sqrt(lambda_base)

    signal = tilt * (2.0 * bits.astype(float) - 1.0)
    observations = normalized + signal[None, :]
    means = observations.mean(axis=0)

    if return_details:
        return means, counts, common_mode
    return means


def decode_viewport(means):
    """Paper's componentwise threshold detector: b_hat[j] = 1[m[j] > 0]."""
    means = np.asarray(means, dtype=float).reshape(-1)
    if len(means) != N_SLOTS:
        raise ValueError("Expected 40 viewport means")
    return (means > 0.0).astype(np.uint8)


# ------------------------- Timestamp reconstruction ----------------------

def unwrap_minute_residue(residue_ms: int, receiver_ms: int) -> int:
    """
    Choose the representative residue + 60000*k nearest receiver_ms.
    This is unique when the true timestamp is less than 30 seconds away.
    """
    residue_ms = int(residue_ms) % MODULUS_MS
    receiver_ms = int(receiver_ms)
    k = math.floor((receiver_ms - residue_ms) / MODULUS_MS + 0.5)
    return residue_ms + MODULUS_MS * k


def decode_timestamp(bits, frame_hash, receiver_time_ns=None):
    """Decode, CRC-check, de-whiten, and unwrap the timestamp."""
    if receiver_time_ns is None:
        receiver_time_ns = time.time_ns()

    whitened, delta, crc_valid, received_crc, calculated_crc = decode_frame(bits)
    if not crc_valid:
        return {
            "valid": False,
            "reason": "CRC mismatch",
            "whitened_residue": whitened,
            "delta_cents": delta,
            "received_crc": received_crc,
            "calculated_crc": calculated_crc,
        }

    # Essential receiver operation omitted in the implementation discussed
    # in Section 7 of the paper.
    residue = whitened ^ (int(frame_hash) & 0xFFFF)
    if residue >= MODULUS_MS:
        return {
            "valid": False,
            "reason": "De-whitened residue is outside [0, 60000)",
            "whitened_residue": whitened,
            "residue_ms": residue,
            "delta_cents": delta,
            "received_crc": received_crc,
            "calculated_crc": calculated_crc,
        }

    receiver_ms = receiver_time_ns // 1_000_000
    decoded_ms = unwrap_minute_residue(residue, receiver_ms)
    decoded_ns = decoded_ms * 1_000_000
    difference_seconds = (decoded_ns - receiver_time_ns) / 1_000_000_000

    return {
        "valid": True,
        "whitened_residue": whitened,
        "residue_ms": residue,
        "delta_cents": delta,
        "received_crc": received_crc,
        "calculated_crc": calculated_crc,
        "decoded_time_ns": decoded_ns,
        "decoded_time_utc": datetime.fromtimestamp(
            decoded_ns / 1e9, tz=timezone.utc
        ).isoformat(),
        "time_difference_seconds": difference_seconds,
    }


# ------------------------------ Diagnostics -------------------------------

def normal_cdf(x):
    """Standard normal CDF using the math library only."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gaussian_q(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def theoretical_bit_error(windows):
    sigma_z = math.sqrt(1.0 + LAMBDA_BASE * SIGMA_C ** 2)
    z = TILT * math.sqrt(windows) / sigma_z
    return gaussian_q(z), z


def run_benchmark(seed=1234, trials=200, window_values=(50, 100, 200, 500, 1000, 2000)):
    """Empirically measure bit error rates at several averaging budgets."""
    rng = np.random.default_rng(seed)
    print("\n[MONTE CARLO] Bit-error benchmark")
    print(" windows | theory Q(z) | measured bit errors | total bits")
    print("-" * 66)

    for windows in window_values:
        errors = 0
        total = 0
        for _ in range(trials):
            tx = rng.integers(0, 2, size=N_SLOTS, dtype=np.uint8)
            means = simulate_viewport(
                tx, windows=windows, seed=int(rng.integers(0, 2**32 - 1))
            )
            rx = decode_viewport(means)
            errors += int(np.count_nonzero(tx != rx))
            total += N_SLOTS
        theory, z = theoretical_bit_error(windows)
        measured = errors / total
        print(f"{windows:8d} | {theory:11.3e} | "
              f"{measured:19.3e} | {total:10d}  (z={z:.2f})")


def make_plots(means, tx_bits, rx_bits, output_prefix="rp_ghost"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOTS] Install matplotlib: pip install matplotlib")
        return

    slots = np.arange(N_SLOTS)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(slots, means, marker="o", linewidth=1, label="Viewport mean")
    ax.axhline(0.0, linestyle="--", linewidth=1, label="Decision threshold")
    ax.set_xlabel("Slot")
    ax.set_ylabel("Averaged normalized count")
    ax.set_title("RP-GHOST 40-slot viewport")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_viewport.png", dpi=150)

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.step(slots, tx_bits, where="mid", label="Transmitted", linewidth=1.5)
    ax.step(slots, rx_bits + 0.04, where="mid", label="Decoded",
            linewidth=1.2, linestyle="--")
    ax.set_yticks([0, 1])
    ax.set_xlabel("Slot")
    ax.set_ylabel("Bit")
    ax.set_title("Transmitted versus decoded frame")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_bits.png", dpi=150)
    plt.show()
    print(f"[PLOTS] Saved {output_prefix}_viewport.png and "
          f"{output_prefix}_bits.png")


# ---------------------------------- Main -----------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simulate and decode the RP-GHOST timestamp viewport."
    )
    parser.add_argument("--windows", type=int, default=DEFAULT_WINDOWS,
                        help="Number of averaged photon-count windows (default: 2000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible viewport noise")
    parser.add_argument("--webcam", type=int, default=None, metavar="INDEX",
                        help="Optional OpenCV webcam index used to derive frame hash")
    parser.add_argument("--hash", type=lambda x: int(x, 0), default=None,
                        help="Supply a fixed 16-bit whitening hash (e.g. 0x1234)")
    parser.add_argument("--delta-cents", type=int, default=0,
                        help="Signed 16-bit delta field (default: 0)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run Monte Carlo bit-error benchmark")
    parser.add_argument("--trials", type=int, default=200,
                        help="Trials per benchmark window setting")
    parser.add_argument("--plots", action="store_true",
                        help="Save and display viewport and bit comparison plots")
    args = parser.parse_args()

    if args.windows < 1:
        parser.error("--windows must be at least 1")
    if args.hash is not None and not 0 <= args.hash <= 0xFFFF:
        parser.error("--hash must be in the range 0..65535")

    frame_hash = args.hash if args.hash is not None else get_frame_hash(args.webcam)
    frame_hash &= 0xFFFF

    # Match the paper: schedule one second ahead, retaining millisecond
    # quantisation in the transmitted timestamp.
    start_ns = time.time_ns()
    scheduled_ns = start_ns + 1_000_000_000
    scheduled_ms = scheduled_ns // 1_000_000
    receiver_ns = time.time_ns()

    tx_bits, true_residue, whitened, check = encode_frame(
        scheduled_ms, args.delta_cents, frame_hash
    )

    print("=" * 72)
    print(" RP-GHOST OPTOELECTRONIC TIMESTAMP VIEWPORT")
    print("=" * 72)
    print(f"[ENCODE] UTC target       : "
          f"{datetime.fromtimestamp(scheduled_ns / 1e9, timezone.utc).isoformat()}")
    print(f"[ENCODE] Timestamp residue: {true_residue} ms")
    print(f"[ENCODE] Frame hash       : 0x{frame_hash:04X}")
    print(f"[ENCODE] Whitened residue : {whitened} (0x{whitened:04X})")
    print(f"[ENCODE] CRC-8            : 0x{check:02X}")
    print(f"[CHANNEL] Lambda          : {LAMBDA_BASE}")
    print(f"[CHANNEL] Common-mode sd  : {SIGMA_C}")
    print(f"[CHANNEL] Tilt            : {TILT}")
    print(f"[CHANNEL] Averaged windows: {args.windows}")

    means = simulate_viewport(tx_bits, windows=args.windows, seed=args.seed)
    rx_bits = decode_viewport(means)
    bit_errors = int(np.count_nonzero(tx_bits != rx_bits))
    decoded = decode_timestamp(rx_bits, frame_hash, receiver_ns)

    print(f"\n[DECODE] Bit errors       : {bit_errors}/{N_SLOTS}")
    print(f"[DECODE] CRC valid        : {decoded['valid']}")
    if decoded["valid"]:
        print(f"[DECODE] Residue          : {decoded['residue_ms']} ms")
        print(f"[DECODE] Delta            : {decoded['delta_cents']} cents")
        print(f"[DECODE] UTC timestamp    : {decoded['decoded_time_utc']}")
        print(f"[DECODE] Time difference  : "
              f"{decoded['time_difference_seconds']:+.6f} s")
        print("[DECODE] Expected         : approximately +1.000 s")
    else:
        print(f"[DECODE] Rejected         : {decoded['reason']}")
        print(f"[DECODE] CRC received     : 0x{decoded['received_crc']:02X}")
        print(f"[DECODE] CRC calculated   : 0x{decoded['calculated_crc']:02X}")

    theory, z = theoretical_bit_error(args.windows)
    print(f"\n[THEORY] Per-bit Gaussian Q(z): {theory:.4e} (z={z:.3f})")
    print(f"[THEORY] Marginal slot sd     : "
          f"{math.sqrt(1 + LAMBDA_BASE * SIGMA_C**2) / math.sqrt(args.windows):.6f}")

    if args.plots:
        make_plots(means, tx_bits, rx_bits)

    if args.benchmark:
        run_benchmark(trials=args.trials)


if __name__ == "__main__":
    main()
