import cv2
import numpy as np
import hashlib
import time
import math
import argparse

from datetime import datetime, timezone

ROWS = 5
COLS = 8
SLOTS = 40
WINDOWS = 2000
MODULUS = 60000
CRC_POLY = 0x07


def crc8(data):
    crc = 0

    for byte in data:
        crc ^= byte

        for _ in range(8):
            if crc & 128:
                crc = ((crc << 1) ^ CRC_POLY) & 255
            else:
                crc = (crc << 1) & 255

    return crc


def frame_hash(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    digest = hashlib.sha256(gray.tobytes()).digest()
    return int.from_bytes(digest[:2], "big")


def select_roi(frame):
    roi = cv2.selectROI(
        "Select camera observation area",
        frame,
        showCrosshair=True
    )

    cv2.destroyWindow("Select camera observation area")

    x, y, w, h = map(int, roi)

    if w < 80 or h < 50:
        raise RuntimeError("Invalid observation area")

    return x, y, w, h


def extract_channels(frame, roi):

    x, y, w, h = roi

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    crop = gray[y:y+h, x:x+w]

    ch, cw = crop.shape

    values = []

    for row in range(ROWS):

        for col in range(COLS):

            x0 = int(col * cw / COLS)
            x1 = int((col + 1) * cw / COLS)

            y0 = int(row * ch / ROWS)
            y1 = int((row + 1) * ch / ROWS)

            cell = crop[y0:y1, x0:x1]

            values.append(
                np.mean(cell)
            )

    return np.array(values)


def capture_camera(camera_index, windows):

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ok, first_frame = cap.read()

    if not ok:
        cap.release()
        raise RuntimeError("Initial camera capture failed")

    roi = select_roi(first_frame)

    measurements = np.zeros(
        (windows, SLOTS),
        dtype=np.float64
    )

    temporal_variance = np.zeros(
        (windows, SLOTS),
        dtype=np.float64
    )

    print("[CAMERA] Starting real optical acquisition")

    start = time.perf_counter()

    for i in range(windows):

        ok, frame = cap.read()

        if not ok:
            break

        measurements[i] = extract_channels(
            frame,
            roi
        )

        if i % 5 == 0:

            display = frame.copy()

            x, y, w, h = roi

            cv2.rectangle(
                display,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"FRAME {i+1}/{windows}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "LIVE CAMERA",
                display
            )

            if cv2.waitKey(1) & 255 in (27, ord("q")):
                break

    elapsed = time.perf_counter() - start

    cap.release()

    if i < 1:
        raise RuntimeError("Insufficient camera frames")

    measurements = measurements[:i+1]

    return measurements, roi, elapsed


def decode_channels(measurements):

    # Temporal average for each spatial channel.
    means = np.mean(measurements, axis=0)

    # Relative intensity threshold across the 40 channels.
    threshold = np.median(means)

    bits = (means > threshold).astype(np.uint8)

    # Camera intensity is not a calibrated Poisson photon count.
    # These are image-derived intensity measurements.

    return bits, means, threshold


def decode_payload(bits, whitening_hash):

    raw = np.packbits(bits).tobytes()

    payload = raw[:4]

    received_crc = raw[4]

    expected_crc = crc8(payload)

    valid = received_crc == expected_crc

    whitened = int.from_bytes(
        payload[:2],
        "big"
    )

    delta = int.from_bytes(
        payload[2:4],
        "big"
    )

    if delta >= 32768:
        delta -= 65536

    # Only de-whiten after CRC verification.
    if not valid:
        return None

    residue = whitened ^ whitening_hash

    if residue >= MODULUS:
        return None

    return {
        "residue": residue,
        "delta": delta,
        "crc": True
    }


def unwrap_timestamp(residue, reference_ms):

    k = math.floor(
        (reference_ms - residue) / MODULUS + 0.5
    )

    return residue + k * MODULUS


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        type=int,
        default=0
    )

    parser.add_argument(
        "--windows",
        type=int,
        default=WINDOWS
    )

    args = parser.parse_args()

    measurements, roi, elapsed = capture_camera(
        args.camera,
        args.windows
    )

    # Derive a reproducible receiver-side hash from the last
    # captured image. This is not necessarily the sender's hash.
    cap = cv2.VideoCapture(args.camera)
    ok, final_frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("Could not retrieve final camera frame")

    whitening_hash = frame_hash(final_frame)

    bits, means, threshold = decode_channels(
        measurements
    )

    decoded = decode_payload(
        bits,
        whitening_hash
    )

    print("\n========== CAMERA-ONLY RESULTS ==========")

    print("Camera frames:", len(measurements))
    print("Acquisition time:", round(elapsed, 3), "seconds")
    print("Channels:", SLOTS)
    print("Intensity threshold:", round(threshold, 3))
    print("Receiver hash:", hex(whitening_hash))

    print("\nExtracted bits:")
    print("".join(map(str, bits.tolist())))

    print("\nChannel intensity:")
    for i, value in enumerate(means):
        print(
            f"Channel {i:02d}: "
            f"{value:.3f} "
            f"bit={bits[i]}"
        )

    if decoded is not None:

        current_ms = time.time_ns() // 1_000_000

        recovered_ms = unwrap_timestamp(
            decoded["residue"],
            current_ms
        )

        recovered_time = datetime.fromtimestamp(
            recovered_ms / 1000,
            timezone.utc
        )

        print("\n[CRC] Valid")
        print("[DECODE] Residue:", decoded["residue"])
        print("[DECODE] Delta:", decoded["delta"])
        print("[DECODE] Timestamp:", recovered_time.isoformat())

        print(
            "[WARNING] A valid CRC does not prove that this "
            "timestamp was transmitted."
        )

    else:

        print("\n[DECODE] No valid timestamp frame.")
        print("[CRC] Invalid or residue outside valid range.")

    # Visualize the 40 measured channels.
    chart = np.zeros(
        (500, 1000, 3),
        dtype=np.uint8
    )

    for i, value in enumerate(means):

        x0 = i * 25
        height = int(np.clip(value, 0, 255) * 1.5)

        cv2.rectangle(
            chart,
            (x0, 450-height),
            (x0+20, 450),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            chart,
            str(bits[i]),
            (x0, 480),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 255, 0),
            1
        )

    cv2.imshow(
        "40 CAMERA CHANNELS",
        chart
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
