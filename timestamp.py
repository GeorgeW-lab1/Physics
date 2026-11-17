import cv2
import numpy as np
import hashlib
import time
import math
import sys

# ============================================================
# RP-GHOST OPTOELECTRONIC TIMESTAMP RECEIVER
# Real webcam + OpenCV + simulated photon-count viewport
# ============================================================

CAMERA_INDEX = 0

N_SLOTS = 40
WINDOWS = 2000

LAMBDA_BASE = 10000.0
SIGMA_C = 0.005
TILT = 0.25

LEAD_NS = 1_000_000_000_0  # 1 second lead time
MINUTE_MS = 60000

# Enable the latch-and-hold synchronization rule
HOLD_UNTIL_TARGET = True

rng = np.random.default_rng()

# ============================================================
# CRC-8: polynomial 0x07
# ============================================================

def crc8(data):
    crc = 0

    for byte in data:
        crc ^= byte

        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF

    return crc


# ============================================================
# CAMERA WHITENING
# ============================================================

def camera_hash(gray):
    digest = hashlib.sha256(gray.tobytes()).digest()
    return int.from_bytes(digest[:2], "big")


# ============================================================
# FRAME ENCODING
# ============================================================

def encode_frame(timestamp_ms, delta_cents, webcam_hash):

    code = timestamp_ms % MINUTE_MS

    whitened_code = code ^ webcam_hash

    delta_word = delta_cents & 0xFFFF

    payload = (
        whitened_code.to_bytes(2, "big") +
        delta_word.to_bytes(2, "big")
    )

    checksum = crc8(payload)

    frame = payload + bytes([checksum])

    bits = np.unpackbits(
        np.frombuffer(frame, dtype=np.uint8)
    )

    return bits.astype(np.uint8)


# ============================================================
# SIMULATED OPTOELECTRONIC CHANNEL
# ============================================================

def generate_viewport(bits):

    epsilon = rng.normal(
        0.0,
        SIGMA_C,
        WINDOWS
    )

    rates = LAMBDA_BASE * (
        1.0 + epsilon[:, None]
    )

    rates = np.maximum(rates, 1.0)

    rates = np.broadcast_to(
        rates,
        (WINDOWS, N_SLOTS)
    )

    photon_counts = rng.poisson(rates)

    normalized = (
        photon_counts - LAMBDA_BASE
    ) / math.sqrt(LAMBDA_BASE)

    tilt = TILT * (
        2.0 * bits.astype(float) - 1.0
    )

    viewport = normalized + tilt[None, :]

    return viewport.mean(axis=0)


# ============================================================
# BIT DECODER
# ============================================================

def decode_bits(measurements):

    return (
        measurements > 0.0
    ).astype(np.uint8)


# ============================================================
# FRAME DECODING AND CRC
# ============================================================

def decode_frame(bits, webcam_hash):

    packed = np.packbits(bits)

    data = packed.tobytes()

    if len(data) != 5:
        return None

    payload = data[:4]
    received_crc = data[4]

    calculated_crc = crc8(payload)

    crc_valid = (
        received_crc == calculated_crc
    )

    if not crc_valid:
        return None

    whitened_code = int.from_bytes(
        payload[:2], "big"
    )

    decoded_code = (
        whitened_code ^ webcam_hash
    )

    delta_word = int.from_bytes(
        payload[2:4], "big"
    )

    if delta_word >= 32768:
        delta_word -= 65536

    return decoded_code, delta_word


# ============================================================
# MODULAR TIMESTAMP RECONSTRUCTION
# ============================================================

def reconstruct_timestamp(code, reference_ns):

    reference_ms = reference_ns // 1_000_000

    minute_number = (
        reference_ms - code + MINUTE_MS // 2
    ) // MINUTE_MS

    decoded_ms = (
        code + minute_number * MINUTE_MS
    )

    return decoded_ms * 1_000_000


# ============================================================
# OPENCV DASHBOARD
# ============================================================

def draw_dashboard(
    measurements,
    decoded,
    crc_status,
    deadline_status,
    latency_ms,
    slack_ms,
    webcam_hash,
    target_ns,
    released=False
):

    height = 620
    width = 1000

    panel = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    panel[:] = (20, 24, 30)

    def text(label, x, y, scale=0.65,
             color=(220, 220, 220)):

        cv2.putText(
            panel,
            str(label),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA
        )

    text("RP-GHOST OPTOELECTRONIC RECEIVER [LATCHED]",
         25, 38, 0.85, (100, 220, 255))

    text("40 CHANNEL PHOTON VIEWPORT",
         25, 85, 0.65)

    for i, value in enumerate(measurements):

        x = 25 + (i % 10) * 95
        y = 115 + (i // 10) * 72

        cv2.rectangle(
            panel,
            (x, y),
            (x + 70, y + 35),
            (65, 65, 65),
            1
        )

        bar_length = int(
            min(abs(value) * 100, 65)
        )

        if value >= 0:
            color = (80, 220, 100)
            start = x + 35
            end = min(x + 35 + bar_length, x + 69)
        else:
            color = (80, 100, 240)
            start = max(x + 35 - bar_length, x + 1)
            end = x + 35

        cv2.line(
            panel,
            (start, y + 17),
            (end, y + 17),
            color,
            5
        )

        text(str(i), x, y + 51, 0.4)

    text("DECODER STATUS", 25, 420, 0.7,
         (100, 220, 255))

    text("Webcam hash: 0x%04X" % webcam_hash,
         25, 455)

    text("CRC: " + str(crc_status),
         25, 485)

    text("Deadline: " + str(deadline_status),
         25, 515)

    text("Decode latency: %.3f ms" % latency_ms,
         25, 545)

    text("Remaining slack: %.3f ms" % slack_ms,
         25, 575)

    if decoded is not None:

        text(
            "Timestamp: " +
            time.strftime(
                "%H:%M:%S",
                time.localtime(decoded / 1e9)
            ),
            520, 455, 0.65,
            (100, 255, 150)
        )

        text(
            "Released: " + str(released),
            520, 485
        )

    cv2.imshow(
        "RP-GHOST Receiver",
        panel
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("RP-GHOST OPENCV TIMESTAMP DECODER [LATCHED MODE]")
    print("=" * 65)

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    total = 0
    successful = 0
    missed = 0

    latencies = []
    latched_target_ns = None

    try:
        start_ns = time.time_ns()

        if latched_target_ns is None or start_ns >= latched_target_ns:
            latched_target_ns = start_ns + LEAD_NS
        while True:

            start_ns = time.time_ns()

            ret, frame = camera.read()

            if not ret:
                print("[ERROR] Camera frame unavailable.")
                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            webcam_h = camera_hash(gray)

            # ------------------------------------------------
            # 1. Hardware Latch Logic:
            # Hold/lock the target timestamp steady until the 
            # current time reaches or exceeds it.
            # ------------------------------------------------

            if latched_target_ns is not None and start_ns >= latched_target_ns:
                latched_target_ns = start_ns + LEAD_NS

            target_ns = latched_target_ns
            target_ms = target_ns // 1_000_000

            # ------------------------------------------------
            # 2. Encode latched target timestamp into frame
            # ------------------------------------------------

            bits = encode_frame(
                target_ms,
                0,
                webcam_h
            )

            # ------------------------------------------------
            # 3. Generate simulated viewport & decode
            # ------------------------------------------------

            measurements = generate_viewport(bits)
            decoded_bits = decode_bits(measurements)

            result = decode_frame(
                decoded_bits,
                webcam_h
            )

            decode_finished_ns = time.time_ns()

            latency_ms = (
                decode_finished_ns - start_ns
            ) / 1e6

            total += 1

            decoded_ns = None
            crc_status = "FAIL"
            deadline_status = "NOT ACCEPTED"
            slack_ms = 0.0
            released = False

            if result is not None:

                decoded_code, delta = result

                decoded_ns = reconstruct_timestamp(
                    decoded_code,
                    start_ns
                )

                slack_ms = (
                    decoded_ns - decode_finished_ns
                ) / 1e6

                crc_status = "PASS"

                if slack_ms > 0:

                    successful += 1
                    deadline_status = "PASS"

                    latencies.append(latency_ms)

                    print(
                        "[DECODE %05d] CRC=PASS | "
                        "Latched Delta=%.3f s | "
                        "Latency=%.3f ms | "
                        "Slack=%.3f ms | "
                        "Hash=0x%04X"
                        % (
                            total,
                            (decoded_ns - start_ns) / 1e9,
                            latency_ms,
                            slack_ms,
                            webcam_h
                        )
                    )

                else:
                    missed += 1
                    deadline_status = "MISSED"
                    print("[MISSED] Decode completed after target timestamp.")

            else:
                missed += 1
                print("[ERROR] CRC mismatch. Frame rejected.")

            # Display webcam feed & dashboard
            cv2.putText(
                frame,
                "LIVE WEBCAM | SHA256 WHITENING",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 100),
                2
            )

            cv2.putText(
                frame,
                "Hash: %04X" % webcam_h,
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 100),
                1
            )

            cv2.imshow("OpenCV Camera", frame)

            draw_dashboard(
                measurements,
                decoded_ns,
                crc_status,
                deadline_status,
                latency_ms,
                slack_ms,
                webcam_h,
                target_ns,
                released
            )

            # ------------------------------------------------
            # 4. Optional Hold-Until-Target Latch Rule
            # ------------------------------------------------

            if (
                HOLD_UNTIL_TARGET
                and decoded_ns is not None
                and slack_ms > 0
            ):

                while time.time_ns() < decoded_ns:

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return

                    remaining = (
                        decoded_ns - time.time_ns()
                    ) / 1e9

                    if remaining > 0.002:
                        time.sleep(
                            min(remaining - 0.001, 0.01)
                        )

                release_ns = time.time_ns()
                released = True

                release_error_ms = (
                    release_ns - decoded_ns
                ) / 1e6

                print(
                    "[RELEASE] Latch target reached | "
                    "Release error: %.3f ms"
                    % release_error_ms
                )

                # Reset the latch so a new target is captured after release
                latched_target_ns = start_ns + LEAD_NS

            if total % 20 == 0:
                mean_latency = np.mean(latencies) if latencies else 0
                print("-" * 60)
                print(f"Cycles: {total} | Successful: {successful} | Missed: {missed}")
                print(f"Mean decode latency: {mean_latency:.3f} ms")
                print("-" * 60)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[STOP] Keyboard interrupt.")

    finally:
        camera.release()
        cv2.destroyAllWindows()

        print("\n========== FINAL STATISTICS ==========")
        print("Total cycles:", total)
        print("Successful:", successful)
        print("Missed:", missed)

        if latencies:
            print(f"Mean latency: {np.mean(latencies):.3f} ms")
            print(f"Maximum latency: {np.max(latencies):.3f} ms")
        print("======================================")


if __name__ == "__main__":
    main()
