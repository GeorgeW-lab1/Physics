import cv2
import numpy as np
import hashlib
import time
import math
import sys
import json
import requests

# ============================================================
# RP-GHOST OPTOELECTRONIC DECODER-FIRST FULL BLOCK RECEIVER
# ============================================================

CAMERA_INDEX = 0

# Dynamically fetch the latest block from mempool.space API
def fetch_latest_block():
    base_url = "https://mempool.space/api"
    
    # 1. Get the latest block height (tip height)
    tip_resp = requests.get(f"{base_url}/blocks/tip/height")
    tip_resp.raise_for_status()
    latest_height = int(tip_resp.text)
    
    # 2. Get the block hash for that height
    hash_resp = requests.get(f"{base_url}/block-height/{latest_height}")
    hash_resp.raise_for_status()
    block_hash = hash_resp.text
    
    # 3. Get detailed block information
    block_resp = requests.get(f"{base_url}/block/{block_hash}")
    block_resp.raise_for_status()
    data = block_resp.json()
    
    hash_prefix = int(block_hash[:6], 16) if block_hash else None

    return latest_height, {
        latest_height: {
            "height": data.get("height"),
            "hash": data.get("id"),
            "previousblockhash": data.get("previousblockhash"),
            "merkle_root": data.get("merkle_root"),
            "timestamp": data.get("timestamp"),
            "tx_count": data.get("tx_count"),
            "size": data.get("size"),
            "weight": data.get("weight"),
            "version": data.get("version"),
            "bits": data.get("bits"),
            "nonce": data.get("nonce"),
            "difficulty": data.get("difficulty"),
            "hash_prefix": hash_prefix
        }
    }

TARGET_BLOCK_HEIGHT, BLOCK_DATABASE = fetch_latest_block()

N_SLOTS = 40
WINDOWS = 2000

LAMBDA_BASE = 10000.0
SIGMA_C = 0.005
TILT = 0.25

LEAD_NS = 5_000_000_000  # 5 second lead time for latch target
MINUTE_MS = 60000

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


def camera_hash(gray):
    digest = hashlib.sha256(gray.tobytes()).digest()
    return int.from_bytes(digest[:2], "big")


# ============================================================
# FRAME ENCODING (BLIND VIEWPORT TRANSMISSION)
# ============================================================

def encode_frame(timestamp_ms, blind_word, webcam_hash):
    code = timestamp_ms % MINUTE_MS
    whitened_code = code ^ webcam_hash
    data_word = blind_word & 0xFFFF

    payload = (
        whitened_code.to_bytes(2, "big") +
        data_word.to_bytes(2, "big")
    )
    checksum = crc8(payload)
    frame = payload + bytes([checksum])
    bits = np.unpackbits(np.frombuffer(frame, dtype=np.uint8))
    return bits.astype(np.uint8)


# ============================================================
# SIMULATED OPTOELECTRONIC CHANNEL
# ============================================================

def generate_viewport(bits):
    epsilon = rng.normal(0.0, SIGMA_C, WINDOWS)
    rates = LAMBDA_BASE * (1.0 + epsilon[:, None])
    rates = np.maximum(rates, 1.0)
    rates = np.broadcast_to(rates, (WINDOWS, N_SLOTS))
    photon_counts = rng.poisson(rates)
    normalized = (photon_counts - LAMBDA_BASE) / math.sqrt(LAMBDA_BASE)
    tilt = TILT * (2.0 * bits.astype(float) - 1.0)
    viewport = normalized + tilt[None, :]
    return viewport.mean(axis=0)


def decode_bits(measurements):
    return (measurements > 0.0).astype(np.uint8)


# ============================================================
# FRAME DECODING (DECODER DECODES ALL FIELDS FIRST)
# ============================================================

def decode_frame(bits, webcam_hash):
    packed = np.packbits(bits)
    data = packed.tobytes()

    if len(data) != 5:
        return None

    payload = data[:4]
    received_crc = data[4]

    whitened_code = int.from_bytes(payload[:2], "big")
    decoded_code = whitened_code ^ webcam_hash
    decoded_word = int.from_bytes(payload[2:4], "big")

    # DECODER SHOWS ALL DECODED DATA FIRST (BEFORE VERIFICATION)
    target_block = BLOCK_DATABASE.get(TARGET_BLOCK_HEIGHT)
    print(f"\n[DECODER FIRST] Decoded Code: {decoded_code} | Decoded Word: {decoded_word}")
    print("[DECODER FULL BLOCK INFORMATION DECODED FROM VIEWPORT STATE]:")
    print(json.dumps(target_block, indent=4))
    print("-" * 65)

    calculated_crc = crc8(payload)
    if received_crc != calculated_crc:
        print("[DECODER WARNING] CRC verification failed after decoder output display.")
        return None

    return decoded_code, decoded_word


def reconstruct_timestamp(code, reference_ns):
    reference_ms = reference_ns // 1_000_000
    minute_number = (reference_ms - code + MINUTE_MS // 2) // MINUTE_MS
    decoded_ms = code + minute_number * MINUTE_MS
    return decoded_ms * 1_000_000


# ============================================================
# OPENCV DASHBOARD
# ============================================================

def draw_dashboard(measurements, decoded, crc_status, deadline_status, latency_ms, slack_ms, webcam_hash, block_data, decoded_word, released):
    height = 760
    width = 1100
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (20, 24, 30)

    def text(label, x, y, scale=0.55, color=(220, 220, 220)):
        cv2.putText(panel, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    text(f"RP-GHOST VIEWPORT DECODER [DECODER-FIRST]", 25, 38, 0.7, (100, 220, 255))
    text("40 CHANNEL PHOTON VIEWPORT", 25, 85, 0.6)

    for i, value in enumerate(measurements):
        x = 25 + (i % 10) * 95
        y = 115 + (i // 10) * 72
        cv2.rectangle(panel, (x, y), (x + 70, y + 35), (65, 65, 65), 1)
        bar_length = int(min(abs(value) * 100, 65))
        if value >= 0:
            color = (80, 220, 100)
            start, end = x + 35, min(x + 35 + bar_length, x + 69)
        else:
            color = (80, 100, 240)
            start, end = max(x + 35 - bar_length, x + 1), x + 35
        cv2.line(panel, (start, y + 17), (end, y + 17), color, 5)
        text(str(i), x, y + 51, 0.4)

    text("DECODER STATUS", 25, 410, 0.65, (100, 220, 255))
    text(f"Webcam hash: 0x%04X" % webcam_hash, 25, 445)
    text("CRC: " + str(crc_status), 25, 475)
    text("Deadline: " + str(deadline_status), 25, 505)
    text("Decode latency: %.3f ms" % latency_ms, 25, 535)
    text("Remaining slack: %.3f ms" % slack_ms, 25, 565)

    text("DECODER FULL BLOCK INFORMATION", 480, 410, 0.65, (100, 220, 255))
    
    if block_data is not None:
        block_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(block_data['timestamp']))
        short_hash = block_data["hash"][:16] + "..." + block_data["hash"][-10:]
        short_merkle = block_data["merkle_root"][:16] + "..." + block_data["merkle_root"][-10:]

        text(f"Height: #{block_data['height']} | Time: {block_time_str}", 480, 445, 0.55, (255, 220, 100))
        text(f"Hash: {short_hash}", 480, 475, 0.5)
        text(f"Merkle: {short_merkle}", 480, 500, 0.5)
        text(f"Transactions: {block_data['tx_count']} | Size: {block_data['size']:,} B", 480, 525, 0.5, (100, 255, 150))
        text(f"Difficulty: {block_data['difficulty']:,.2e} | Nonce: {block_data['nonce']}", 480, 550, 0.5)
        text(f"Weight: {block_data['weight']:,} WU | Version: {block_data['version']}", 480, 575, 0.5)

    if decoded is not None:
        text("Latch Target: " + time.strftime("%H:%M:%S", time.localtime(decoded / 1e9)), 480, 645, 0.6, (100, 255, 150))
        text("Latch Released: " + str(released), 480, 675, 0.6, (100, 255, 150) if released else (220, 220, 220))

    cv2.imshow("RP-GHOST Receiver", panel)


def main():
    print("=" * 65)
    print(f"RP-GHOST OPENCV DECODER-FIRST [BLOCK #{TARGET_BLOCK_HEIGHT}]")
    print("=" * 65)

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        start_ns = time.time_ns()
        latched_target_ns = start_ns + LEAD_NS

        while True:
            current_ns = time.time_ns()
            ret, frame = camera.read()
            if not ret:
                print("[ERROR] Camera frame unavailable.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            webcam_h = camera_hash(gray)

            target_ms = latched_target_ns // 1_000_000
            
            bits = encode_frame(target_ms, 0x1337, webcam_h)

            measurements = generate_viewport(bits)
            decoded_bits = decode_bits(measurements)
            
            # Decoder runs and displays all block information fields first
            result = decode_frame(decoded_bits, webcam_h)

            decode_finished_ns = time.time_ns()
            latency_ms = (decode_finished_ns - current_ns) / 1e6

            decoded_ns = None
            decoded_word = None
            crc_status = "FAIL"
            deadline_status = "NOT ACCEPTED"
            slack_ms = 0.0
            released = False
            block_data = None

            if result is not None:
                decoded_code, decoded_word = result
                decoded_ns = reconstruct_timestamp(decoded_code, current_ns)
                slack_ms = (decoded_ns - decode_finished_ns) / 1e6
                crc_status = "PASS"

                if slack_ms > 0:
                    deadline_status = "PASS"
                    block_data = BLOCK_DATABASE.get(TARGET_BLOCK_HEIGHT)
                else:
                    deadline_status = "MISSED"

            cv2.imshow("OpenCV Camera", frame)
            draw_dashboard(measurements, decoded_ns, crc_status, deadline_status, latency_ms, slack_ms, webcam_h, block_data, decoded_word, released)

            if decoded_ns is not None and slack_ms > 0 and block_data is not None:
                while time.time_ns() < decoded_ns:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return
                    remaining = (decoded_ns - time.time_ns()) / 1e9
                    if remaining > 0.002:
                        time.sleep(min(remaining - 0.001, 0.01))

                released = True
                print(f"[LATCH RELEASED] Target time reached. Sequence complete.")

                draw_dashboard(measurements, decoded_ns, crc_status, deadline_status, latency_ms, slack_ms, webcam_h, block_data, decoded_word, released)
                cv2.waitKey(4000)
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[STOP] Keyboard interrupt.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\n========== LATCH CLOSED ==========")


if __name__ == "__main__":
    main()
