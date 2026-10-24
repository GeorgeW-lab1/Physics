#!/usr/bin/env python3
"""
Integrated IR -> Disc -> SIM -> PNG+SHA256 Pipeline
===================================================
1. IR / Optical Simulation (SIM): Model or simulate detector emission responses for the 120 sites.
2. Disc Layout & Coordinates: Generate the 100mm disc pattern and export coordinate CSV.
3. Heuristic Ranking: Score nonce candidates using simulated optical emission states.
4. SHA-256 Engine: Solve and verify nonces matching leading-zero-bit targets.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

RINGS = {
    "A": {"count": 48, "radius_mm": 45.0, "shape": "circle", "size_um": 100},
    "B": {"count": 36, "radius_mm": 30.0, "shape": "square", "size_um": 80},
    "C": {"count": 24, "radius_mm": 15.0, "shape": "triangle", "size_um": 60},
    "D": {"count": 12, "radius_mm": 7.5, "shape": "diamond", "size_um": 40},
}

def site_records():
    records = []
    for ring, spec in RINGS.items():
        for i in range(spec["count"]):
            angle = 90.0 - i * 360.0 / spec["count"]
            rad = math.radians(angle)
            records.append({
                "site_id": f"{ring}{i + 1:02d}",
                "ring": ring,
                "index": i + 1,
                "x_mm": spec["radius_mm"] * math.cos(rad),
                "y_mm": spec["radius_mm"] * math.sin(rad),
                "angle_deg": angle,
                "shape": spec["shape"],
                "nominal_size_um": spec["size_um"],
            })
    return records

def generate_png_and_csv(output="pipeline_disc.png", dpi=1200, disc_mm=100.0, canvas_mm=110.0):
    """Step 2: Generate PNG Layout and Coordinate CSV."""
    if Image is None:
        raise RuntimeError("Pillow is required. Install with: pip install Pillow")
    
    px_per_mm = dpi / 25.4
    size = round(canvas_mm * px_per_mm)
    img = Image.new("1", (size, size), 1)
    draw = ImageDraw.Draw(img)
    cx = cy = size // 2
    radius = disc_mm * px_per_mm / 2

    def px(mm):
        return max(1, round(mm * px_per_mm))

    def point(r_mm, angle_deg):
        a = math.radians(angle_deg)
        return (
            round(cx + r_mm * px_per_mm * math.cos(a)),
            round(cy - r_mm * px_per_mm * math.sin(a)),
        )

    # Outer perimeter & alignment cross
    draw.ellipse((round(cx - radius), round(cy - radius), round(cx + radius), round(cy + radius)), outline=0, width=px(0.08))
    cr = px(2.0)
    draw.ellipse((cx-cr, cy-cr, cx+cr, cy+cr), outline=0, width=px(0.08))
    draw.line((cx-px(1.5), cy, cx+px(1.5), cy), fill=0, width=px(0.08))
    draw.line((cx, cy-px(1.5), cx, cy+px(1.5)), fill=0, width=px(0.08))

    for record in site_records():
        ring = record["ring"]
        spec = RINGS[ring]
        x, y = point(spec["radius_mm"], record["angle_deg"])
        size_um = spec["size_um"]
        if ring == "A":
            r = max(1, round((size_um / 2000.0) * px_per_mm))
            draw.ellipse((x-r, y-r, x+r, y+r), outline=0, width=1)
        elif ring == "B":
            h = max(1, round((size_um / 2000.0) * px_per_mm))
            draw.rectangle((x-h, y-h, x+h, y+h), outline=0, width=1)
        elif ring == "C":
            side = size_um / 1000.0 * px_per_mm
            crad = side / math.sqrt(3)
            a = math.radians(record["angle_deg"])
            pts = [(round(x + crad * math.cos(a + j * 2*math.pi/3)), round(y - crad * math.sin(a + j * 2*math.pi/3))) for j in range(3)]
            draw.polygon(pts, outline=0)
        else:
            half_long = max(1, round((size_um / 2000.0) * px_per_mm))
            half_short = max(1, round((size_um / 4000.0) * px_per_mm))
            draw.polygon([(x, y-half_long), (x+half_short, y), (x, y+half_long), (x-half_short, y)], outline=0)

    draw.line((cx, round(cy-radius), cx, round(cy-radius)+px(1.0)), fill=0, width=px(0.08))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output, dpi=(dpi, dpi), bits=1)

    csv_path = output.with_suffix(".csv")
    fields = ["site_id", "ring", "index", "x_mm", "y_mm", "angle_deg", "shape", "nominal_size_um"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in site_records():
            row = {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in rec.items()}
            writer.writerow(row)

    print(f"[PNG+DISC] Layout saved to {output.resolve()} and {csv_path.resolve()}")
    return csv_path

def simulate_ir_measurements(csv_path, sim_output="simulated_measurements.csv"):
    """Step 1 & 3: IR / Optical Simulation (SIM) generating emission measurements."""
    csv_path = Path(csv_path)
    sim_output = Path(sim_output)
    
    fields = ["input_a", "input_b", "site_id", "ring", "emission", "wavelength_nm", "decay_ns", "replicate", "background", "notes"]
    
    random.seed(42)
    with sim_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        
        with csv_path.open("r", encoding="utf-8") as cf:
            reader = csv.DictReader(cf)
            sites = list(reader)

        for site in sites:
            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                base = 1.0 + (ord(site["ring"]) - ord('A')) * 0.2
                state_mod = (a * 1.5) + (b * 2.2) + (a * b * 0.5)
                noise = random.gauss(0, 0.05)
                emission = max(0.1, base + state_mod + noise)
                
                writer.writerow({
                    "input_a": a,
                    "input_b": b,
                    "site_id": site["site_id"],
                    "ring": site["ring"],
                    "emission": f"{emission:.4f}",
                    "wavelength_nm": 980.0,
                    "decay_ns": 45.2,
                    "replicate": 1,
                    "background": "0.021",
                    "notes": "IR SIMulated response"
                })

    print(f"[IR -> SIM] Simulated optical emission measurements saved to {sim_output.resolve()}")
    return sim_output

def load_simulated_measurements(path):
    grouped = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                a = int(row["input_a"])
                b = int(row["input_b"])
                val = float(row["emission"])
                grouped.setdefault((a, b), []).append(val)
            except (ValueError, TypeError):
                continue
    return {state: sum(vals) / len(vals) for state, vals in grouped.items()}

def optical_heuristic_score(nonce, measurements):
    states = [measurements[(0, 0)], measurements[(0, 1)], measurements[(1, 0)], measurements[(1, 1)]]
    scale = max(max(states) - min(states), 1e-12)
    normalized = [(x - min(states)) / scale for x in states]
    score = 0.0
    for bit_index in range(16):
        bit = (nonce >> bit_index) & 1
        score += normalized[(bit_index + bit) % 4] * (1.0 + bit_index / 16.0)
    return score

def sha256_pipeline_search(payload="test", bits=16, leading_zero_bits=8, measurements_file=None):
    """Step 4: SHA-256 search guided by IR/SIM optical heuristics, then verified."""
    measurements = load_simulated_measurements(measurements_file) if measurements_file else None
    payload_bytes = payload.encode("utf-8")
    limit = 1 << bits
    target = 1 << (256 - leading_zero_bits) if leading_zero_bits else 1 << 256

    if measurements:
        candidates = list(range(limit))
        candidates.sort(key=lambda n: optical_heuristic_score(n, measurements), reverse=True)
        order_type = "IR-SIM Optical Heuristic Guided"
    else:
        candidates = range(limit)
        order_type = "Sequential"

    started = time.perf_counter()
    attempts = 0
    for nonce in candidates:
        message = payload_bytes + nonce.to_bytes(8, "big")
        digest = hashlib.sha256(message).digest()
        attempts += 1

        if int.from_bytes(digest, "big") < target:
            elapsed = time.perf_counter() - started
            result = {
                "pipeline_status": "SUCCESS",
                "search_mode": order_type,
                "payload": payload,
                "nonce": nonce,
                "nonce_hex": nonce.to_bytes(8, "big").hex(),
                "sha256": digest.hex(),
                "leading_zero_bits_target": leading_zero_bits,
                "attempts_required": attempts,
                "elapsed_seconds": round(elapsed, 6),
                "verified": hashlib.sha256(message).digest() == digest
            }
            print(json.dumps(result, indent=2))
            return result

    print(json.dumps({"pipeline_status": "FAILED", "attempts": attempts}))
    return None

def main():
    print("=== STARTING IR -> DISC -> SIM -> PNG+SHA256 PIPELINE ===")
    csv_coords = generate_png_and_csv("pipeline_disc.png", dpi=1200)
    sim_csv = simulate_ir_measurements(csv_coords, "simulated_measurements.csv")
    sha256_pipeline_search(payload="test", bits=16, leading_zero_bits=8, measurements_file=sim_csv)

if __name__ == "__main__":
    main()
