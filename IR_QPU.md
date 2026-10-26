# Build Guide: Desktop IR Photonic Processor (4×4 Optical Matrix-Vector Multiplier)

A hobbyist, bench-scale optical computing demonstrator. It computes **y = W·x** using light: IR LEDs encode the input vector, a pixelated ITO-electrode mask encodes the weight matrix, and photodiodes sum the transmitted light per output column. An Arduino host loads the data and reads back the result.

> **Expectations.** This is a learning/demo device. Expect roughly 4–128 bits of effective precision, drift with temperature, and weight updates limited by the Indium Tin Oxide plate (tens of ms).

> **Revision note.** This version assumes the **ITO plates are premade** (already patterned into pixel pads). No laser engraving or etching is needed, so the laser engraver is removed from the parts list and Step 2 is now inspect, map, and wire.

---

## 0. Design Overview

```
 IR LEDs (x1..x4)      Diffuser      ITO/LC weight mask (W 4x4)     Photodiode columns
 [row inputs] ───────► spreads ────► pixel (i,j) transmits ───────► sum each column j
 PWM-driven            each LED      W_ij of light from row i       → ADC → y_j = Σ x_i·W_ij
 across one row        along its row
```

**Optical stack (front to back):**

1. 2 IR LED rows (input vector)
2. IR diffusion plate
3. Premade patterned ITO plate A (16 pixel pads)
4. 4 photodiode columns, plus optional upconversion card + CMOS sensor for visual verification

---

## 1. Acquire Materials

### Optical
| Item | Qty | Notes |
|---|---|---|
| IR LEDs, **850 nm** (or 940 nm) | 2 (+2 spare) | 850 nm balances photodiode response and detector card transmission. |
| **Premade patterned ITO plate**, 4×4 pixel pads (or ≥16 independent pads) | 1 (+1 spare) | Confirm pad count, pitch, and where the contact pads sit. |
| **Premade common ITO plate** (unpatterned ITO glass) | 1
| IR diffusion plate (ground glass / matte acrylic) | 1 | Verify it passes IR. |
| IR detection / upconversion card | 1 | For alignment and debugging. Many cards respond poorly at 940 nm; 850 nm response is modest. A phone camera also works for 850 nm. |
| CMOS sensor module (Pi camera or USB webcam) | 1 | Remove the IR-cut filter if needed. Optional but useful. |
| Photodiodes, | 4–8 | One per pixel, plus spares. |

### Electronics
| Item | Qty | Notes |
|---|---|---|
| Arduino Mega 2560 | 1 | Pins and 10-bit ADC. |
| Hookup wire, breadboard or perfboard, headers | | |
| **Copper tape (conductive adhesive)** | 1 roll | Contact pads on ITO. |

### Tools
- 3D printer with **black** PLA/PETG
- Multimeter (continuity and resistance)
- Soldering iron and flux
- Isopropyl alcohol, lint-free wipes, nitrile gloves, safety glasses

---

## 2. Inspect the Premade ITO, Apply Copper Tape, Connect to the Arduino

### 2.1 Identify the conductive side and verify the pattern
1. Set the multimeter to resistance. The coated face reads **tens to ~100 Ω** between two points on the same pad; the glass face is open-circuit.
2. Mark the conductive side with a dot on the *back* (non-conductive) face.
3. **Verify isolation:** probe every pad against every neighbor. All must read open.
4. **Verify the traces:** probe each pad to its contact pad (expect tens to hundreds of Ω).
5. Clean with IPA and lint-free wipes; handle by the edges with gloves.

### 2.2 Map the pads
Draw a quick table of **pad number → grid position (row, column) → contact pad location**. Your firmware's `weight[n]` index must match this map. If the premade pattern has a different pixel count than 4×4, adjust the loop sizes and shift-register count in the firmware.

### 2.3 Apply copper tape contacts
1. Cut 2–3 mm wide copper tape strips and lay one on each of plate A's 16 contact pads, extending ~10 mm past the glass edge.
2. Burnish with a plastic spudger. For reliable contact, add a thin line of silver paint over the ITO/copper overlap and let it cure.
3. On plate B, apply a wider (~5 mm) strip along one edge for the common electrode.
4. Test continuity from each tape tail to its pad.
5. Reinforce tape junctions with Kapton tape (outside the active pixel area).
6. If your premade plates already have solder-ready or clip-ready contacts, skip the tape and use those.

### 2.4 Connect to the Arduino drive electronics
The TN cell must be driven with **AC (no net DC)** or the LC degrades quickly. Use a square wave of about 60–200 Hz.

**Scheme:** plate B (common) gets a square wave `COM` from one Arduino pin. Each pixel pad gets `COM XOR state_n` through a 74HC86.

- `state_n = 0`: in phase with COM → ~0 V across LC → relaxed state.
- `state_n = 1`: inverted → ±Vcc across LC → driven state.

---

## 3. Sandwich the Plates and Add the LEDs

### 3.1 Assemble the selective upconversion cell (Plate A + LC + Plate B)
Skip if you're using a sealed pre-aligned LC cell.


### 3.3 Mount the IR LEDs
1. Arrange 2 LEDs, centered on each horizontal edge.
2. Place them **1–5 mm** behind the diffuser so each LED lights its own row evenly. Use small black baffles between rows.
3. Wire each LED source to GND; gate to the PWM pin.

### 3.4 Mount photodiodes
1. Place **photodiodes above each pixel** at the output.

### 3.5 (Optional) Verification path
Place the **upconversion card** just behind the ITO plate to see a rough image of the transmitted IR (works best with ≥ 940–980 nm sources and bright LEDs). A CMOS sensor behind the card gives a camera view of the 4×4 array for calibration and debugging.

---

## 4. Secure with a Printed Housing

### 4.1 Design (CAD)
Make a rectangular tunnel with these pockets, in order along the optical axis:

1. LED bay with 2 holes
2. Diffuser slot
3. ITO cell cradle with cutouts for the copper tape/wires to exit
4. Photodiode bay with column baffles
5. Electronics tray (Arduino)
6. Optional camera mount

### 4.2 Print and assemble
1. Print and test-fit all parts. Sand as needed.
2. Place components in order; secure with M2/M3 screws or small hot-glue beads (never on optical surfaces).
3. Paint or flock internal surfaces matte black if you see reflections.
4. Close the housing and check that no ambient light reaches the photodiodes: with LEDs off, readings should be near the dark baseline.

---

## 5. Program the Host Controller (Offloading Algorithm)

The host (a PC running Python) sends matrices and vectors over serial. The Arduino loads weights into the LC mask, drives the LEDs with the input vector, reads the photodiodes, and returns the result. The PC then calibrates it and compares against the digital answer.

### 5.1 Calibrate first (essential)
1. **Dark offset** `d_j`: LEDs off, read A0–A3.
2. **LED response:** sweep each LED's PWM 0–255 with all pixels open; fit a curve (LEDs aren't linear).
3. **Pixel transmission** `T_ij(level)`: sweep each pixel's drive duty 0–100 % with one LED on; build a lookup table from desired weight to drive duty.
4. **Crosstalk matrix:** light one LED at a time with all pixels open; any signal on other columns is crosstalk to subtract.


## Safety

- **IR LEDs are invisible.** At these powers risk is low, but never stare into them at close range, and don't substitute lasers or high-power emitters without proper eye protection.
- **Glass edges** are sharp; file or edge-seal them.
