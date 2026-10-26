# Build Guide: Desktop IR Photonic Processor (4×4 Optical Matrix-Vector Multiplier)

A hobbyist, bench-scale optical computing demonstrator. It computes **y = W·x** using light: IR LEDs encode the input vector, a pixelated ITO-electrode liquid-crystal mask encodes the weight matrix, and photodiodes sum the transmitted light per output column. An Arduino host loads the data and reads back the result.

> **Expectations.** This is a learning/demo device. Expect roughly 4–6 bits of effective precision, drift with temperature, and speeds limited by the liquid crystal (tens of ms per weight update). It will not beat a CPU. Its value is showing how analog optical multiply-accumulate works.

---

## 0. Design Overview

```
 IR LEDs (x1..x4)      Diffuser      ITO/LC weight mask (W 4x4)     Photodiode columns
 [row inputs] ───────► spreads ────► pixel (i,j) transmits ───────► sum each column j
 PWM-driven            each LED      W_ij of light from row i       → ADC → y_j = Σ x_i·W_ij
 across one row        along its row
```

**Optical stack (front to back):**

1. 4 IR LED rows (input vector)
2. IR diffusion plate (spreads each LED evenly across its row of 4 pixels)
3. Front polarizer
4. ITO plate A, etched into 16 pixel pads
5. Liquid crystal layer (twisted-nematic cell)
6. ITO plate B, unetched common electrode
7. Rear polarizer (crossed or parallel per your LC type)
8. 4 photodiode columns, plus optional upconversion card + CMOS sensor for visual verification

### Important design note: ITO alone does not modulate light

ITO is a *transparent conductor*. It supplies the electrodes, but it does not change transmission by itself. To get programmable weights you need an electro-optic layer between two ITO plates:

| Option | Drive | Verdict |
|---|---|---|
| **TN liquid-crystal cell + polarizers** | ~3–5 V AC square wave | **Recommended.** Arduino-compatible. Salvage a TN/LCD panel or buy a blank ITO LC cell. |
| PDLC (switchable privacy) film | ~24–60 V AC | Works without polarizers, but needs a high-voltage AC driver. Higher shock risk. |

This guide uses the TN option. Check that your polarizers and LC cell actually transmit at your LED wavelength (see Step 1 notes).

---

## 1. Acquire Materials

### Optical
| Item | Qty | Notes |
|---|---|---|
| IR LEDs, **850 nm** (or 940 nm) | 4 (+2 spare) | 850 nm is a good balance: visible-ish on cameras, good photodiode response, reasonable polarizer/LC transmission. |
| ITO-coated glass plates (≈ 20 Ω/sq or lower) | 2 min, buy 4+ | Spares for etching mistakes. Float glass 1.1 mm. |
| TN liquid-crystal material / blank LC cell, **or** a salvaged LCD | 1 | Needed for modulation. See design note above. |
| Linear polarizer sheets rated for near-IR | 2 | **Standard visible polarizers often stop transmitting beyond ~800 nm.** Buy IR-rated film for 850/940 nm. |
| IR diffusion plate (or ground glass / matte acrylic) | 1 | Test that it passes IR. |
| IR detection / upconversion card | 1 | Used for alignment and debugging. **Most upconversion cards are tuned for ~800–1600 nm and respond poorly at 940 nm, and 850 nm response is modest.** A phone camera also works for 850 nm. |
| CMOS sensor module (e.g., Raspberry Pi camera or USB webcam) | 1 | Remove the IR-cut filter if the camera can't see your LEDs. Optional but great for visual debugging. |
| Photodiodes, e.g. BPW34 or SFH 206K (IR-sensitive) | 4–8 | One per output column; spares for matching. |

### Electronics
| Item | Qty | Notes |
|---|---|---|
| Arduino Mega 2560 (or Uno + shift registers) | 1 | Mega has the pins and 10-bit ADC you need. |
| 74HC595 shift registers | 2 | Drive 16 pixel pads from 3 Arduino pins. |
| 74HC86 (XOR gates) or equivalent | 1–2 | Generates AC drive phase per pixel. |
| MOSFETs, e.g. 2N7000 / logic-level N-channel | 4 | LED current drivers. |
| Op-amps for transimpedance amplifiers (TIA), e.g. MCP6002/TLC2272 | 2 | One per 2 photodiodes. Rail-to-rail, single supply. |
| Resistors: LED current limiting (~47–100 Ω), TIA feedback (1–10 MΩ), 10 kΩ pulldowns | assorted | |
| Capacitors: 100 nF decoupling, 10–47 pF TIA feedback | assorted | |
| Hookup wire, breadboard or perfboard, header pins | | |
| **Copper tape (conductive adhesive)** | 1 roll | For ITO contact pads. |
| Conductive silver paint or ink (optional) | | Improves copper-to-ITO contact. |

### Tools
| Item | Notes |
|---|---|
| **Laser engraver** | See Step 2: must be capable of ablating ITO. |
| 3D printer + black PLA/PETG | Black, opaque filament only (blocks stray light). |
| Multimeter (with continuity and resistance) | Essential for verifying etches. |
| Soldering iron, flux, helping hands | |
| Isopropyl alcohol, lint-free wipes | |
| Safety glasses, nitrile gloves | |

### Notes before you buy
- **Laser choice matters.** ITO ablation usually needs a **fiber (1064 nm), UV (355 nm), or CO2 (10.6 µm)** laser, or a pulsed source. Common 450 nm blue diode engravers generally cannot clean-ablate ITO. If yours can't, use the **chemical-etch alternative** in Step 2B.
- Confirm the **conductive side** of every ITO plate with a multimeter (Step 2.1). Only one face is coated.

---

## 2. Engrave the ITO Arrays and Attach Contacts

### 2.1 Identify and clean
1. Set the multimeter to resistance. Touch both probes to one face of the plate, ~1 cm apart. The conductive side reads **tens to ~100 Ω**; the glass side is open-circuit.
2. Mark the conductive side with a corner dot using a marker on the *back* (non-conductive) side.
3. Clean with isopropyl alcohol and lint-free wipes. Handle by edges with gloves.

### 2.2 Make the pixel template (Plate A)
Design a **4×4 grid of pixel pads**, each 6 × 6 mm, with 1 mm gaps, each with a thin trace running to an edge contact pad.

- Draw the pattern in Inkscape/LightBurn/any vector tool. Use **filled shapes for areas to remove** (the gaps and everything *outside* the pads and traces).
- Pixel pads and traces stay as ITO; the laser removes the ITO *around* them.
- Keep trace width ≥ 0.5 mm, spacing between isolated conductors ≥ 0.3 mm.
- Put all 16 contact pads along **one long edge**, ~2 mm wide at 2.54 mm pitch where possible, so copper tape or header pins fit.
- Export as SVG/DXF.

**Plate B** stays **unetched** (common ground electrode). Only add one contact strip along an edge.

### 2.3 Laser etch (Method A: laser ablation)
1. Place Plate A **coated side up** on a sacrificial backing (ceramic tile or aluminum). Secure with tape on the corners.
2. Start with **test strips on a spare plate**. Sweep power and speed to find a setting that fully removes ITO without cracking the glass. Start low-power, fast, and multiple passes if needed.
3. Verify each test line with the multimeter. Resistance across the line must be **open (OL / >20 MΩ)**.
4. Run the real template on Plate A.
5. **Continuity check:** probe each pixel pad against each neighbor. All must be isolated. Probe each pad against its contact pad (should read ~tens to hundreds of Ω depending on trace length).
6. Clean with IPA to remove ablation debris.

### 2.3B Chemical etch (Method B: if your laser can't ablate ITO)
> Uses acid. Do this outdoors or in a ventilated area, with gloves, goggles, and an apron. Neutralize and dispose of waste per local regulations.

1. Print the **pad pattern as positive mask** onto transfer paper, or apply a tape/resist mask over the pads and traces (the laser engraver can cut vinyl or tape stencil if you have one).
2. Sprinkle zinc powder over the exposed ITO, then apply dilute **hydrochloric acid** (~10–20 %) with a cotton swab. Zinc plus acid removes the exposed ITO within seconds to a minute.
3. Rinse thoroughly with water, remove the resist, clean with IPA, then repeat the continuity checks from 2.3 step 5.

### 2.4 Apply copper tape contacts
1. Cut 2–3 mm wide copper tape strips and lay one on each of Plate A's 16 contact pads, extending past the plate edge by ~10 mm.
2. Burnish with a plastic spudger. For reliable contact, add a thin line of silver paint at the ITO/copper overlap and let it cure.
3. Do the same for Plate B's common strip (use a wider strip, ~5 mm).
4. Test continuity from the tape end to the pad.
5. Reinforce tape junctions with a dab of Kapton tape over the glass (not over the active pixel area).

### 2.5 Connect to the Arduino drive electronics
The TN cell must be driven with **AC (no net DC)** or the LC degrades quickly. Use a square wave around 60–200 Hz.

**Scheme:** the common electrode (Plate B) gets a square wave `COM` from one Arduino pin. Each pixel pad gets `COM` XOR `state_i` through a 74HC86.

- `state_i = 0`: pixel is in phase with COM → ~0 V across LC → relaxed state.
- `state_i = 1`: pixel is inverted → ±Vcc across LC → driven state.

**Wiring (Mega example):**

| Signal | Arduino Pin |
|---|---|
| 74HC595 DATA | D4 |
| 74HC595 CLOCK | D5 |
| 74HC595 LATCH | D6 |
| COM square wave | D7 |
| LED row 1–4 gates (via MOSFETs, PWM) | D2, D3, D8, D9 |
| Photodiode column 1–4 (after TIAs) | A0–A3 |

1. Wire the two 74HC595s in series → 16 outputs `Q0..Q15` (pixel states).
2. Feed each `Qn` into one input of an XOR gate; feed `COM` to the other input; the XOR output goes through a ~1 kΩ series resistor to pad `n`.
3. Connect Plate B's copper strip to the `COM` net **directly**.
4. Share ground with the Arduino. Decouple every IC with 100 nF.

**Test before assembly:** with a spare LC cell (or the final plates once assembled in Step 3), push `state=1` on a pixel and confirm with a polarized-light setup (or the camera) that it visibly changes transmission.

---

## 3. Sandwich the Plates and Add the LEDs

### 3.1 Build the LC cell (Plate A + LC + Plate B)
Skip this if you're reusing a pre-made LC cell; in that case etch/replace electrodes per your cell type and move on to 3.2.

1. Place **spacer beads (5–10 µm)** or thin Mylar spacers at the corners of Plate B.
2. Apply a thin perimeter bead of UV-cure or epoxy sealant to Plate B, leaving a small fill gap.
3. Align Plate A on top, **conductive faces inward**, with the contact edges offset so both sets of copper tape are accessible.
4. Cure the seal, fill with LC by capillary action through the gap, then seal the gap.
5. Alignment layers (rubbed polyimide) are required for a proper twisted-nematic orientation. If you can't make those, buy or salvage a pre-aligned LC cell and only re-pattern the ITO if its structure allows, or use PDLC film (no alignment layer needed) with a high-voltage driver.

### 3.2 Add polarizers and diffuser
1. Stick the **front polarizer** to the outside of Plate A, and the **rear polarizer** to the outside of Plate B. Use **crossed** axes for "normally white," or **parallel** for "normally black" (consistent with your LC twist). Verify with the camera/IR card.
2. Place the **IR diffusion plate** in front of the front polarizer, 2–5 mm gap.

### 3.3 Mount the IR LEDs
1. Arrange **4 LEDs in a vertical column** at the left edge, one per pixel row, centered on each row (~7 mm pitch to match the pixel pitch).
2. Place them **15–25 mm** back from the diffuser so each LED illuminates its own row evenly. Use small black baffles between rows to prevent crosstalk.
3. Wire each LED: `+5 V → LED → resistor (~47–100 Ω) → MOSFET drain`; MOSFET source to GND; gate to the Arduino PWM pin through 100 Ω with a 10 kΩ pulldown.
4. Choose resistors for ~20–50 mA per LED (check the datasheet for forward voltage at your LED).

### 3.4 Mount photodiodes and amplifiers
1. Place **4 photodiodes in a horizontal row** at the output side, one per pixel column, each with a **small lens or aperture** that gathers light from its entire column only.
2. Add a baffle between columns to prevent crosstalk.
3. Wire each as a **transimpedance amplifier**: photodiode reverse-biased (or zero-bias for low noise) into the op-amp's inverting input, feedback resistor 1–10 MΩ in parallel with 10–47 pF. Output to A0–A3.
4. Add a 100 nF capacitor on the TIA supply.

### 3.5 (Optional) Verification path
Place the **upconversion card** just behind the rear polarizer to see a rough image of transmitted IR (works best with ≥ 940–980 nm sources and bright LEDs). A CMOS sensor behind the card gives you a camera view of the 4×4 pixel array lit up, useful for calibration and debugging.

---

## 4. Secure with a Printed Housing

### 4.1 Design the housing (CAD)
Make a rectangular tunnel with these pockets, in order along the optical axis:

1. LED bay with 4 holes and row baffles
2. Diffuser slot
3. LC cell cradle with cutouts for copper tape to exit
4. Photodiode bay with column baffles
5. Electronics tray (Arduino, shift registers, TIAs) below or behind
6. Optional camera mount

**Design rules:**
- Wall thickness ≥ 2 mm; print in **black** filament, ≥ 30 % infill.
- Light-tight seams: use overlapping lips or add black foam tape.
- Cable slots for the 16 pad wires + COM; strain relief near the glass.
- Make the LC cradle **spring-loaded or foam-padded** (glass breaks).

### 4.2 Print and assemble
1. Print and test-fit all parts. Sand or file as needed.
2. Place components in order, secure with M2/M3 screws or small beads of hot glue (not on optical surfaces).
3. Cover internal surfaces with matte black paint or flocking if you see internal reflections.
4. Close the housing and verify no ambient light reaches the photodiodes: with LEDs off, ADC readings should be near the dark baseline.

---

## 5. Program the Host Controller (Offloading Algorithm)

The host (a PC, running Python) sends matrices and vectors over serial. The Arduino loads weights into the LC mask, drives LEDs with the input vector, reads the photodiodes, and returns the analog result. The PC then post-processes it (calibration, normalization) and compares against the digital result.

### 5.1 Calibrate first (essential)
Because the optics are analog, calibrate:

1. **Dark offset** `d_j`: LEDs off, read A0–A3.
2. **LED response**: for each LED, sweep PWM 0–255 with all pixels open and record photodiode outputs. Fit a curve; LEDs are not linear.
3. **Pixel transmission** `T_ij(level)`: for each pixel, sweep drive duty 0–100 % with one LED on and record the column signal. Build a lookup table mapping desired weight → drive duty.
4. **Crosstalk matrix**: light one LED at a time with all pixels open; any signal on other columns is crosstalk to subtract.

### 5.2 Offloading logic (conceptual flow)

```
Host PC                                  Arduino + Optical Bench
-------                                  -----------------------
 W (4x4 weights, 0..1)  ── "LOADW" ───►  map via LUT → pixel duty → shift registers
 x (4 inputs, 0..1)     ── "RUN x" ───►  set LED PWM, wait for settle, average ADC
 y_raw (4 values)       ◄─ "Y ..." ────  photodiode readings
 y = calibrate(y_raw)   (subtract dark, divide by gain, undo crosstalk)
 compare with W @ x     report error
```

### 5.3 Arduino firmware (reference sketch)

```cpp
// IR Photonic Processor — Arduino Mega reference firmware
// Serial protocol (115200 baud, newline-terminated):
//   W r0c0 r0c1 ... r3c3      -> load 16 weights, integers 0..255
//   X x0 x1 x2 x3             -> run with 4 inputs, integers 0..255; replies "Y a b c d"
//   D                         -> dark-offset measurement, replies "Y a b c d"

const uint8_t DATA = 4, CLK = 5, LATCH = 6, COM = 7;
const uint8_t LED_PIN[4] = {2, 3, 8, 9};      // must be PWM-capable on Mega
const uint8_t ADC_PIN[4] = {A0, A1, A2, A3};

uint8_t weight[16];            // 0..255 desired drive level per pixel
uint16_t pixelState = 0;       // bit n = invert pixel n during this slice

// LC drive: COM square wave, pixel = COM XOR state (done in hardware).
// Gray levels: software-PWM the 'state' bit across N slices per AC cycle.
const uint8_t SLICES = 16;     // gray resolution ~4 bits
volatile uint8_t sliceIdx = 0;
volatile bool comLevel = false;

void shiftOut16(uint16_t v) {
  digitalWrite(LATCH, LOW);
  shiftOut(DATA, CLK, MSBFIRST, v >> 8);
  shiftOut(DATA, CLK, MSBFIRST, v & 0xFF);
  digitalWrite(LATCH, HIGH);
}

void setupTimer() {
  // ~2 kHz tick on Timer1 (CTC mode): each tick = one slice.
  // One full AC period = 2 * SLICES ticks (~60 Hz here, adjust OCR1A to taste).
  noInterrupts();
  TCCR1A = 0; TCCR1B = 0; TCNT1 = 0;
  OCR1A = 999;                         // 16 MHz / 8 / 1000 = 2 kHz
  TCCR1B |= (1 << WGM12) | (1 << CS11);
  TIMSK1 |= (1 << OCIE1A);
  interrupts();
}

ISR(TIMER1_COMPA_vect) {
  // Update pixel states for this slice: pixel inverted if weight > slice threshold.
  uint16_t s = 0;
  uint8_t thresh = (sliceIdx * 255) / SLICES;
  for (uint8_t i = 0; i < 16; i++) if (weight[i] > thresh) s |= (1 << i);
  shiftOut16(s);                        // NOTE: keep this fast; consider SPI for speed
  sliceIdx++;
  if (sliceIdx >= SLICES) { sliceIdx = 0; comLevel = !comLevel; digitalWrite(COM, comLevel); }
}

void setLEDs(const uint8_t x[4]) { for (int i = 0; i < 4; i++) analogWrite(LED_PIN[i], x[i]); }

void readColumns(uint16_t out[4], uint8_t avg = 64) {
  uint32_t acc[4] = {0, 0, 0, 0};
  for (uint8_t n = 0; n < avg; n++) for (int j = 0; j < 4; j++) acc[j] += analogRead(ADC_PIN[j]);
  for (int j = 0; j < 4; j++) out[j] = acc[j] / avg;
}

void setup() {
  Serial.begin(115200);
  pinMode(DATA, OUTPUT); pinMode(CLK, OUTPUT); pinMode(LATCH, OUTPUT); pinMode(COM, OUTPUT);
  for (int i = 0; i < 4; i++) pinMode(LED_PIN[i], OUTPUT);
  setupTimer();
}

void loop() {
  if (!Serial.available()) return;
  char cmd = Serial.read();
  if (cmd == 'W') {
    for (int i = 0; i < 16; i++) weight[i] = Serial.parseInt();
  } else if (cmd == 'X' || cmd == 'D') {
    uint8_t x[4] = {0, 0, 0, 0};
    if (cmd == 'X') for (int i = 0; i < 4; i++) x[i] = Serial.parseInt();
    setLEDs(x);
    delay(150);                         // LC and LED settle time; tune for your cell
    uint16_t y[4]; readColumns(y);
    Serial.print("Y");
    for (int j = 0; j < 4; j++) { Serial.print(' '); Serial.print(y[j]); }
    Serial.println();
    setLEDs((const uint8_t[4]){0, 0, 0, 0});
  }
  while (Serial.available() && Serial.peek() != '\n') Serial.read();
}
```

### 5.4 Host-side offload script (Python)

```python
# host.py  — pip install pyserial numpy
import serial, time, numpy as np

ser = serial.Serial("/dev/ttyACM0", 115200, timeout=2)   # COM3 on Windows
time.sleep(2)

def send(line): ser.write((line + "\n").encode())

def read_y():
    parts = ser.readline().decode().split()
    return np.array(parts[1:], dtype=float)

# --- Calibration (simplified) ---
send("D");  dark = read_y()

# Optical response per LED with all pixels "open" (weight=0 means relaxed/transmitting
# in a normally-white configuration; invert if yours is normally-black).
send("W " + " ".join(["0"] * 16))
gain = np.zeros((4, 4))
for i in range(4):
    x = [0, 0, 0, 0]; x[i] = 255
    send("X " + " ".join(map(str, x)))
    gain[i] = read_y() - dark          # row i of crosstalk/gain matrix
gain_norm = gain.max()

def optical_matvec(W, x):
    """W: 4x4 in [0,1], x: length 4 in [0,1]. Returns optical estimate of x @ W."""
    # Map desired transmission -> drive level (replace with your measured LUT).
    drive = np.clip(((1.0 - W) * 255).round(), 0, 255).astype(int)   # normally-white
    send("W " + " ".join(map(str, drive.flatten())))
    send("X " + " ".join(map(str, (np.clip(x, 0, 1) * 255).round().astype(int))))
    y_raw = read_y() - dark
    return y_raw / gain_norm * 4.0     # rough normalization, refine with real calibration

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    W = rng.random((4, 4)); x = rng.random(4)
    y_opt = optical_matvec(W, x)
    y_dig = x @ W
    print("optical :", np.round(y_opt, 3))
    print("digital :", np.round(y_dig, 3))
    print("rel err :", np.round(np.abs(y_opt - y_dig) / (np.abs(y_dig) + 1e-9), 3))
```

### 5.5 Offloading policy
Use the optical bench as an **accelerator for small matrix-vector multiplies**; keep everything else on the host.

1. Host splits large matrices into 4×4 tiles.
2. For each tile: load weights, stream input chunks, collect partial sums.
3. Host accumulates partial sums digitally and applies nonlinearities (ReLU, etc.).
4. If measured error exceeds a threshold (e.g., > 10 %), re-run calibration or fall back to digital for that tile.

---

## 6. Test, Validate, Iterate

| Test | Pass criterion |
|---|---|
| Dark reading, LEDs off | Stable, low noise (< 1–2 ADC counts of jitter after averaging) |
| Single LED, all pixels open | Signal on all 4 columns, roughly equal |
| Single pixel closed | Matching column drops, others unchanged |
| Linearity sweep of x | Response linear within a few % after LUT correction |
| Random 4×4 W·x trials | Typical error ≲ 5–10 % |

**Common problems**

| Symptom | Likely cause |
|---|---|
| No response when pixels switch | Polarizers don't work at your IR wavelength; LC not aligned; AC drive absent |
| Pixels slowly darken/fail | DC component in drive; verify COM XOR scheme |
| Large crosstalk | Missing baffles; diffuser too thick/too far |
| Drift over minutes | LED heating or LC temperature; warm up for 5–10 min and recalibrate periodically |
| Noisy readings | Ambient light leaks, TIA unshielded, or ADC reference noise; add shielding and average more |
| Camera sees nothing | IR-cut filter still present; use IR card, or try a different camera |

---

## Safety

- **IR LEDs are invisible.** At these powers they are low risk, but never stare into them at close range, and don't substitute lasers or high-power IR emitters without proper eye protection.
- **Laser engraver:** wear wavelength-rated eye protection, use enclosure/ventilation, and never leave it unattended. ITO ablation produces **fine dust and indium fumes**, so use extraction and a P2/N95 or better respirator, and clean up with damp wipes.
- **Chemical etch:** HCl and zinc powder need gloves, goggles, ventilation, and proper neutralization/disposal.
- **Glass edges** are sharp; edge-seal or file them.
- **PDLC option** needs tens of volts AC; use an isolated supply and keep wiring insulated.
- Respect LED current ratings and use heat-sinking if driven continuously.

---

## Where to go next
- Move from 4×4 to 8×8 with row/column multiplexing (passive matrix) and an ADC multiplexer.
- Use a lock-in style modulation: modulate LEDs at a few kHz and demodulate the photodiode signals to reject ambient light.
- Replace the LC mask with a salvaged LCD or DMD for much higher resolution.
- Add a closed-loop calibration routine so the host retunes the LUT automatically.
