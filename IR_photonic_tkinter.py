#!/usr/bin/env python3
"""Variable-cell IR arithmetic bench and 20 x 20 mm printable mask.
Install: python -m pip install pillow
Run: python reusable_ir_bench.py
Cells can be changed in the GUI (1..64); DEFAULT_CELLS sets the startup value.
Modes: full-adder test, optical X+Y, trial-division factoring, and
subset-sum with MINIMUM CELLS (one lane per distinct reachable sum <= T, run in chunks;
spare cells hold majority-vote replicas, or are blocked by the ITO plate).
This is a hybrid intensity/feedback model, not an all-optical solver.
An optional SIMULATED ITO IR-blocking plate (series element over the apertures) cuts
closed-state leakage, rejects ambient IR, and blocks idle lanes; all plate numbers are assumptions.
Programmable shutters, LED routing and camera calibration are required physically.
Mask aperture diameter shrinks automatically if needed; geometry is unvalidated.
"""
import math
import random
from bisect import bisect_right
import csv
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque, Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageTk

DEFAULT_CELLS = 512
MAX_CELLS = 1024
MAX_SUBSET_ELEMENTS = 512
DPI = 2540
SIZE_MM = 20
PX_PER_MM = 100
DIAMETER_MM = .7
SHOT_K = .06  # ambient-IR shot-noise coefficient (simulation)
RANDOM_MAX = 30000  # largest random element produced by Populate
MAX_TARGET = 20_000_000  # planner bitset limit
MAX_EXPOSURES = 1_500_000  # refuse runs estimated longer than this


def grid_shape(cells):
    cols = min(cells, math.ceil(math.sqrt(2 * cells)))
    return cols, math.ceil(cells / cols)


def mm(v):
    return round(v * PX_PER_MM)


def make_mask(cells=DEFAULT_CELLS):
    if not 1 <= cells <= MAX_CELLS:
        raise ValueError(f'Cells must be 1..{MAX_CELLS}.')
    cols, rows = grid_shape(cells)
    pitchx, pitchy = 16 / cols, 16 / rows
    diameter = min(DIAMETER_MM, pitchx / 4.5, pitchy / 4)
    dx, dy = pitchx * .25, pitchy * .16
    image = Image.new('1', (mm(SIZE_MM), mm(SIZE_MM)), 0)
    draw = ImageDraw.Draw(image)
    layout = []
    r = mm(diameter) / 2
    for i in range(cells):
        cx, cy = 2 + (i % cols + .5) * pitchx, 2 + (i // cols + .5) * pitchy
        for label, ox, oy in [('a', -dx, -dy), ('b', 0, -dy), ('carry', dx, -dy),
                              ('ref_A', -dx / 2, dy), ('ref_B', dx / 2, dy)]:
            x, y = mm(cx + ox), mm(cy + oy)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=1)
            layout.append((i + 1, label, cx + ox, cy + oy, diameter))
    for x, y in [(1, 1), (19, 1), (1, 19), (19, 19)]:
        x, y = mm(x), mm(y)
        draw.line((x - mm(.3), y, x + mm(.3), y), fill=1, width=mm(.06))
        draw.line((x, y - mm(.3), x, y + mm(.3)), fill=1, width=mm(.06))
    draw.rectangle((mm(1.6), mm(.9), mm(1.8), mm(1.1)), fill=1)
    meta = dict(cells=cells, cols=cols, rows=rows, diameter=diameter, pitchx=pitchx, pitchy=pitchy)
    return image, layout, meta


def export_mask(folder, cells=DEFAULT_CELLS):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    image, layout, meta = make_mask(cells)
    stem = f'physical_adder_mask_20mm_{cells}cells'
    path = folder / (stem + '.png')
    image.save(path, dpi=(DPI, DPI))
    with (folder / (stem + '.csv')).open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cell', 'aperture', 'x_mm', 'y_mm', 'diameter_mm'])
        w.writerows(layout)
    (folder / (stem + '_notes.txt')).write_text(
        f'20 x 20 mm at {DPI} DPI; 2000 x 2000 pixels.\n'
        f'{cells} cells in {meta["cols"]} columns x {meta["rows"]} rows; last row may be partial.\n'
        f'Aperture diameter: {meta["diameter"]:.6f} mm nominal.\n'
        f'Cell pitch: {meta["pitchx"]:.6f} x {meta["pitchy"]:.6f} mm.\n'
        'White=open; black=opaque. Print actual size, no fit-to-page. Verify scale.\n'
        'Upper holes a,b,carry; lower holes reference A,B.\n'
        'All occupied cells identical; no factors/products/subsets encoded.\n'
        'Subset assignment lives in the shutter pattern, not in the mask.\n'
        'ITO IR-blocking plate is a simulation parameter set only; not part of this mask geometry.\n'
        'Proposed geometry only. Smaller holes require appropriate fabrication and imaging.\n'
        'Requires programmable shutters/SLM, separate LED routing and calibrated camera.\n'
        'Verify mask opacity at operating IR wavelength.\n')
    return path, image, meta


class Optics:
    def __init__(self, loss=1.0, leak=0.0, noise=0.0, seed=None):
        self.loss, self.leak, self.noise = loss, leak, noise
        self.rng = random.Random(seed)
        self.gates = 0

    def _fix(self, p, q):
        self.gates += 1
        p = min(1.0, p * self.loss + self.leak)
        q = min(1.0, q * self.loss + self.leak)
        return (p, q)

    def AND(self, *xs):
        p = 1.0
        for x in xs:
            p *= x[0]
        q = sum(x[1] for x in xs)
        return self._fix(p, min(1.0, q))

    def OR(self, *xs):
        q = 1.0
        for x in xs:
            q *= x[1]
        p = sum(x[0] for x in xs)
        return self._fix(min(1.0, p), q)

    @staticmethod
    def NOT(x):
        return (x[1], x[0])

    def XOR(self, x, y):
        return self.OR(self.AND(x, self.NOT(y)), self.AND(self.NOT(x), y))

    def XNOR(self, x, y):
        return self.NOT(self.XOR(x, y))

    def full_add(self, a, b, c):
        t = self.XOR(a, b)
        return self.XOR(t, c), self.OR(self.AND(a, b), self.AND(c, t))


ONE, ZERO = (1.0, 0.0), (0.0, 1.0)


def bits_of(v, w):
    return [ONE if (v >> k) & 1 else ZERO for k in range(w)]


def multiply(o, a, b):
    W = len(a) + len(b)
    acc = [ZERO] * W
    for i, ai in enumerate(a):
        row = [o.AND(ai, bj) for bj in b]
        c = ZERO
        for k in range(i, i + len(b) + 1):
            y = row[k - i] if k - i < len(row) else ZERO
            acc[k], c = o.full_add(acc[k], y, c)
    return acc


class ITOPlate:
    """Simulated ITO IR-blocking plate in series with every aperture.

    extinction_db : attenuation of a blocked aperture (adds to the shutter's own leak)
    loss          : fractional insertion loss of an open aperture
    reject_db     : rejection of ambient/stray IR entering through the plate
    tau_us        : electro-optic response time constant
    settle_us     : wait before each exposure; unsettled apertures keep a residual of
                    the previous state, exp(-settle/tau)
    Idle lanes (chunk padding) are driven blocked, so they add no stray light.
    All numbers are assumptions; real ITO gating is weaker and slower than a shutter.
    """
    def __init__(self, enabled=False, extinction_db=20., loss=.08, reject_db=15., tau_us=50., settle_us=250.):
        self.enabled = bool(enabled)
        self.extinction_db, self.loss = extinction_db, loss
        self.reject_db, self.tau_us, self.settle_us = reject_db, tau_us, settle_us

    @property
    def p_on(self):
        return 1.0 - self.loss

    @property
    def p_off(self):
        return 10 ** (-self.extinction_db / 10)

    @property
    def ambient_factor(self):
        return 10 ** (-self.reject_db / 10)

    @property
    def decay(self):
        return math.exp(-self.settle_us / self.tau_us) if self.tau_us > 0 else 0.0

    def step(self, cur, bit):
        target = self.p_on if bit else self.p_off
        return target + (cur - target) * self.decay


class CameraBench:
    def __init__(self, cells=DEFAULT_CELLS, exposure=.22, leak=.001, noise=.002, seed=1, led_a=True, led_b=True,
                 ito=None, ambient=.04, xtalk=.02):
        self.cells = cells
        self.exposure, self.leak, self.noise = exposure, leak, noise
        self.rng = random.Random(seed)
        self.led_a, self.led_b = led_a, led_b
        self.ito = ito if ito is not None else ITOPlate(False)
        self.ambient, self.xtalk = ambient, xtalk
        self.itx = [[self.ito.p_off] * 5 for _ in range(cells)] if self.ito.enabled else None
        self.pbits = [[0] * 5 for _ in range(cells)] if self.ito.enabled else None
        self.exposures = 0
        self.last = None
        self.worst_margin = 0.0
        self.err_sq, self.err_n = 0.0, 0
        self.plate_time_us = 0.0
        self.slots_useful = self.slots_replica = self.slots_idle = 0
        self.corrected = 0  # lanes fixed by replica majority vote

    @property
    def rms_error(self):
        return math.sqrt(self.err_sq / self.err_n) if self.err_n else 0.0

    def capture(self, inputs, active=None, replicas=0):
        n = self.cells
        if len(inputs) != n:
            raise ValueError(f'Exactly {self.cells} input cells required.')
        act = [True] * n if active is None else list(active)
        if len(act) != n:
            raise ValueError(f'Exactly {n} active flags required.')
        n_act = sum(act)
        self.slots_useful += n_act - replicas
        self.slots_replica += replicas
        self.slots_idle += n - n_act
        changed = False  # does any plate aperture need to switch this exposure?
        ito = self.ito
        use = ito.enabled
        ambient = self.ambient * (ito.ambient_factor if use else 1.0)
        dark = .01 + ambient
        noise = math.sqrt(self.noise ** 2 + SHOT_K ** 2 * ambient)
        pe = self.exposure
        lit = sum(act) if use else n  # plate blocks idle lanes; without it every lane is lit
        stray = self.xtalk * pe * lit / n
        p_on = ito.p_on if use else 1.0
        closed = self.leak * (ito.p_off if use else 1.0)  # closed-aperture transmission
        lam = closed / p_on                                # closed/open ratio used by the decoder
        A, B = float(self.led_a), float(self.led_b)
        t = lambda x: self.leak + (1 - self.leak) * x

        def pixel(v):
            return max(0., min(1., v + self.rng.gauss(0, noise)))
        rows = []
        for i, (a, b, c) in enumerate(inputs):
            if any(x not in (0, 1) for x in (a, b, c)):
                raise ValueError('Binary inputs required.')
            if use:
                idle = not act[i]
                bits = (0, 0, 0, 0, 0) if idle else (a, b, c, 1, 1)
                st = self.itx[i]
                if list(bits) != self.pbits[i]:
                    changed = True
                    self.pbits[i][:] = bits
                for j in range(5):
                    st[j] = ito.step(st[j], bits[j])
                pa, pb, pc, pra, prb = st
            else:
                pa = pb = pc = pra = prb = 1.0
            rows.append(dict(signal=pixel(dark + stray + pe * (A * (t(a) * pa + t(b) * pb) + B * t(c) * pc)),
                             black=pixel(dark), refa=pixel(dark + stray + pe * A * pra),
                             refb=pixel(dark + stray + pe * B * prb),
                             baseline=pixel(dark + stray + pe * closed * (2 * A + B))))
        self.exposures += 1
        if use and changed:  # settle wait only when some aperture actually switched
            self.plate_time_us += ito.settle_us
        decoded = []
        for i, row in enumerate(rows):
            if use and not act[i]:
                decoded.append(dict(level=0.0, k=0, sum=0, carry=0))
                continue
            Ar, Br = row['refa'] - row['black'], row['refb'] - row['black']
            if min(Ar, Br) < .04 or abs(Ar - Br) > .25 * max(Ar, Br):
                raise RuntimeError('Insufficient/unequal LED references.')
            level = (row['signal'] - row['baseline']) / ((Ar + Br) / 2 * (1 - lam))
            k = round(level)
            self.worst_margin = max(self.worst_margin, abs(level - k))
            self.err_sq += (level - k) ** 2
            self.err_n += 1
            if row['signal'] > .98 or not 0 <= k <= 3 or abs(level - k) > .35:
                raise RuntimeError('Ambiguous/saturated camera level. Adjust exposure/noise/plate.')
            decoded.append(dict(level=level, k=k, sum=k % 2, carry=int(k >= 2)))
        self.last = dict(inputs=list(inputs), rows=rows, decoded=decoded)
        return decoded


def ito_ab_test(base, plate_kw, cells=64, exposures=2000, seed=7):
    """Same random full-adder exposures with and without the ITO plate.
    A quarter of the lanes are idle (like chunk padding)."""
    out = {}
    for label, enabled in (('without plate', False), ('with plate', True)):
        bench = CameraBench(cells, ito=ITOPlate(enabled, **plate_kw), **base)
        rng = random.Random(seed)
        active = [i < cells - cells // 4 for i in range(cells)]
        aborted = wrong = 0
        for _ in range(exposures):
            inputs = [(rng.getrandbits(1), rng.getrandbits(1), rng.getrandbits(1)) if active[i] else (0, 0, 0)
                      for i in range(cells)]
            try:
                dec = bench.capture(inputs, active)
            except RuntimeError:
                aborted += 1
                continue
            wrong += sum(1 for i in range(cells) if active[i] and dec[i]['k'] != sum(inputs[i]))
        out[label] = dict(aborted=aborted, wrong=wrong, worst=bench.worst_margin, rms=bench.rms_error,
                          plate_ms=bench.plate_time_us / 1000)
    return out


def optical_add(bench, x, y, width, active=None, replicas=0):
    n = bench.cells
    if len(x) != n or len(y) != n:
        raise ValueError(f'{n} lanes required.')
    carry, result = [0] * n, [0] * n
    for k in range(width):
        inputs = [((x[i] >> k) & 1, (y[i] >> k) & 1, carry[i]) for i in range(n)]
        ds = bench.capture(inputs, active, replicas)
        for i, d in enumerate(ds):
            result[i] |= d['sum'] << k
            carry[i] = d['carry']
        yield dict(bit=k, width=width)
    if any(carry):
        raise RuntimeError('Arithmetic overflow.')
    return result


def factor_job(bench, N, report):
    n = bench.cells
    wa = max(2, (n + 1) // 2)
    wb = max(2, n - 1)
    W = wa + wb

    amax, bmax = (1 << wa) - 1, (1 << wb) - 1
    report(f'Dynamic bit allocation: wa={wa} bits, wb={wb} bits (Total W={W} bits, Cells={n})')

    candidates = list(range(2, min(amax + 1, math.isqrt(N) + 1)))
    if not candidates:
        candidates = [2]

    for start in range(0, len(candidates), n):
        batch = candidates[start:start + n]
        active = [i < len(batch) for i in range(n)]
        divisors = batch + [0] * (n - len(batch))
        sums, counts = [0] * n, [0] * n
        report(f'Electronic scheduler: trial divisors {batch}')
        while any(active):
            increments = [divisors[i] if active[i] else 0 for i in range(n)]
            sums = yield from optical_add(bench, sums, increments, W, list(active))  # finished/idle lanes get plate-blocked
            hits = []
            for i in range(n):
                if not active[i]:
                    continue
                counts[i] += 1
                if sums[i] == N:
                    hits.append((divisors[i], counts[i]))
                    active[i] = False
                elif sums[i] > N:
                    active[i] = False
            if hits:
                return min(hits)
    return None


def truth_job(bench, report):
    n = bench.cells
    tested = 0
    for start in range(0, 8, n):
        ids = list(range(start, min(8, start + n)))
        inputs = [((i >> 2) & 1, (i >> 1) & 1, i & 1) for i in ids]
        inputs += [(0, 0, 0)] * (n - len(inputs))
        decoded = bench.capture(inputs, [j < len(ids) for j in range(n)])  # unused lanes plate-blocked
        yield dict(bit=0, width=1)
        for j, i in enumerate(ids):
            if decoded[j]['k'] != sum(inputs[j]):
                raise RuntimeError(f'Truth table failure for {inputs[j]}')
            tested += 1
    return f'All {tested} full-adder combinations passed in {bench.exposures} exposure(s).'


def popcount(v):
    return v.bit_count() if hasattr(v, 'bit_count') else bin(v).count('1')


def plan_subset(S, T, copies=1):
    """CPU planner: how many lanes does each stage need, and how many cells?

    Lane = one distinct reachable sum r <= T - s that must be fed into an adder.
    Cells = peak lanes x copies (capped at MAX_CELLS). Each lane is run on
    `copies` cells so the camera results can be majority-voted; stages that
    don't fill the cells reuse the spare cells as extra copies. Larger stages
    run in several chunks. Uses a big-int bitset.
    """
    if T > MAX_TARGET:
        raise ValueError(f'Target T too large for the planner (max {MAX_TARGET:,}).')
    W = max(1, T.bit_length())
    reach = 1  # bit v set <=> sum v reachable; start with {0}
    lanes = []
    for s in S:
        if (reach >> T) & 1:
            break
        if s == 0 or s > T:
            continue
        feed = reach & ((1 << (T - s + 1)) - 1)
        lanes.append(popcount(feed))
        reach |= feed << s
    peak = max(lanes, default=1)
    n = max(1, min(peak * copies, MAX_CELLS))
    per = max(1, n // copies)  # distinct lanes per chunk
    groups = sum(-(-l // per) for l in lanes)
    return dict(W=W, n=n, peak=peak, lanes=lanes, per=per, copies=copies, exposures=groups * W,
                solvable=bool((reach >> T) & 1))


def subset_sum_job(bench, S, T, report, copies=1, redundant=True):
    """Minimum-cell subset sum that uses every cell and the ITO plate.

    State = set of distinct reachable sums (each a lane). For every element s,
    the CPU feeds the lanes with r <= T - s into the optical adder (r + s),
    decodes the camera result, and keeps sums not seen before. Lanes beyond
    the chunk size run in further chunks. Spare cells either hold replicas of
    the lanes (redundant=True: results are majority-voted) or are idle and
    blocked by the ITO plate (redundant=False). Stops as soon as T is reached.
    Returns the list of chosen element indices, or None.
    """
    n = bench.cells
    per = max(1, n // copies)
    W = max(1, T.bit_length())  # r + s <= T < 2**W: overflow impossible
    parent = {0: -1}  # sum -> index of the element that first produced it
    order = [0]       # sorted reachable sums
    report(f'Minimum-cell subset sum: {n} cells, {per} lanes/chunk, W={W} bits, {len(S)} elements, '
           f'spare cells: {"vote replicas" if redundant else "plate-blocked"}')
    for j, s in enumerate(S):
        if T in parent:
            break
        if s == 0 or s > T:
            continue
        feed = order[:bisect_right(order, T - s)]
        new = []
        for start in range(0, len(feed), per):
            chunk = feed[start:start + per]
            L = len(chunk)
            src = [i % L for i in range(n)] if redundant else list(range(L)) + [None] * (n - L)
            x = [chunk[k] if k is not None else 0 for k in src]
            y = [s if k is not None else 0 for k in src]
            labels = [(f'{chunk[k]}+{s}' + (' ↻' if i >= L else '')) if k is not None else ''
                      for i, k in enumerate(src)]
            act = [k is not None for k in src]
            gen = optical_add(bench, x, y, W, act, (n - L) if redundant else 0)
            while True:
                try:
                    e = next(gen)
                except StopIteration as stop:
                    res = stop.value
                    break
                e['stage'] = (f'element {j + 1}/{len(S)} (s={s}), '
                              f'lanes {start + 1}-{start + L} of {len(feed)}, {n // L if redundant else 1} cop.')
                e['labels'] = labels
                yield e
            for i, r in enumerate(chunk):
                vals = [res[q] for q in range(i, n, L)] if redundant else [res[i]]
                if len(vals) > 1:
                    top, cnt = Counter(vals).most_common(1)[0]
                    if 2 * cnt <= len(vals):
                        raise RuntimeError(f'Replica disagreement for {r}+{s}: {vals}')
                    if cnt < len(vals):
                        bench.corrected += 1
                    v = top
                else:
                    v = vals[0]
                if v != r + s:
                    raise RuntimeError(f'Optical/CPU mismatch: {r}+{s} decoded as {v}')
                if v not in parent:
                    parent[v] = j
                    new.append(v)
            if T in parent:
                break
        if new:
            order = sorted(order + new)
        report(f'Element {j + 1}/{len(S)} s={s}: {len(feed)} lanes, {len(new)} new sums, {len(parent)} reachable')
    if T not in parent:
        return None
    idx, v = [], T
    while v:
        k = parent[v]
        idx.append(k)
        v -= S[k]
    idx.sort()
    if sum(S[k] for k in idx) != T:
        raise RuntimeError('Reconstruction check failed.')
    return idx


def parse_set(text):
    try:
        S = [int(v) for v in text.replace(';', ',').replace(' ', ',').split(',') if v.strip()]
    except ValueError:
        raise ValueError('Set S must be comma-separated integers, e.g. 3,5,7,11.')
    if not 1 <= len(S) <= MAX_SUBSET_ELEMENTS:
        raise ValueError(f'Set S needs 1..{MAX_SUBSET_ELEMENTS} elements.')
    if any(v < 0 for v in S):
        raise ValueError('Set S elements must be >= 0.')
    return S


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Variable-cell IR bench + 20 mm mask PNG')
        self.geometry('1300x880')
        self.vars = {k: tk.StringVar(value=str(v)) for k, v in dict(
            cells=DEFAULT_CELLS, N=143, x=11, y=13, S='3,5,7,11', T=15, count=32, copies=1,
            exposure=.22, leak=.001, noise=.002, seed=1, delay=5,
            ambient=.04, ito_er=20, ito_loss=8, ito_rej=15, ito_tau=50, ito_settle=250).items()}

        self.vars['cells'].trace_add('write', self.on_cells_changed)

        self.A, self.B = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        self.ITO = tk.BooleanVar(value=True)
        self.REPL = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value='Set Cells, then run. Mask and arithmetic use the same count.')
        self.job = None
        self.last_camera = None
        self.display_cells = DEFAULT_CELLS
        self.cell_labels = []
        self.mask_window = None
        self.log_lines = deque(maxlen=250)

        top = ttk.Frame(self, padding=8)
        top.pack(fill='x')
        self.entries = []
        labels = [('cells', 'Cells (1–1024)'), ('N', 'Factor N'), ('x', 'Adder X'), ('y', 'Adder Y'),
                  ('S', 'Subset set S'), ('T', 'Target T'), ('count', 'Count (1–32)'), ('copies', 'Copies (1–9)'), ('exposure', 'Exposure'), ('leak', 'Leak'),
                  ('noise', 'Noise σ'), ('seed', 'Seed'), ('delay', 'Delay ms')]
        for j, (key, label) in enumerate(labels):
            ttk.Label(top, text=label).grid(row=0, column=j, padx=3)
            if key == 'cells':
                entry = ttk.Spinbox(top, from_=1, to=MAX_CELLS, textvariable=self.vars[key], width=10,
                                    command=self.on_spinbox_changed)
                entry.bind('<KeyRelease>', self.on_cells_key_release)
            else:
                entry = ttk.Entry(top, textvariable=self.vars[key], width=16 if key == 'S' else 10)
            entry.grid(row=1, column=j, padx=3)
            self.entries.append(entry)

        ito_bar = ttk.Frame(self, padding=(8, 0))
        ito_bar.pack(fill='x')
        ttk.Checkbutton(ito_bar, text='ITO IR-blocking plate (simulated, next run)', variable=self.ITO).grid(row=1, column=0, padx=3)
        for j, (key, label) in enumerate([('ambient', 'Ambient IR'), ('ito_er', 'Block ER dB'), ('ito_loss', 'Open loss %'),
                                          ('ito_rej', 'Ambient reject dB'), ('ito_tau', 'Plate τ µs'),
                                          ('ito_settle', 'Settle µs')], start=1):
            ttk.Label(ito_bar, text=label).grid(row=0, column=j, padx=3)
            entry = ttk.Entry(ito_bar, textvariable=self.vars[key], width=10)
            entry.grid(row=1, column=j, padx=3)
            self.entries.append(entry)
        ttk.Checkbutton(ito_bar, text='Spare cells = vote replicas (off: plate-blocked)', variable=self.REPL).grid(row=1, column=7, padx=8)

        bar = ttk.Frame(self, padding=8)
        bar.pack(fill='x')
        self.buttons = []
        for label, command in [('Full-adder test', lambda: self.start('truth')),
                               ('Optical X + Y', lambda: self.start('add')),
                               ('Factor N (hybrid)', lambda: self.start('factor')),
                               ('Populate random S', self.populate),
                               ('Subset sum (min cells)', lambda: self.start('subset')),
                               ('ITO A/B test', self.ito_ab),
                               ('Export/show mask PNG', self.show_mask)]:
            b = ttk.Button(bar, text=label, command=command)
            b.pack(side='left', padx=4)
            self.buttons.append(b)
        ttk.Button(bar, text='Stop', command=self.stop).pack(side='left', padx=4)
        ttk.Checkbutton(bar, text='LED A (next run)', variable=self.A).pack(side='left', padx=5)
        ttk.Checkbutton(bar, text='LED B (next run)', variable=self.B).pack(side='left')
        ttk.Label(self, textvariable=self.status, padding=8, wraplength=1250).pack(fill='x')

        frame = ttk.Frame(self)
        frame.pack(fill='both', expand=True, padx=8)
        self.canvas = tk.Canvas(frame, bg='#101827', highlightthickness=0)
        scroll = ttk.Scrollbar(frame, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda e: self.draw())

        self.log = tk.Text(self, height=8, wrap='word', font=('TkFixedFont', 10))
        self.log.pack(fill='x', padx=8, pady=8)
        ttk.Label(self, text='Fixed mask + programmable shutters + camera intensity readout. CPU supplies decoding, carry feedback, trial search and final comparison.\n'
                             '20 × 20 mm mask is invariant across problems; aperture size adapts to cell count. Subset sum: lanes = distinct reachable sums ≤ T on the minimum cell count. No all-optical solving claim.',
                  padding=8).pack(fill='x')
        self.say('Fewer than 8 cells: truth test uses multiple exposures. More than 8: unused test lanes receive zero inputs.')
        self.say('Subset sum auto-sets Cells to the minimum needed (peak distinct reachable sums, capped at MAX_CELLS).')
        self.draw()

    def on_spinbox_changed(self):
        self.on_cells_changed()

    def on_cells_key_release(self, event):
        self.on_cells_changed()

    def on_cells_changed(self, *args):
        try:
            n = self.cell_count()
            if 1 <= n <= MAX_CELLS:
                self.display_cells = n
                self.last_camera = None
                self.cell_labels = []

                wa = max(2, (n + 1) // 2)
                wb = max(2, n - 1)
                W = wa + wb
                self.status.set(f'Cells updated to {n} | Dynamic Bit Widths: wa={wa}, wb={wb}, total W={W}')

                self.draw()
                if self.mask_window is not None and self.mask_window.winfo_exists():
                    self.show_mask(n)
        except ValueError:
            pass

    def populate(self):
        try:
            k = int(self.vars['count'].get())
            if not 1 <= k <= MAX_SUBSET_ELEMENTS:
                raise ValueError
        except ValueError:
            messagebox.showerror('Populate', f'Count must be an integer 1..{MAX_SUBSET_ELEMENTS}.')
            return
        rng = random.Random()
        S = [rng.randint(1, RANDOM_MAX) for _ in range(k)]
        pick = [v for v in S if rng.random() < .5] or [S[0]]  # guarantees a solvable T
        T = sum(pick)
        self.vars['S'].set(','.join(map(str, S)))
        self.vars['T'].set(str(T))
        shown = S if k <= 12 else f'{S[:8]}… ({k} numbers)'
        note = f'Populated S={shown}, T={T} (sum of a random subset, so a solution exists).'
        try:
            try:
                copies = max(1, min(9, int(self.vars['copies'].get())))
            except ValueError:
                copies = 1
            plan = plan_subset(S, T, copies)
            self.vars['cells'].set(str(plan['n']))  # minimum cells; keeps mask preview in sync
            note += f' Minimum cells: {plan["n"]} (peak {plan["peak"]} lanes), ~{plan["exposures"]:,} exposures.'
            if plan['exposures'] > MAX_EXPOSURES:
                note += f' Too large to run (limit {MAX_EXPOSURES:,}): lower Count or RANDOM_MAX.'
        except ValueError as e:
            note += f' {e}'
        self.say(note)

    def ito_ab(self):
        if self.job is not None:
            return
        try:
            self.params('add')  # validates fields and fills base_kw / plate_kw
        except ValueError as e:
            messagebox.showerror('Input', str(e))
            return
        res = ito_ab_test(self.base_kw, self.plate_kw)
        for label, r in res.items():
            self.say(f'ITO A/B {label}: {r["aborted"]} aborted, {r["wrong"]} wrong decodes, '
                     f'worst margin {r["worst"]:.2f}, rms level error {r["rms"]:.3f}, plate time {r["plate_ms"]:.0f} ms')
        w, p = res['without plate'], res['with plate']
        self.status.set(f'ITO A/B (64 lanes, 2000 exposures, 25% idle): rms error {w["rms"]:.3f} → {p["rms"]:.3f}; '
                        f'aborted {w["aborted"]} → {p["aborted"]}; wrong {w["wrong"]} → {p["wrong"]}')

    def say(self, s):
        self.log_lines.append(s)
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.insert('end', '\n'.join(self.log_lines))
        self.log.see('end')
        self.log.configure(state='disabled')

    def cell_count(self):
        val = self.vars['cells'].get().strip()
        n = int(val) if val.isdigit() else DEFAULT_CELLS
        if not 1 <= n <= MAX_CELLS:
            raise ValueError(f'Cells must be 1..{MAX_CELLS}.')
        return n

    def params(self, mode):
        n = self.cell_count()
        N = x = y = 0
        S, T = [], 0
        if mode == 'subset':
            S = parse_set(self.vars['S'].get())
            T = int(self.vars['T'].get())
            if T < 0:
                raise ValueError('Target T must be >= 0.')
            copies = int(self.vars['copies'].get())
            if not 1 <= copies <= 9:
                raise ValueError('Copies must be 1..9.')
            self.copies = copies
            self.plan = plan_subset(S, T, copies)
            if self.plan['exposures'] > MAX_EXPOSURES:
                raise ValueError(f'Needs ~{self.plan["exposures"]:,} exposures (limit {MAX_EXPOSURES:,}). '
                                 'Lower Count/random max or use a smaller target.')
            n = self.plan['n']  # minimum cells
        else:
            N, x, y = [int(self.vars[k].get()) for k in ('N', 'x', 'y')]
            if not 4 <= N or not 0 <= x or not 0 <= y:
                raise ValueError('N must be >= 4; X/Y must be >= 0.')
        p, l, z = [float(self.vars[k].get()) for k in ('exposure', 'leak', 'noise')]
        if not all(math.isfinite(v) for v in (p, l, z)) or not (0 <= p <= 1 and 0 <= l < .5 and 0 <= z <= 1):
            raise ValueError('Exposure 0–1, leak 0–<0.5, noise 0–1.')
        delay = int(self.vars['delay'].get())
        if not 0 <= delay <= 1000:
            raise ValueError('Delay: 0–1000 ms.')
        amb, er, loss, rej, tau, settle = [float(self.vars[k].get()) for k in
                                           ('ambient', 'ito_er', 'ito_loss', 'ito_rej', 'ito_tau', 'ito_settle')]
        if not all(math.isfinite(v) for v in (amb, er, loss, rej, tau, settle)):
            raise ValueError('ITO/ambient values must be finite numbers.')
        if not (0 <= amb <= .5 and 0 <= er <= 60 and 0 <= loss <= 50 and 0 <= rej <= 60 and tau >= 0 and settle >= 0):
            raise ValueError('Ambient 0–0.5, ER 0–60 dB, loss 0–50 %, reject 0–60 dB, τ/settle ≥ 0.')
        self.plate_kw = dict(extinction_db=er, loss=loss / 100, reject_db=rej, tau_us=tau, settle_us=settle)
        self.base_kw = dict(exposure=p, leak=l, noise=z, seed=int(self.vars['seed'].get()),
                            led_a=self.A.get(), led_b=self.B.get(), ambient=amb)
        bench = CameraBench(n, ito=ITOPlate(self.ITO.get(), **self.plate_kw), **self.base_kw)
        return N, x, y, S, T, delay, bench

    def start(self, mode):
        if self.job is not None:
            return
        try:
            self.N, self.x, self.y, self.S, self.T, self.delay, self.bench = self.params(mode)
        except ValueError as e:
            messagebox.showerror('Input', str(e))
            return
        self.mode = mode
        self.display_cells = self.bench.cells
        pl = self.bench.ito
        if pl.enabled:
            self.say(f'ITO plate on: block {pl.extinction_db:.0f} dB, open loss {pl.loss * 100:.0f}%, ambient reject {pl.reject_db:.0f} dB, '
                     f'tau {pl.tau_us:.0f} us, settle {pl.settle_us:.0f} us (only waited when apertures switch)')
        self.last_camera = None
        self.cell_labels = []

        if mode == 'subset':
            # Sync the Cells box (and open mask window) to the minimum cell count.
            if self.vars['cells'].get().strip() != str(self.bench.cells):
                self.vars['cells'].set(str(self.bench.cells))
            self.display_cells = self.bench.cells
            self.last_camera = None
            self.cell_labels = []
            self.say(f'Plan: {self.plan["n"]} cells (peak {self.plan["peak"]} lanes), '
                     f'W={self.plan["W"]}, ~{self.plan["exposures"]:,} exposures max.')

        for e in self.entries + self.buttons:
            e.configure(state='disabled')

        n = self.bench.cells
        wa = max(2, (n + 1) // 2)
        wb = max(2, n - 1)
        W = wa + wb

        if mode == 'factor':
            self.job = factor_job(self.bench, self.N, self.say)
        elif mode == 'add':
            self.job = optical_add(self.bench, [self.x] * n, [self.y] * n, max(W, self.N.bit_length(), (self.x + self.y).bit_length() + 1))
        elif mode == 'subset':
            self.job = subset_sum_job(self.bench, self.S, self.T, self.say, self.copies, self.REPL.get())
        else:
            self.job = truth_job(self.bench, self.say)
        self.after(1, self.tick)

    def tick(self):
        if self.job is None:
            return
        try:
            e = next(self.job)
            self.last_camera = self.bench.last
            if 'labels' in e:
                self.cell_labels = e['labels']
            stage = f' | {e["stage"]}' if 'stage' in e else ''
            self.status.set(f'{self.mode}: {self.bench.cells} cells | exposure {self.bench.exposures} | bit {e["bit"] + 1}/{e["width"]} | margin {self.bench.worst_margin:.2f}{stage}')
            self.draw()
            self.after(self.delay, self.tick)
        except StopIteration as e:
            r = e.value
            if self.mode == 'factor':
                msg = f'Camera-derived candidate: {self.N} = {r[0]} × {r[1]}' if r else 'No divisor detected; a noisy search is not a primality certificate.'
            elif self.mode == 'add':
                top, cnt = Counter(r).most_common(1)[0]  # every cell computes X+Y; majority vote
                msg = f'Camera-derived sum: {self.x} + {self.y} → {top}; {cnt}/{len(r)} lanes agree'
            elif self.mode == 'subset':
                if r is not None:
                    vals = [self.S[k] for k in r]
                    shown = vals if len(vals) <= 12 else f'{vals[:12]}… ({len(vals)} elements)'
                    msg = f'Subset sum: {shown} totals {self.T} (one solution; indices {r[:12]}{"…" if len(r) > 12 else ""})'
                else:
                    msg = f'Subset sum: no subset totals {self.T}'
            else:
                msg = r
            self.status.set(msg + f' | {self.bench.exposures} exposures')
            self.say(self.status.get())
            self.say(f'Decode: worst margin {self.bench.worst_margin:.2f}, rms level error {self.bench.rms_error:.3f}; '
                     f'ITO {"on" if self.bench.ito.enabled else "off"}, plate settle time {self.bench.plate_time_us / 1000:.1f} ms')
            bn = self.bench
            tot = bn.slots_useful + bn.slots_replica + bn.slots_idle
            if tot:
                self.say(f'Cell use: {100 * bn.slots_useful / tot:.0f}% useful, {100 * bn.slots_replica / tot:.0f}% vote replicas '
                         f'({bn.corrected} lanes corrected), {100 * bn.slots_idle / tot:.0f}% idle '
                         f'({"plate-blocked" if bn.ito.enabled else "lit but unused"})')
            self.finish()
            self.show_mask(self.bench.cells)
        except Exception as e:
            self.status.set('Stopped: ' + str(e))
            self.say(self.status.get())
            self.finish()

    def finish(self):
        self.job = None
        for e in self.entries + self.buttons:
            e.configure(state='normal')

    def stop(self):
        if self.job is not None:
            self.job.close()
            self.finish()
            self.status.set('Stopped. No completed-run export.')

    def show_mask(self, cells=None):
        try:
            n = self.cell_count() if not isinstance(cells, int) else cells
            path, image, meta = export_mask(Path(__file__).resolve().parent / 'mask', n)
            self.say(f'Mask: {n} cells, diameter {meta["diameter"]:.3f} mm | {path}')
            if self.mask_window is not None and self.mask_window.winfo_exists():
                self.mask_window.destroy()
            win = tk.Toplevel(self)
            self.mask_window = win
            win.title('20 × 20 mm physical mask PNG')
            preview = image.convert('RGB').resize((600, 600), Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)
            photo = ImageTk.PhotoImage(preview)
            label = ttk.Label(win, image=photo)
            label.image = photo
            label.pack(padx=8, pady=8)
            ttk.Label(win, text=f'{n} cells | {meta["cols"]} columns × {meta["rows"]} rows | aperture {meta["diameter"]:.3f} mm\n20 × 20 mm at 2540 DPI. White=open. Print actual size.', padding=8).pack()
        except Exception as e:
            messagebox.showerror('Mask export', str(e))

    def draw(self):
        cv = self.canvas
        cv.delete('all')
        w = max(450, cv.winfo_width())
        n = self.display_cells
        cols = min(n, max(1, int(w / 260)))
        rows = math.ceil(n / cols)
        cw, ch = (w - 24) / cols, 210
        cv.configure(scrollregion=(0, 0, w, rows * ch + 40))
        for i in range(n):
            x, y = 12 + (i % cols) * cw, 12 + (i // cols) * ch
            cv.create_rectangle(x, y, x + cw - 10, y + ch - 10, fill='#182338', outline='#475569')
            title = f'Cell {i + 1}'
            if i < len(self.cell_labels) and self.cell_labels[i]:
                title += f'  {self.cell_labels[i]}'
            cv.create_text(x + 12, y + 18, text=title, fill='white', anchor='w')
            data = self.last_camera
            inp = data['inputs'][i] if data and i < len(data['inputs']) else (0, 0, 0)
            for j, label in enumerate(('a / A', 'b / A', 'carry / B')):
                xx = x + 20 + j * (cw - 50) / 3
                cv.create_oval(xx, y + 36, xx + 18, y + 54, fill='#fbbf24' if inp[j] else '#080d19', outline='#94a3b8')
                cv.create_text(xx, y + 68, text=label, fill='#94a3b8', anchor='w', font=('TkDefaultFont', 9))
            if data and i < len(data['rows']):
                r, d = data['rows'][i], data['decoded'][i]
                for j, (label, key) in enumerate((('signal', 'signal'), ('ref A', 'refa'), ('ref B', 'refb'))):
                    yy = y + 94 + j * 23
                    cv.create_text(x + 12, yy, text=label, fill='#94a3b8', anchor='w')
                    left, right = x + 70, x + cw - 58
                    cv.create_rectangle(left, yy - 5, right, yy + 5, fill='#080d19', outline='')
                    cv.create_rectangle(left, yy - 5, left + (right - left) * r[key], yy + 5, fill='#34d399', outline='')
                    cv.create_text(right + 4, yy, text=f'{r[key]:.2f}', fill='white', anchor='w', font=('TkDefaultFont', 9))
                cv.create_text(x + 12, y + 179, text=f'Level {d["level"]:.2f} → sum {d["sum"]} carry {d["carry"]}', fill='#67e8f9', anchor='w', font=('TkDefaultFont', 10))
            else:
                cv.create_text(x + 12, y + 110, text='Ready for exposure', fill='#94a3b8', anchor='w')


if __name__ == '__main__':
    App().mainloop()
