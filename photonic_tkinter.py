#!/usr/bin/env python3
"""Variable-cell IR arithmetic bench and 20 x 20 mm printable mask.
Install: python -m pip install pillow
Run: python reusable_ir_bench.py
Cells can be changed in the GUI (1..64); DEFAULT_CELLS sets the startup value.
This is a hybrid intensity/feedback model, not an all-optical factorer.
Programmable shutters, LED routing and camera calibration are required physically.
Mask aperture diameter shrinks automatically if needed; geometry is unvalidated.
"""
import math
import random
import csv
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
from pathlib import Path
from PIL import Image, ImageDraw, ImageTk

DEFAULT_CELLS = 8
MAX_CELLS = 64
DPI = 2540
SIZE_MM = 20
PX_PER_MM = 100
DIAMETER_MM = .7

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
                             ('ref_A', -dx/2, dy), ('ref_B', dx/2, dy)]:
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
        'All occupied cells identical; no factors/products encoded.\n'
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

class CameraBench:
    def __init__(self, cells=DEFAULT_CELLS, exposure=.22, leak=.001, noise=.002, seed=1, led_a=True, led_b=True):
        self.cells = cells
        self.exposure, self.leak, self.noise = exposure, leak, noise
        self.rng = random.Random(seed)
        self.led_a, self.led_b = led_a, led_b
        self.exposures = 0
        self.last = None

    def capture(self, inputs):
        if len(inputs) != self.cells:
            raise ValueError(f'Exactly {self.cells} input cells required.')
        def pixel(v):
            return max(0., min(1., v + self.rng.gauss(0, self.noise)))
        rows = []
        for a, b, c in inputs:
            if any(x not in (0, 1) for x in (a, b, c)):
                raise ValueError('Binary inputs required.')
            dark, p, A, B = .01, self.exposure, float(self.led_a), float(self.led_b)
            t = lambda x: self.leak + (1 - self.leak) * x
            rows.append(dict(signal=pixel(dark + p * (A * (t(a) + t(b)) + B * t(c))),
                black=pixel(dark), refa=pixel(dark + p * A), refb=pixel(dark + p * B),
                baseline=pixel(dark + p * self.leak * (2 * A + B))))
        self.exposures += 1
        decoded = []
        for row in rows:
            A, B = row['refa'] - row['black'], row['refb'] - row['black']
            if min(A, B) < .04 or abs(A - B) > .25 * max(A, B):
                raise RuntimeError('Insufficient/unequal LED references.')
            level = (row['signal'] - row['baseline']) / ((A + B) / 2 * (1 - self.leak))
            k = round(level)
            if row['signal'] > .98 or not 0 <= k <= 3 or abs(level - k) > .35:
                raise RuntimeError('Ambiguous/saturated camera level. Adjust exposure/noise.')
            decoded.append(dict(level=level, k=k, sum=k % 2, carry=int(k >= 2)))
        self.last = dict(inputs=list(inputs), rows=rows, decoded=decoded)
        return decoded

def optical_add(bench, x, y, width):
    n = bench.cells
    if len(x) != n or len(y) != n:
        raise ValueError(f'{n} lanes required.')
    carry, result = [0] * n, [0] * n
    for k in range(width):
        inputs = [((x[i] >> k) & 1, (y[i] >> k) & 1, carry[i]) for i in range(n)]
        ds = bench.capture(inputs)
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
            sums = yield from optical_add(bench, sums, increments, W)
            hits = []
            for i in range(n):
                if not active[i]: continue
                counts[i] += 1
                if sums[i] == N:
                    hits.append((divisors[i], counts[i])); active[i] = False
                elif sums[i] > N:
                    active[i] = False
            if hits: return min(hits)
    return None

def truth_job(bench, report):
    n = bench.cells
    tested = 0
    for start in range(0, 8, n):
        ids = list(range(start, min(8, start + n)))
        inputs = [((i >> 2) & 1, (i >> 1) & 1, i & 1) for i in ids]
        inputs += [(0, 0, 0)] * (n - len(inputs))
        decoded = bench.capture(inputs)
        yield dict(bit=0, width=1)
        for j, i in enumerate(ids):
            if decoded[j]['k'] != sum(inputs[j]):
                raise RuntimeError(f'Truth table failure for {inputs[j]}')
            tested += 1
    return f'All {tested} full-adder combinations passed in {bench.exposures} exposure(s).'

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Variable-cell IR bench + 20 mm mask PNG')
        self.geometry('1240x880')
        self.vars = {k: tk.StringVar(value=str(v)) for k, v in dict(cells=DEFAULT_CELLS, N=143, x=11, y=13, exposure=.22, leak=.001, noise=.002, seed=1, delay=5).items()}
        
        self.vars['cells'].trace_add('write', self.on_cells_changed)

        self.A, self.B = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        self.status = tk.StringVar(value='Set Cells, then run. Mask and arithmetic use the same count.')
        self.job = None
        self.last_camera = None
        self.display_cells = DEFAULT_CELLS
        self.mask_window = None
        self.log_lines = deque(maxlen=250)
        
        top = ttk.Frame(self, padding=8); top.pack(fill='x')
        self.entries = []
        labels = [('cells', 'Cells (1–64)'), ('N', 'Factor N'), ('x', 'Adder X'), ('y', 'Adder Y'), ('exposure', 'Exposure'), ('leak', 'Leak'), ('noise', 'Noise σ'), ('seed', 'Seed'), ('delay', 'Delay ms')]
        for j, (key, label) in enumerate(labels):
            ttk.Label(top, text=label).grid(row=0, column=j, padx=3)
            if key == 'cells':
                entry = ttk.Spinbox(top, from_=1, to=MAX_CELLS, textvariable=self.vars[key], width=10, command=self.on_spinbox_changed)
                entry.bind('<KeyRelease>', self.on_cells_key_release)
            else:
                entry = ttk.Entry(top, textvariable=self.vars[key], width=10)
            entry.grid(row=1, column=j, padx=3)
            self.entries.append(entry)
            
        bar = ttk.Frame(self, padding=8); bar.pack(fill='x')
        self.buttons = []
        for label, command in [('Full-adder test', lambda: self.start('truth')), ('Optical X + Y', lambda: self.start('add')), ('Factor N (hybrid)', lambda: self.start('factor')), ('Export/show mask PNG', self.show_mask)]:
            b = ttk.Button(bar, text=label, command=command); b.pack(side='left', padx=4); self.buttons.append(b)
        ttk.Button(bar, text='Stop', command=self.stop).pack(side='left', padx=4)
        ttk.Checkbutton(bar, text='LED A (next run)', variable=self.A).pack(side='left', padx=5)
        ttk.Checkbutton(bar, text='LED B (next run)', variable=self.B).pack(side='left')
        ttk.Label(self, textvariable=self.status, padding=8, wraplength=1200).pack(fill='x')
        
        frame = ttk.Frame(self); frame.pack(fill='both', expand=True, padx=8)
        self.canvas = tk.Canvas(frame, bg='#101827', highlightthickness=0)
        scroll = ttk.Scrollbar(frame, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y'); self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda e: self.draw())
        
        self.log = tk.Text(self, height=8, wrap='word', font=('TkFixedFont', 10)); self.log.pack(fill='x', padx=8, pady=8)
        ttk.Label(self, text='Fixed mask + programmable shutters + camera intensity readout. CPU supplies decoding, carry feedback and trial search.\n20 × 20 mm mask is invariant across input numbers; aperture size adapts to cell count. No all-optical factoring claim.', padding=8).pack(fill='x')
        self.say('Fewer than 8 cells: truth test uses multiple exposures. More than 8: unused test lanes receive zero inputs.')
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
                
                wa = max(2, (n + 1) // 2)
                wb = max(2, n - 1)
                W = wa + wb
                self.status.set(f'Cells updated to {n} | Dynamic Bit Widths: wa={wa}, wb={wb}, total W={W}')
                
                self.draw()
                if self.mask_window is not None and self.mask_window.winfo_exists():
                    self.show_mask(n)
        except ValueError:
            pass

    def say(self, s):
        self.log_lines.append(s); self.log.configure(state='normal'); self.log.delete('1.0', 'end')
        self.log.insert('end', '\n'.join(self.log_lines)); self.log.see('end'); self.log.configure(state='disabled')

    def cell_count(self):
        val = self.vars['cells'].get().strip()
        n = int(val) if val.isdigit() else DEFAULT_CELLS
        if not 1 <= n <= MAX_CELLS: raise ValueError(f'Cells must be 1..{MAX_CELLS}.')
        return n

    def params(self):
        n = self.cell_count()
        N, x, y = [int(self.vars[k].get()) for k in ('N', 'x', 'y')]
        # Removed the restrictive N < 256 upper bound to allow numbers > 255
        if not 4 <= N or not 0 <= x or not 0 <= y: raise ValueError('N must be >= 4; X/Y must be >= 0.')
        p, l, z = [float(self.vars[k].get()) for k in ('exposure', 'leak', 'noise')]
        if not all(math.isfinite(v) for v in (p, l, z)) or not (0 <= p <= 1 and 0 <= l < .5 and 0 <= z <= 1): raise ValueError('Exposure 0–1, leak 0–<0.5, noise 0–1.')
        delay = int(self.vars['delay'].get())
        if not 0 <= delay <= 1000: raise ValueError('Delay: 0–1000 ms.')
        return N, x, y, delay, CameraBench(n, p, l, z, int(self.vars['seed'].get()), self.A.get(), self.B.get())

    def start(self, mode):
        if self.job is not None: return
        try:
            self.N, self.x, self.y, self.delay, self.bench = self.params()
        except ValueError as e:
            messagebox.showerror('Input', str(e)); return
        self.mode = mode; self.display_cells = self.bench.cells; self.last_camera = None
        for e in self.entries + self.buttons: e.configure(state='disabled')
        
        n = self.bench.cells
        wa = max(2, (n + 1) // 2)
        wb = max(2, n - 1)
        W = wa + wb

        if mode == 'factor': self.job = factor_job(self.bench, self.N, self.say)
        elif mode == 'add': self.job = optical_add(self.bench, [self.x] * self.bench.cells, [self.y] * self.bench.cells, max(W, self.N.bit_length()))
        else: self.job = truth_job(self.bench, self.say)
        self.after(1, self.tick)

    def tick(self):
        if self.job is None: return
        try:
            e = next(self.job); self.last_camera = self.bench.last
            self.status.set(f'{self.mode}: {self.bench.cells} cells | exposure {self.bench.exposures} | bit {e["bit"] + 1}/{e["width"]}')
            self.draw(); self.after(self.delay, self.tick)
        except StopIteration as e:
            r = e.value
            if self.mode == 'factor': msg = f'Camera-derived candidate: {self.N} = {r[0]} × {r[1]}' if r else 'No divisor detected; a noisy search is not a primality certificate.'
            elif self.mode == 'add': msg = f'Camera-derived sum: {self.x} + {self.y} → {r[0]}; {len(r)} lanes'
            else: msg = r
            self.status.set(msg + f' | {self.bench.exposures} exposures'); self.say(self.status.get())
            self.finish(); self.show_mask(self.bench.cells)
        except Exception as e:
            self.status.set('Stopped: ' + str(e)); self.say(self.status.get()); self.finish()

    def finish(self):
        self.job = None
        for e in self.entries + self.buttons: e.configure(state='normal')

    def stop(self):
        if self.job is not None: self.job.close(); self.finish(); self.status.set('Stopped. No completed-run export.')

    def show_mask(self, cells=None):
        try:
            n = self.cell_count() if cells is None else cells
            path, image, meta = export_mask(Path(__file__).resolve().parent / 'mask', n)
            self.say(f'Mask: {n} cells, diameter {meta["diameter"]:.3f} mm | {path}')
            if self.mask_window is not None and self.mask_window.winfo_exists(): self.mask_window.destroy()
            win = tk.Toplevel(self); self.mask_window = win; win.title('20 × 20 mm physical mask PNG')
            preview = image.convert('RGB').resize((600, 600), Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)
            photo = ImageTk.PhotoImage(preview); label = ttk.Label(win, image=photo); label.image = photo; label.pack(padx=8, pady=8)
            ttk.Label(win, text=f'{n} cells | {meta["cols"]} columns × {meta["rows"]} rows | aperture {meta["diameter"]:.3f} mm\n20 × 20 mm at 2540 DPI. White=open. Print actual size.', padding=8).pack()
        except Exception as e:
            messagebox.showerror('Mask export', str(e))

    def draw(self):
        cv = self.canvas; cv.delete('all')
        w = max(450, cv.winfo_width()); n = self.display_cells
        cols = min(n, max(1, int(w / 260))); rows = math.ceil(n / cols)
        cw, ch = (w - 24) / cols, 210
        cv.configure(scrollregion=(0, 0, w, rows * ch + 40))
        for i in range(n):
            x, y = 12 + (i % cols) * cw, 12 + (i // cols) * ch
            cv.create_rectangle(x, y, x + cw - 10, y + ch - 10, fill='#182338', outline='#475569')
            cv.create_text(x + 12, y + 18, text=f'Cell {i + 1}', fill='white', anchor='w')
            data = self.last_camera; inp = data['inputs'][i] if data and i < len(data['inputs']) else (0, 0, 0)
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
