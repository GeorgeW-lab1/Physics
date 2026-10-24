#!/usr/bin/env python3
"""Tkinter demonstration of the supplied abstract dual-rail optical gate model.
Run: python photonic_tkinter.py
No third-party packages required. This is a logic/intensity model, not an
optical propagation or fabrication simulation. Multiplication of two signal
intensities is an assumed gate operation, not implemented by a static mask.
"""
import math
import random
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

ONE, ZERO = (1.0, 0.0), (0.0, 1.0)

class Optics:
    def __init__(self, loss=1.0, leak=0.0, noise=0.0, seed=None):
        self.loss, self.leak, self.noise = loss, leak, noise
        self.rng = random.Random(seed)
        self.gates = 0
    def _fix(self, p, q):
        self.gates += 1
        return min(1.0, p*self.loss+self.leak), min(1.0, q*self.loss+self.leak)
    def AND(self, *xs):
        return self._fix(math.prod(x[0] for x in xs), min(1.0, sum(x[1] for x in xs)))
    def OR(self, *xs):
        return self._fix(min(1.0, sum(x[0] for x in xs)), math.prod(x[1] for x in xs))
    @staticmethod
    def NOT(x):
        return x[1], x[0]
    def XOR(self, x, y):
        return self.OR(self.AND(x, self.NOT(y)), self.AND(self.NOT(x), y))
    def XNOR(self, x, y):
        return self.NOT(self.XOR(x, y))
    def full_add(self, a, b, c):
        t = self.XOR(a, b)
        return self.XOR(t, c), self.OR(self.AND(a, b), self.AND(c, t))

def bits_of(v, w):
    return [ONE if (v >> k) & 1 else ZERO for k in range(w)]

def multiply(o, a, b):
    W = len(a)+len(b)
    acc = [ZERO]*W
    for i, ai in enumerate(a):
        row = [o.AND(ai, bj) for bj in b]
        c = ZERO
        for k in range(i, i+len(b)+1):
            y = row[k-i] if k-i < len(row) else ZERO
            acc[k], c = o.full_add(acc[k], y, c)
    return acc

def run(N, nbits=8, loss=1.0, leak=0.0, noise=0.0, adc_bits=10,
        seed=None, progress=None, cancel=None):
    if not 4 <= N < 1 << nbits:
        raise ValueError('N must satisfy 4 <= N < 2**bits.')
    if not (0 <= loss <= 1 and 0 <= leak <= 1 and noise >= 0):
        raise ValueError('Loss and leak must be in [0, 1]; noise must be nonnegative.')
    o = Optics(loss, leak, noise, seed)
    wa, wb = (nbits+1)//2, nbits-1
    amax, bmax = (1 << wa)-1, (1 << wb)-1
    W = wa+wb
    nb = bits_of(N, W)
    cell, matchA = {}, {}
    for a in range(2, amax+1):
        if cancel is not None and cancel.is_set():
            raise InterruptedError('Cancelled')
        ab, row = bits_of(a, wa), []
        for b in range(2, bmax+1):
            if cancel is not None and b % 32 == 0 and cancel.is_set():
                raise InterruptedError('Cancelled')
            prod = multiply(o, ab, bits_of(b, wb))
            m = o.AND(*[o.XNOR(prod[k], nb[k]) for k in range(W)])
            cell[a, b] = m
            row.append(m)
        matchA[a] = o.OR(*row)
        if progress:
            progress((a-1)/(amax-1)*0.9, f'Computed candidate row a={a}')
    seen, win = ZERO, {}
    for a in range(2, amax+1):
        win[a] = o.AND(matchA[a], o.NOT(seen))
        seen = o.OR(seen, matchA[a])
    a_beams = [o.OR(*[win[a] for a in win if a >> k & 1])[0] for k in range(wa)]
    b_beams = []
    for k in range(wb):
        terms = [o.AND(win[a], cell[a, b]) for a, b in cell if b >> k & 1]
        b_beams.append(o.OR(*terms)[0])
    full = (1 << adc_bits)-1
    def detect(beams, maximum):
        ideal = sum(v*(1 << k)/maximum for k, v in enumerate(beams))
        analog = ideal+o.rng.gauss(0, noise)
        code = max(0, min(full, round(analog*full)))
        return dict(ideal=ideal, analog=analog, code=code,
                    decoded=round(code/full*maximum), maximum=maximum)
    da, db = detect(a_beams, amax), detect(b_beams, bmax)
    a, b = da['decoded'], db['decoded']
    return dict(N=N, a=a, b=b, prime=(a < 2 or b < 2),
                valid=a >= 2 and b >= 2 and a*b == N,
                gates=o.gates, cells=len(cell), cell=cell, win=win,
                matchA=matchA, da=da, db=db, a_beams=a_beams,
                b_beams=b_beams, wa=wa, wb=wb, amax=amax, bmax=bmax,
                adc_bits=adc_bits, seen=seen, params=(loss, leak, noise))

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Photonic factoring bench — 2 LEDs / 2 detectors')
        self.geometry('1180x850')
        self.minsize(950, 710)
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self.result = None
        self.vars = {k: tk.StringVar(value=str(v)) for k, v in
                     dict(N=143, bits=8, loss=1.0, leak=0.0, noise=0.0, adc=10, seed=1).items()}
        self.status = tk.StringVar(value='Ready. Set parameters and press Simulate.')
        self.summary = tk.StringVar(value='No exposure simulated yet.')
        self.hover = tk.StringVar(value='Hover over the card to inspect a candidate.')
        self.make_ui()
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.after(70, self.poll)
    def make_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill='x')
        for i, (key, label, width) in enumerate([
            ('N','Number N',9), ('bits','Input bits',6), ('loss','Transmission / gate',9),
            ('leak','Leak / gate',9), ('noise','Detector noise σ',9),
            ('adc','ADC bits',6), ('seed','Seed (blank=random)',10)]):
            ttk.Label(top, text=label).grid(row=0, column=i, sticky='w', padx=4)
            ttk.Entry(top, textvariable=self.vars[key], width=width).grid(row=1, column=i, padx=4, pady=5)
        actions = ttk.Frame(self, padding=(12,0))
        actions.pack(fill='x')
        self.go = ttk.Button(actions, text='Simulate one exposure', command=self.start)
        self.go.pack(side='left')
        self.test = ttk.Button(actions, text='Ideal self-test: N=4…255', command=lambda:self.start(True))
        self.test.pack(side='left', padx=8)
        ttk.Button(actions, text='Cancel', command=self.cancel.set).pack(side='left')
        self.bar = ttk.Progressbar(actions, maximum=100)
        self.bar.pack(side='left', fill='x', expand=True, padx=12)
        ttk.Label(self, textvariable=self.status, padding=(12,7)).pack(anchor='w')
        ttk.Label(self, textvariable=self.summary, padding=(12,4), font=('TkDefaultFont',12,'bold')).pack(anchor='w')
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=12, pady=7)
        card = ttk.Frame(self.notebook)
        self.notebook.add(card, text='Patterned card / candidate grid')
        self.canvas = tk.Canvas(card, bg='#111827', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda e:self.draw_card())
        self.canvas.bind('<Motion>', self.motion)
        ttk.Label(card, textvariable=self.hover, padding=6).pack(anchor='w')
        read = ttk.Frame(self.notebook)
        self.notebook.add(read, text='Two photodetectors / ADC')
        self.detector = tk.Canvas(read, bg='#111827', highlightthickness=0)
        self.detector.pack(fill='both', expand=True)
        self.detector.bind('<Configure>', lambda e:self.draw_detectors())
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text='Results / model notes')
        self.log = tk.Text(log_frame, wrap='word', font=('TkFixedFont',11))
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.log.pack(fill='both', expand=True)
        self.write('MODEL NOTES\n'
            'This program preserves the supplied abstract dual-rail gate rules.\n'
            'The grid shows one multiplier/comparator for every candidate pair.\n'
            'Priority selects the smallest matching a; weighted TRUE-rail sums feed two ADCs.\n'
            'Noise is applied only at the two detector outputs. Per-gate loss and leak are deterministic.\n'
            'Fan-out is free in this model; merging clips at 1; optical power is not conserved.\n'
            'A static mask supplies fixed transmission, not signal-controlled multiplication.\n'
            'Therefore this is not a physical validation of a passive two-LED factoring card.\n'
            'One exposure refers to the abstract circuit, not Python runtime or zero propagation time.\n'
            'A new N changes the comparator pattern. Full recursive factorization needs additional exposures.\n'
            'A dark/corrupt output is not by itself a primality certificate.\n'
            'Self-test ground truth uses ordinary CPU trial division, not optical output.\n\n')
        ttk.Label(self, text='Static abstract circuit • no clock or sweep in the modeled logic • GUI progress is software evaluation', padding=(12,4)).pack(anchor='w')
    def write(self, text):
        self.log.insert('end', text)
        self.log.see('end')
    def parameters(self):
        v = self.vars
        n, bits, adc = int(v['N'].get()), int(v['bits'].get()), int(v['adc'].get())
        if not 3 <= bits <= 10:
            raise ValueError('Choose 3–10 input bits. Larger cards grow rapidly.')
        if not 1 <= adc <= 24:
            raise ValueError('Choose 1–24 ADC bits.')
        loss, leak, noise = [float(v[k].get()) for k in ('loss','leak','noise')]
        if not all(map(math.isfinite, (loss,leak,noise))):
            raise ValueError('Optical parameters must be finite.')
        if not (0 <= loss <= 1 and 0 <= leak <= 1 and noise >= 0):
            raise ValueError('Transmission and leak must be in [0,1]; noise must be nonnegative.')
        if not 4 <= n < 1 << bits:
            raise ValueError(f'N must be between 4 and {(1 << bits)-1}.')
        seed = int(v['seed'].get()) if v['seed'].get().strip() else None
        return n, dict(nbits=bits, loss=loss, leak=leak, noise=noise, adc_bits=adc, seed=seed)
    def start(self, testing=False):
        if self.busy:
            return
        try:
            params = None if testing else self.parameters()
        except ValueError as e:
            messagebox.showerror('Invalid input', str(e))
            return
        self.busy = True
        self.cancel.clear()
        self.go.configure(state='disabled')
        self.test.configure(state='disabled')
        self.bar['value'] = 0
        self.status.set('Running ideal exhaustive self-test…' if testing else 'Evaluating the static gate network…')
        threading.Thread(target=self.worker, args=(params,testing), daemon=True).start()
    def worker(self, params, testing):
        start = time.perf_counter()
        try:
            if testing:
                failures = []
                for n in range(4,256):
                    if self.cancel.is_set():
                        raise InterruptedError('Cancelled')
                    r = run(n, cancel=self.cancel)
                    truth = next((d for d in range(2,math.isqrt(n)+1) if n%d == 0), None)
                    good = r['prime'] if truth is None else r['valid'] and r['a'] == truth
                    if not good:
                        failures.append(n)
                    self.events.put(('progress', ((n-3)/252, f'Self-test: {n}/255, failures={len(failures)}')))
                self.events.put(('tested', (failures,time.perf_counter()-start)))
            else:
                n, kw = params
                r = run(n, **kw, cancel=self.cancel,
                        progress=lambda p,s:self.events.put(('progress',(p,s))))
                r['seconds'] = time.perf_counter()-start
                self.events.put(('result',r))
        except InterruptedError:
            self.events.put(('cancelled',None))
        except Exception as e:
            self.events.put(('error',str(e)))
    def finish(self):
        self.busy = False
        self.go.configure(state='normal')
        self.test.configure(state='normal')
    def poll(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == 'progress':
                    p, s = data
                    self.bar['value'] = 100*p
                    self.status.set(s)
                else:
                    self.finish()
                    if kind == 'result':
                        self.result = data
                        self.show_result()
                    elif kind == 'tested':
                        fails, sec = data
                        self.bar['value'] = 100
                        text = f'Ideal self-test: {252-len(fails)}/252 correct ({sec:.2f} s)'
                        self.status.set(text)
                        self.write(text+'\n'+(f'Failures: {fails}\n' if fails else '')+'\n')
                    elif kind == 'cancelled':
                        self.status.set('Cancelled; previous completed result retained.')
                    else:
                        self.status.set('Simulation failed.')
                        messagebox.showerror('Simulation error',data)
        except queue.Empty:
            pass
        self.after(70,self.poll)
    def show_result(self):
        r = self.result
        if r['valid']:
            msg = f"N={r['N']}: {r['a']} × {r['b']} = {r['N']} — valid factor pair"
        elif r['prime']:
            msg = f"N={r['N']}: no usable pair decoded — prime-like / inconclusive output"
        else:
            msg = f"N={r['N']}: READOUT ERROR — {r['a']} × {r['b']} ≠ {r['N']}"
        self.summary.set(msg)
        self.bar['value'] = 100
        self.status.set(f"{r['cells']:,} cells | {r['gates']:,} gate evaluations | {r['seconds']:.3f} s CPU elapsed")
        self.write(msg+'\n'+self.status.get()+'\n')
        for name in ('a','b'):
            d = r['d'+name]
            self.write(f"Detector {name}: pre-noise={d['ideal']:.8f}, noisy={d['analog']:.8f}, ADC={d['code']}/{(1 << r['adc_bits'])-1}, decoded={d['decoded']}\n")
        self.write(f"Transmission / leak / noise = {r['params']}\n\n")
        self.draw_card()
        self.draw_detectors()
    def draw_card(self):
        c = self.canvas
        c.delete('all')
        w, h = c.winfo_width(), c.winfo_height()
        if not self.result:
            c.create_text(w/2,h/2,text='Press “Simulate one exposure” to illuminate the candidate card.',fill='white')
            return
        r = self.result
        c.create_oval(20,16,44,40,fill='#ff695e',outline='')
        c.create_text(52,28,text='LED A: TRUE rail',fill='#ffaba4',anchor='w')
        c.create_oval(235,16,259,40,fill='#60a5fa',outline='')
        c.create_text(267,28,text='LED B: FALSE rail',fill='#93c5fd',anchor='w')
        c.create_text(20,59,text='Candidate equality TRUE intensity: dark = 0, bright = 1; cyan outline = decoded valid pair',fill='#d1d5db',anchor='w')
        x0,y0,x1,y1 = 62,100,max(63,w-30),max(101,h-47)
        cols, rows = r['bmax']-1, r['amax']-1
        dx,dy = (x1-x0)/cols,(y1-y0)/rows
        self.grid_geometry = (x0,y0,x1,y1,dx,dy)
        for (a,b),(p,q) in r['cell'].items():
            v = max(0,min(1,p))
            color = '#%02x%02x%02x' % (int(15+65*v),int(23+232*v),int(42+73*v))
            x,y = x0+(b-2)*dx,y0+(a-2)*dy
            c.create_rectangle(x,y,x+dx,y+dy,fill=color,outline='')
            if r['valid'] and (a,b) == (r['a'],r['b']):
                c.create_rectangle(x,y,x+dx,y+dy,outline='#22d3ee',width=2)
        for a in range(2,r['amax']+1):
            c.create_text(x0-8,y0+(a-1.5)*dy,text=str(a),fill='#cbd5e1',anchor='e')
        step = max(1,math.ceil(cols/16))
        for b in range(2,r['bmax']+1,step):
            c.create_text(x0+(b-1.5)*dx,y0-12,text=str(b),fill='#cbd5e1')
        c.create_text(20,y0-12,text='a / b',fill='#cbd5e1',anchor='w')
        c.create_text(x0,h-20,text=f"{rows} candidate a rows × {cols} candidate b columns. Every cell is evaluated; no candidate pruning.",fill='#cbd5e1',anchor='w')
    def motion(self,event):
        if not self.result or not hasattr(self,'grid_geometry'):
            return
        x0,y0,x1,y1,dx,dy = self.grid_geometry
        if x0 <= event.x < x1 and y0 <= event.y < y1:
            a,b = int((event.y-y0)/dy)+2,int((event.x-x0)/dx)+2
            p,q = self.result['cell'][a,b]
            wp,wq = self.result['win'][a]
            self.hover.set(f'a={a}, b={b}, a×b={a*b} | equality rails T={p:.6g}, F={q:.6g} | row winner T={wp:.6g}, F={wq:.6g}')
    def draw_detectors(self):
        c = self.detector
        c.delete('all')
        if not self.result:
            return
        r = self.result
        w = c.winfo_width()
        c.create_text(24,25,text='Two analog detector channels — weighted TRUE-rail sums',fill='white',font=('TkDefaultFont',14,'bold'),anchor='w')
        for index,name in enumerate(('a','b')):
            y = 75+index*190
            d, beams = r['d'+name],r[name+'_beams']
            c.create_text(25,y,text=f"Detector {index+1} → {name}: decoded {d['decoded']} / range 0…{d['maximum']}",fill='#e5e7eb',anchor='w')
            x0,x1 = 25,max(26,w-30)
            c.create_rectangle(x0,y+22,x1,y+60,fill='#273449',outline='')
            c.create_rectangle(x0,y+22,x0+(x1-x0)*max(0,min(1,d['analog'])),y+60,fill='#34d399' if index == 0 else '#60a5fa',outline='')
            c.create_text(25,y+80,text=f"Intensity {d['analog']:.8f}   |   ADC {d['code']} / {(1 << r['adc_bits'])-1}   |   noiseless sum {d['ideal']:.8f}",fill='#d1d5db',anchor='w')
            c.create_text(25,y+107,text='Bit beams (LSB first): '+ '  '.join(f'{v:.4g}' for v in beams),fill='#cbd5e1',anchor='w')
            c.create_text(25,y+132,text=f"Weights: 2^k / {d['maximum']} ; adjacent integer spacing = {1/d['maximum']:.6g} full scale",fill='#94a3b8',anchor='w')
        c.create_text(25,470,text='Displayed bars clip to [0,1]; the numeric noisy intensity is shown before ADC clipping.',fill='#94a3b8',anchor='w')
    def close(self):
        self.cancel.set()
        self.destroy()

if __name__ == '__main__':
    App().mainloop()
