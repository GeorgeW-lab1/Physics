#!/usr/bin/env python3
"""Eight reusable passive optical-adder cells, camera readout, electronic feedback.
Run: python reusable_ir_bench.py
Standard library only (Tkinter required).

Not an all-optical factorer: CPU schedules trial divisors and repeated additions.
Each addition uses measured camera intensities for sum and carry. No modulo or
integer multiplication is used to decide a candidate match. Camera electronics
threshold measured levels. Input modulation and carry feedback are electronic.
Two LEDs illuminate three signal apertures (A: a,b; B: carry) and references.
Independently addressable shutters/SLM regions are REQUIRED to set the inputs.
A fixed printed mask and two unmodulated LEDs alone cannot supply those inputs.

The camera model is a calibrated integrating ROI per cell: incoherent light
from equal-power apertures is summed within that ROI. A physical build needs
collection optics/defocus or pixel summation, flat-field calibration, IR-sensitive
camera, modulation hardware, and repeatable exposure. No diffraction model.
"""
import math
import random
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
from pathlib import Path
from PIL import Image, ImageDraw

CELLS = 8

DPI=2540  # Exactly 100 pixels/mm: 2000 pixels = 20 mm.
SIZE_MM=20
PX_PER_MM=100
DIAMETER_MM=0.7

def mm(v):
    return round(v*PX_PER_MM)

def make_mask():
    image=Image.new('1',(2000,2000),0)
    draw=ImageDraw.Draw(image)
    layout=[]
    radius=mm(DIAMETER_MM)/2
    for i in range(8):
        cx=3.25+(i%4)*4.5
        cy=6+(i//4)*8
        for label,dx,dy in [('a',-1, -.7),('b',0,-.7),('carry',1,-.7),
                            ('ref_A',-.5,.7),('ref_B',.5,.7)]:
            x,y=mm(cx+dx),mm(cy+dy)
            draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=1)
            layout.append((i+1,label,cx+dx,cy+dy))
    for xx,yy in [(1.2,1.2),(18.8,1.2),(1.2,18.8),(18.8,18.8)]:
        x,y=mm(xx),mm(yy)
        draw.line((x-mm(.4),y,x+mm(.4),y),fill=1,width=mm(.08))
        draw.line((x,y-mm(.4),x,y+mm(.4)),fill=1,width=mm(.08))
    draw.rectangle((mm(2),mm(.9),mm(2.25),mm(1.15)),fill=1)
    return image,layout

def export_mask(folder):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    image,layout=make_mask()
    path=folder/'physical_adder_mask_2x4_600dpi.png'
    image.save(path,dpi=(DPI,DPI))
    import csv
    with (folder/'physical_adder_mask_coordinates_mm.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['cell','aperture','x_mm_from_left','y_mm_from_top'])
        writer.writerows(layout)
    (folder/'physical_adder_mask_README.txt').write_text(
        'FIXED PHYSICAL ADDER MASK\n'
        '2 rows x 4 identical cells. White=open; black=opaque.\n'
        'Nominal print size: 20 x 20 mm at 2540 DPI. Print at 100%, never fit-to-page.\n'
        'Raster dimensions are rounded to whole pixels; verify scale with a ruler.\n'
        'Five 1.5 mm nominal holes per cell: a,b,carry above; reference A,B below.\n'
        '25 mm horizontal/vertical cell pitch. Four alignment crosses outside array.\n'
        'Corner square identifies orientation. Coordinates measured from upper-left.\n'
        'No factor/product table; no N-dependent geometry; all cells identical.\n'
        'This is a proposed aperture layout, not an optical prescription or verified device.\n'
        'Requires programmable shutters/SLM for a,b,carry, separate LED routing, and camera optics.\n'
        'The camera sums the three signal aperture regions but measures references separately.\n'
        'For transparent film: black must block the operating IR wavelength; verify experimentally.\n'
        'The image is not an ordinary paper stencil: opaque paper would also block white regions.\n')
    if not path.exists() or not path.stat().st_size:
        raise OSError('Mask PNG export failed.')
    return path,image

class CameraBench:
    def __init__(self, exposure=.22, leak=.001, noise=.002, seed=1, led_a=True, led_b=True):
        self.exposure,self.leak,self.noise = exposure,leak,noise
        self.rng = random.Random(seed)
        self.led_a,self.led_b = led_a,led_b
        self.exposures = 0
        self.last = None
    def capture(self, inputs):
        if len(inputs) != CELLS:
            raise ValueError('Exactly eight input cells required.')
        def pixel(v):
            return max(0.0,min(1.0,v+self.rng.gauss(0,self.noise)))
        rows = []
        for a,b,c in inputs:
            if any(x not in (0,1) for x in (a,b,c)):
                raise ValueError('Inputs must be binary.')
            # All numbers below are light intensities, not Boolean gate outputs.
            dark = .01
            power = self.exposure
            A,B = float(self.led_a),float(self.led_b)
            t = lambda x:self.leak+(1-self.leak)*x
            signal = pixel(dark+power*(A*(t(a)+t(b))+B*t(c)))
            black = pixel(dark)
            refa = pixel(dark+power*A)
            refb = pixel(dark+power*B)
            baseline = pixel(dark+power*self.leak*(2*A+B))
            rows.append(dict(signal=signal,black=black,refa=refa,refb=refb,baseline=baseline))
        self.exposures += 1
        # Decoder receives only measured ROIs, not input bits.
        decoded = []
        for row in rows:
            A = row['refa']-row['black']
            B = row['refb']-row['black']
            if min(A,B) < .04 or abs(A-B) > .25*max(A,B):
                raise RuntimeError('Insufficient / unequal LED reference signal. Check both LEDs and exposure.')
            step = (A+B)/2*(1-self.leak)
            level = (row['signal']-row['baseline'])/step
            k = round(level)
            if row['signal'] > .98 or not 0 <= k <= 3 or abs(level-k) > .35:
                raise RuntimeError('Ambiguous or saturated camera level; reduce noise/leak or adjust exposure.')
            decoded.append(dict(level=level,k=k,sum=k%2,carry=int(k>=2)))
        self.last = dict(inputs=list(inputs),rows=rows,decoded=decoded)
        return decoded


def optical_add(bench,x,y,width):
    """Generator: eight independent bit lanes per exposure; carry dependencies
    are scheduled in later exposures, never resolved by a hidden software adder.
    Lanes are reused for positions k, k+8, ... across eight parallel additions.
    """
    if len(x) != CELLS or len(y) != CELLS:
        raise ValueError('Eight lanes required.')
    carry = [0]*CELLS
    result = [0]*CELLS
    for k in range(width):
        inputs = [((x[i]>>k)&1,(y[i]>>k)&1,carry[i]) for i in range(CELLS)]
        decoded = bench.capture(inputs)
        for i,d in enumerate(decoded):
            result[i] |= d['sum'] << k
            carry[i] = d['carry']
        yield dict(type='frame',bit=k,width=width)
    if any(carry):
        raise RuntimeError('Arithmetic overflow: increase accumulator width.')
    return result


def factor_job(bench,N,report):
    """CPU trial-division schedule; optical repeated-addition arithmetic.
    Uses no stored product table. Only one factor pair is returned.
    Search cost is intentionally visible, not presented as an optical speedup.
    """
    width = N.bit_length()+1
    candidates = list(range(2,math.isqrt(N)+1))
    if not candidates:
        return None
    for start in range(0,len(candidates),CELLS):
        batch = candidates[start:start+CELLS]
        active = [i < len(batch) for i in range(CELLS)]
        divisors = batch+[0]*(CELLS-len(batch))
        sums,counts = [0]*CELLS,[0]*CELLS
        report(f'Electronic scheduler: candidate divisors {batch}; repeated optical additions')
        while any(active):
            increments = [divisors[i] if active[i] else 0 for i in range(CELLS)]
            sums = yield from optical_add(bench,sums,increments,width)
            hits = []
            for i in range(CELLS):
                if not active[i]:
                    continue
                counts[i] += 1
                # Electronic comparison of camera-derived accumulated integers.
                if sums[i] == N:
                    hits.append((divisors[i],counts[i]))
                    active[i] = False
                elif sums[i] > N:
                    active[i] = False
            if hits:
                return min(hits)
    return None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('8 reusable IR arithmetic cells — camera + electronic feedback')
        self.geometry('1190x850')
        self.minsize(1000,720)
        self.vars = {k:tk.StringVar(value=str(v)) for k,v in
            dict(N=143,x=11,y=13,exposure=.22,leak=.001,noise=.002,seed=1,delay=5).items()}
        self.A,self.B = tk.BooleanVar(value=True),tk.BooleanVar(value=True)
        self.status = tk.StringVar(value='Ready. Start with the full-adder truth-table test.')
        self.job = None
        self.frames = 0
        self.last_inputs = [(0,0,0)]*CELLS
        self.last_camera = None
        self.photo = None
        self.log_lines = deque(maxlen=300)
        top = ttk.Frame(self,padding=8)
        top.pack(fill='x')
        labels = [('N','Factor N (4–255)'),('x','Adder X (0–255)'),('y','Adder Y (0–255)'),
                  ('exposure','Exposure'),('leak','Closed transmission'),('noise','Camera noise σ'),
                  ('seed','Seed'),('delay','Frame delay ms')]
        for j,(key,label) in enumerate(labels):
            ttk.Label(top,text=label).grid(row=0,column=j,padx=3,sticky='w')
            ttk.Entry(top,textvariable=self.vars[key],width=12).grid(row=1,column=j,padx=3)
        bar = ttk.Frame(self,padding=8)
        bar.pack(fill='x')
        self.buttons = []
        for text,command in [('Full-adder test',lambda:self.start('truth')),('Optical X + Y',lambda:self.start('add')),
                             ('Factor N (hybrid)',lambda:self.start('factor'))]:
            b = ttk.Button(bar,text=text,command=command)
            b.pack(side='left',padx=4)
            self.buttons.append(b)
        ttk.Button(bar,text='Stop',command=self.stop).pack(side='left',padx=4)
        ttk.Checkbutton(bar,text='LED A',variable=self.A).pack(side='left',padx=8)
        ttk.Checkbutton(bar,text='LED B',variable=self.B).pack(side='left')
        ttk.Label(self,textvariable=self.status,padding=8,wraplength=1150).pack(fill='x')
        self.canvas = tk.Canvas(self,bg='#101827',height=430,highlightthickness=0)
        self.canvas.pack(fill='both',expand=True,padx=8)
        self.canvas.bind('<Configure>',lambda e:self.draw())
        self.log = tk.Text(self,height=9,wrap='word',font=('TkFixedFont',10))
        self.log.pack(fill='x',padx=8,pady=8)
        self.log.configure(state='disabled')
        ttk.Label(self,text='PHYSICAL BOUNDARY: mask = fixed equal apertures; shutters/SLM = programmable inputs; camera = light sums.\n'
            'CPU = thresholding, carry feedback, comparison and candidate search. Multiple exposures, no product lookup, no all-optical factoring claim.',padding=8).pack(fill='x')
        self.say('Eight identical cells. Each has three signal apertures (a,b,c), two reference ROIs and a black/baseline calibration ROI.')
        self.draw()
    def say(self,s):
        self.log_lines.append(s)
        self.log.configure(state='normal')
        self.log.delete('1.0','end')
        self.log.insert('end','\n'.join(self.log_lines))
        self.log.see('end')
        self.log.configure(state='disabled')
    def params(self):
        n,x,y = [int(self.vars[k].get()) for k in ('N','x','y')]
        if not 4 <= n <= 255 or not 0 <= x <= 255 or not 0 <= y <= 255:
            raise ValueError('N must be 4–255; X and Y must be 0–255.')
        power,leak,noise = [float(self.vars[k].get()) for k in ('exposure','leak','noise')]
        if not all(math.isfinite(z) for z in (power,leak,noise)):
            raise ValueError('Parameters must be finite.')
        if not (0 <= power <= 1 and 0 <= leak < .5 and 0 <= noise <= 1):
            raise ValueError('Exposure 0–1, leak 0–<0.5, noise 0–1.')
        delay = int(self.vars['delay'].get())
        if not 0 <= delay <= 1000:
            raise ValueError('Delay must be 0–1000 ms.')
        bench = CameraBench(power,leak,noise,int(self.vars['seed'].get()),self.A.get(),self.B.get())
        return n,x,y,delay,bench
    def start(self,mode):
        if self.job is not None:
            return
        try:
            self.N,self.x,self.y,self.delay,self.bench = self.params()
        except ValueError as e:
            messagebox.showerror('Input',str(e)); return
        self.mode = mode
        self.frames = 0
        self.last_camera = None
        for b in self.buttons:
            b.configure(state='disabled')
        if mode == 'factor':
            self.say(f'Factoring {self.N}: electronic trial-divisor search; camera-derived repeated sums.')
            self.job = factor_job(self.bench,self.N,self.say)
        elif mode == 'add':
            self.say(f'Adding {self.x} and {self.y} in all eight lanes; 9 camera exposures.')
            self.job = optical_add(self.bench,[self.x]*CELLS,[self.y]*CELLS,9)
        else:
            self.job = self.truth_job()
        self.after(1,self.tick)
    def truth_job(self):
        inputs = [((i>>2)&1,(i>>1)&1,i&1) for i in range(CELLS)]
        ds = self.bench.capture(inputs)
        yield dict(type='frame',bit=0,width=1)
        failures = []
        for inp,d in zip(inputs,ds):
            expected = sum(inp)
            if d['k'] != expected:
                failures.append((inp,d['k']))
        if failures:
            raise RuntimeError(f'Incorrect truth-table camera decoding: {failures}')
        return 'All 8 full-adder input combinations passed in one exposure.'
    def tick(self):
        if self.job is None:
            return
        try:
            event = next(self.job)
            self.last_camera = self.bench.last
            self.frames = self.bench.exposures
            self.status.set(f'{self.mode}: exposure {self.frames}; bit {event["bit"]+1}/{event["width"]}; fixed hardware: 8 arithmetic cells')
            self.draw()
            self.after(self.delay,self.tick)
        except StopIteration as stop:
            result = stop.value
            if self.mode == 'factor':
                msg = f'Camera-derived candidate factor pair: {self.N} = {result[0]} × {result[1]}' if result else 'No divisor detected by the hybrid search; a noisy run is not a primality certificate.'
            elif self.mode == 'add':
                msg = f'Camera-derived addition result: {self.x} + {self.y} → {result[0]} (eight lanes: {result})'
            else:
                msg = result
            self.status.set(msg+f' | {self.bench.exposures} exposures')
            self.say(self.status.get())
            self.finish()
        except Exception as e:
            self.status.set('Stopped: '+str(e))
            self.say(self.status.get())
            self.finish()
    def finish(self):
        self.job = None
        for b in self.buttons:
            b.configure(state='normal')
    def stop(self):
        if self.job is not None:
            self.job.close()
            self.finish()
            self.status.set('Stopped. No further exposures scheduled.')
    def draw(self):
        cv = self.canvas
        cv.delete('all')
        w,h = max(900,cv.winfo_width()),max(350,cv.winfo_height())
        cv.create_text(18,23,text='Fixed repeated cell geometry + independently modulated inputs',fill='white',anchor='w',font=('TkDefaultFont',13,'bold'))
        margin,gap = 18,12
        cw = (w-2*margin-3*gap)/4
        ch = (h-65-15)/2
        for i in range(CELLS):
            x,y = margin+(i%4)*(cw+gap),50+(i//4)*(ch+8)
            cv.create_rectangle(x,y,x+cw,y+ch,outline='#475569',fill='#182338')
            cv.create_text(x+10,y+16,text=f'Cell {i+1}',fill='#e2e8f0',anchor='w')
            data = self.last_camera
            inp = data['inputs'][i] if data else (0,0,0)
            for j,label in enumerate(('a / A','b / A','carry / B')):
                xx = x+22+j*(cw-40)/3
                cv.create_rectangle(xx,y+33,xx+19,y+52,fill='#fbbf24' if inp[j] else '#0b1020',outline='#94a3b8')
                cv.create_text(xx+9,y+65,text=label,fill='#94a3b8',font=('TkDefaultFont',9))
            if data:
                r,d = data['rows'][i],data['decoded'][i]
                for j,(label,key) in enumerate((('signal','signal'),('ref A','refa'),('ref B','refb'))):
                    yy = y+85+j*17
                    cv.create_text(x+10,yy,text=label,fill='#94a3b8',anchor='w',font=('TkDefaultFont',9))
                    left,right = x+63,x+cw-50
                    cv.create_rectangle(left,yy-5,right,yy+5,fill='#0b1020',outline='')
                    cv.create_rectangle(left,yy-5,left+(right-left)*r[key],yy+5,fill='#34d399',outline='')
                    cv.create_text(right+5,yy,text=f'{r[key]:.2f}',fill='white',anchor='w',font=('TkDefaultFont',9))
                cv.create_text(x+10,y+ch-16,text=f'Camera level {d["level"]:.2f} → sum {d["sum"]}, carry {d["carry"]}',fill='#67e8f9',anchor='w',font=('TkDefaultFont',10))
            else:
                cv.create_text(x+10,y+100,text='No camera exposure yet',fill='#94a3b8',anchor='w')

if __name__ == '__main__':
    export_mask("mask")
    App().mainloop()
