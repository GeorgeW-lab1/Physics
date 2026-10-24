#!/usr/bin/env python3
"""Launch beside reusable_ir_bench.py. Requires Pillow.
python -m pip install pillow
python reusable_ir_mask_png.py
Exports a fixed, flat 2D eight-cell aperture mask, not a camera screenshot.
White = open; black = opaque. Print at 100% scale, no fit-to-page.
Nominal dimensions are layout choices, not validated optical specifications.
"""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
try:
    from PIL import Image, ImageDraw, ImageTk
except ImportError:
    raise SystemExit('Install Pillow: python -m pip install pillow')
try:
    from reusable_ir_bench import App
except ImportError:
    raise SystemExit('Place this file beside reusable_ir_bench.py.')

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

class MaskApp(App):
    def __init__(self):
        self.mask_preview=None
        self.user_stopping=False
        super().__init__()
        self.title('Reusable IR bench — physical 2D mask PNG')
        ttk.Button(self,text='Export / show physical 2D mask PNG',command=self.show_mask).pack(pady=5)
        # Export immediately and after each successful simulation.
        self.after_idle(self.show_mask)
    def finish(self):
        success=(self.job is not None and self.last_camera is not None
                 and not self.status.get().startswith('Stopped:') and not self.user_stopping)
        super().finish()
        if success:
            self.after_idle(self.show_mask)
    def stop(self):
        self.user_stopping=True
        try:
            super().stop()
        finally:
            self.user_stopping=False
    def show_mask(self):
        try:
            path,image=export_mask(Path(__file__).resolve().parent/'masks')
            self.say('Physical 2D mask PNG saved: '+str(path))
            if self.mask_preview is not None and self.mask_preview.winfo_exists():
                self.mask_preview.lift()
                return
            win=tk.Toplevel(self)
            self.mask_preview=win
            win.title('Physical 2D aperture mask — white=open, black=opaque')
            preview=image.convert('RGB')
            preview.thumbnail((1000,650))
            photo=ImageTk.PhotoImage(preview)
            label=ttk.Label(win,image=photo)
            label.image=photo
            label.pack(padx=12,pady=12)
            ttk.Label(win,text='2 × 4 identical cells | 110 × 65 mm nominal | 600 DPI | white = transmitting aperture',padding=5).pack()
            ttk.Label(win,text='Upper apertures: a, b, carry. Lower apertures: reference A, reference B.\n'
                'Printed PNG has no text or cell outlines inside the active mask.',padding=5).pack()
            ttk.Label(win,text=str(path),wraplength=1000,padding=5).pack()
        except Exception as e:
            messagebox.showerror('Mask export failed',str(e))

if __name__=='__main__':
    MaskApp().mainloop()
