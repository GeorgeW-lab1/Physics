#!/usr/bin/env python3
"""Generate and preview a nominal 20 x 20 mm fixed eight-cell aperture mask.
Requires Pillow: python -m pip install pillow
Run standalone, or import export_mask() from the simulation app.
White=open, black=opaque. Fixed mask: no input or factor encodings.
"""
from pathlib import Path
import csv
from PIL import Image, ImageDraw

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
    path=folder/'physical_adder_mask_20x20mm.png'
    image.save(path,dpi=(DPI,DPI))
    with (folder/'aperture_coordinates_20x20mm.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['cell','aperture','x_mm_from_left','y_mm_from_top'])
        writer.writerows(layout)
    (folder/'mask_20x20mm_notes.txt').write_text(
        'Nominal mask size: 20 x 20 mm. 2000 x 2000 pixels at 2540 DPI.\n'
        'Print at actual size / 100%, not fit-to-page. Verify width with a ruler.\n'
        'White=open, black=opaque. Eight identical cells arranged 2 rows x 4 columns.\n'
        'Aperture diameter: nominal 0.4 mm (raster edge quantization applies).\n'
        'Horizontal cell pitch 4.5 mm; vertical pitch 8 mm.\n'
        'Three upper apertures: a,b,carry; two lower apertures: ref A,ref B.\n'
        'Outside array: alignment crosses and an orientation square.\n'
        'Layout only, not validated optical performance.\n'
        'Requires programmable shutters, appropriate LED routing and resolved camera imaging.\n'
        'Use transparent film / fabricated openings; verify opaque material blocks your IR wavelength.\n')
    return path,image

def main():
    import tkinter as tk
    from tkinter import ttk
    from PIL import ImageTk
    path,image=export_mask(Path(__file__).resolve().parent/'masks')
    root=tk.Tk()
    root.title('Physical mask — 20 mm × 20 mm')
    preview=image.convert('RGB').resize((700,700),Image.Resampling.NEAREST)
    photo=ImageTk.PhotoImage(preview)
    label=ttk.Label(root,image=photo)
    label.image=photo
    label.pack(padx=10,pady=10)
    ttk.Label(root,text='20 × 20 mm | 2 × 4 cells | 0.4 mm apertures | white = open',padding=6).pack()
    ttk.Label(root,text=str(path),wraplength=700,padding=6).pack()
    root.mainloop()

if __name__=='__main__':
    main()
