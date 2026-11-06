"""
Bit-Pooling Higher-Lower Search & Circuit Camera Lab 
(Tkinter / OpenCV Framework with Visual Bit Registers & Target Reveal)
Adapted from the RP-style optoelectronic decision circuit framework.
"""

import asyncio
import queue
import random
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# ==============================================================================
# SECURE ORACLE CLASS (RESTRICTS TARGET ACCESS UNTIL DONE)
# ==============================================================================
class Oracle:
    """Encapsulates the secret target and provides blind higher/lower queries."""
    def __init__(self, max_val, target=None):
        self.max_val = max_val
        self._target = int(np.clip(target, 0, max_val)) if target is not None else random.randint(0, max_val)
        self.is_done = False

    def query(self, guess):
        """Blind query interface returning directional feedback without exposing target."""
        if self.is_done:
            return 0
        
        if guess == self._target:
            self.is_done = True
            return 0  # Correct / Locked
        elif guess < self._target:
            return -1 # Target is higher
        else:
            return 1  # Target is lower

    @property
    def target(self):
        """Raises PermissionError if accessed before the search is done."""
        if not self.is_done:
            raise PermissionError("SECURITY VIOLATION: Oracle target is classified and inaccessible until search is done!")
        return self._target


# ==============================================================================
# BIT-POOLING SEARCH & HARDWARE SIMULATION ENGINE
# ==============================================================================
class BitPoolingSearchEngine:
    def __init__(self, bit_depth=10):
        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1
        
        # Search state initialized with secure Oracle
        self.oracle = Oracle(self.max_val)
        self.low = 0
        self.high = self.max_val
        self.current_guess = (self.low + self.high) // 2
        self.step_count = 0
        self.history = []
        self.converged = False
        self.last_status = "INITIALIZED"

        # Hardware & Pooling Settings
        self.kernel_size = 4        # 2, 4, or 8 bits
        self.stride = 2             # 1, 2, or 4
        self.pool_op = "POPCOUNT"   # "POPCOUNT", "AND", "OR", "XOR", "MAJORITY"
        self.invert_mask = False
        self.bit_shift = 0          # Bit rotation offset
        self.mask_preset = "ALL"    # "ALL", "EVEN", "ODD"
        
        self.schmitt_trigger = 0.5  # Noise gate threshold
        self.thermal_noise = 0.02   # Signal noise variance
        
        # Low-level registers
        self.guess_reg = 0
        self.target_reg = 0
        self.pooled_register = 0
        self.gate_activity = np.zeros((16, 16), dtype=np.float32)
        
        self._update_registers()

    def reset(self, new_target=None):
        self.oracle = Oracle(self.max_val, new_target)
        self.low = 0
        self.high = self.max_val
        self.current_guess = (self.low + self.high) // 2
        self.step_count = 0
        self.history = [self.current_guess]
        self.converged = False
        self.last_status = "READY"
        self._update_registers()

    def _apply_bit_pooling(self, val):
        shifted_val = ((val << self.bit_shift) | (val >> (self.bit_depth - self.bit_shift))) & self.max_val
        bits = [(shifted_val >> i) & 1 for i in range(self.bit_depth)]
        
        if self.mask_preset == "EVEN":
            bits = [b if i % 2 == 0 else 0 for i, b in enumerate(bits)]
        elif self.mask_preset == "ODD":
            bits = [b if i % 2 != 0 else 0 for i, b in enumerate(bits)]
            
        if self.invert_mask:
            bits = [1 - b for b in bits]
            
        pooled_bits = []
        for i in range(0, len(bits), self.stride):
            block = bits[i : i + self.kernel_size]
            if not block:
                break
                
            if self.pool_op == "POPCOUNT":
                res = sum(block) / len(block)
                res_bit = 1 if res >= self.schmitt_trigger else 0
            elif self.pool_op == "AND":
                res_bit = 1 if all(b == 1 for b in block) else 0
            elif self.pool_op == "OR":
                res_bit = 1 if any(b == 1 for b in block) else 0
            elif self.pool_op == "XOR":
                res_bit = sum(block) % 2
            elif self.pool_op == "MAJORITY":
                res_bit = 1 if sum(block) > (len(block) / 2.0) else 0
            else:
                res_bit = block[0]
                
            pooled_bits.append(res_bit)

        out_val = 0
        for idx, b in enumerate(pooled_bits):
            out_val |= (b << idx)
        return out_val

    def step(self):
        if self.converged or self.low > self.high:
            return

        self.step_count += 1
        
        # Query the oracle blindly (no direct access to target value)
        feedback = self.oracle.query(self.current_guess)
        
        if feedback == 0:
            self.converged = True
            self.last_status = f"LOCKED AT {self.current_guess}"
        elif feedback < 0:
            self.low = self.current_guess + 1
            self.last_status = "HIGHER (TARGET > GUESS)"
        else:
            self.high = self.current_guess - 1
            self.last_status = "LOWER (TARGET < GUESS)"
            
        self.current_guess = (self.low + self.high) // 2
        self.history.append(self.current_guess)
        self._update_registers()
        self._stimulate_gates()

    def _update_registers(self):
        self.guess_reg = self.current_guess
        try:
            # Attempt to read target from oracle; fails securely if not converged/done
            self.target_reg = self.oracle.target
        except PermissionError:
            self.target_reg = 0  # Masked while search is active
            
        self.pooled_register = self._apply_bit_pooling(self.current_guess)

    def _stimulate_gates(self):
        for i in range(16):
            for j in range(16):
                if (self.guess_reg >> ((i + j) % 10)) & 1:
                    self.gate_activity[i, j] = min(1.0, self.gate_activity[i, j] + 0.7)

    def update_decay(self, decay_rate):
        self.gate_activity *= decay_rate


# ==============================================================================
# CAMERA SOURCE HANDLING
# ==============================================================================
class Source:
    """Real USB camera via OpenCV."""

    def __init__(self, index: int):
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed. Run: pip install opencv-python")
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not (cap.isOpened() and cap.read()[0]):
            cap.release()
            raise RuntimeError(f"Cannot open camera {index}.")
        self.cap = cap
        self.rgb = None

    def read(self):
        ok, bgr = self.cap.read()
        if not ok:
            return None
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def close(self):
        self.cap.release()


def ppm_photo(rgb: np.ndarray) -> tk.PhotoImage:
    h, w, _ = rgb.shape
    data = b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(rgb, np.uint8).tobytes()
    return tk.PhotoImage(data=data, format="PPM")


# ==============================================================================
# BACKGROUND SIMULATION THREAD (RP FRAMEWORK)
# ==============================================================================
class SimThread(threading.Thread):
    def __init__(self, cam_index, ctl, out, engine):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.ctl = ctl
        self.out = out
        self.engine = engine

    def run(self):
        asyncio.run(self._run())

    async def _run(self):
        ctl, out, engine = self.ctl, self.out, self.engine
        try:
            src = Source(self.cam_index)
        except RuntimeError as e:
            out.put(("error", str(e)))
            out.put(("done", None))
            return
        out.put(("info", f"USB camera {self.cam_index} opened"))

        t_prev = time.perf_counter()
        idx = 0
        try:
            while not ctl["stop"]:
                gray = await asyncio.to_thread(src.read)
                if gray is None:
                    out.put(("info", "camera stopped delivering frames"))
                    break
                now = time.perf_counter()
                dt = min(max(now - t_prev, 1e-3), 0.25)
                t_prev = now

                small = cv2.resize(gray, (128, 96), interpolation=cv2.INTER_AREA)
                motion_detected = idx > 5 and np.std(small) > 5.0

                if motion_detected and ctl.get("auto_step", True):
                    engine.step()

                engine.update_decay(ctl.get("shutter_decay", 0.90))

                rgb = src.rgb if src.rgb is not None else np.stack([gray] * 3, axis=-1)
                preview = self.render_circuit_camera(rgb, engine, ctl)

                # Safely send target value if converged or revealed
                exposed_target = engine.oracle._target if (engine.converged or ctl.get("force_reveal", False)) else 0

                out.put(("frame", dict(
                    preview=preview, idx=idx, dt=dt, target=exposed_target,
                    guess=engine.current_guess, pooled=engine.pooled_register,
                    status=engine.last_status, steps=engine.step_count,
                    history=list(engine.history), converged=engine.converged
                )))
                idx += 1
                await asyncio.sleep(0.03)
        finally:
            src.close()
            out.put(("done", None))

    def render_circuit_camera(self, rgb, engine, ctl):
        h, w = rgb.shape[:2]
        pw, ph = 320, 240
        pv = rgb[np.arange(ph) * h // ph][:, np.arange(pw) * w // pw].copy()
        
        if ctl.get("ir_heatmap", False):
            pv[:, :, 0] = np.clip(pv[:, :, 0].astype(int) + 80, 0, 255)
            pv[:, :, 2] = np.clip(pv[:, :, 2].astype(int) - 50, 0, 255)
        return pv


# ==============================================================================
# MAIN GUI APPLICATION (TKINTER FRAMEWORK)
# ==============================================================================
class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Bit-Pooling Search Lab — Secure Oracle & Visual Registers")
        self.q = queue.Queue()
        self.ctl = dict(stop=False, ir_heatmap=False, shutter_decay=0.90, auto_step=True, force_reveal=False)
        self.engine = BitPoolingSearchEngine(bit_depth=10)
        self.thread = None

        self._build_ui()
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self.window.after(15, self._poll)

    def _build_ui(self):
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Controls Top Frame
        ctrl_frame = ttk.LabelFrame(main_frame, text="Laboratory Controls")
        ctrl_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(ctrl_frame, text="Start Lab", command=self.start_lab).pack(side="left", padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Stop Lab", command=self.stop_lab).pack(side="left", padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Step Search", command=self.step_search).pack(side="left", padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Reset Search", command=self.reset_search).pack(side="left", padx=5, pady=5)

        self.v_ir = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl_frame, text="IR Thermal Heatmap", variable=self.v_ir,
                        command=lambda: self.ctl.update(ir_heatmap=self.v_ir.get())).pack(side="left", padx=10)

        # Target Management Frame (Custom, Random, Reveal)
        target_frame = ttk.LabelFrame(main_frame, text="Hidden Target Management (Secured Oracle)")
        target_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(target_frame, text="Target (0-1023):").pack(side="left", padx=5, pady=5)
        self.v_custom_target = tk.StringVar(value="[SECURE]")
        self.e_custom_target = ttk.Entry(target_frame, width=8, textvariable=self.v_custom_target)
        self.e_custom_target.pack(side="left", padx=5, pady=5)

        ttk.Button(target_frame, text="Set Custom", command=self.set_custom_target).pack(side="left", padx=2, pady=5)
        ttk.Button(target_frame, text="Randomize Target", command=self.randomize_target).pack(side="left", padx=5, pady=5)

        self.v_reveal = tk.BooleanVar(value=False)
        self.chk_reveal = ttk.Checkbutton(target_frame, text="Reveal Target", variable=self.v_reveal,
                                          command=lambda: self.ctl.update(force_reveal=self.v_reveal.get()))
        self.chk_reveal.pack(side="left", padx=15, pady=5)

        # Viewport and Registers Frame
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Left: Circuit Camera Viewport
        cam_frame = ttk.LabelFrame(display_frame, text="Optoelectronic Circuit Camera Viewport")
        cam_frame.pack(side="left", padx=5, pady=5)
        self.cam_canvas = tk.Canvas(cam_frame, width=320, height=240, bg="black")
        self.cam_canvas.pack(padx=5, pady=5)
        self.cam_img_id = self.cam_canvas.create_image(0, 0, anchor="nw")

        # Right: Register & Hardware Status Panel (Visual Bit Cells)
        right_panel = ttk.Frame(display_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        reg_frame = ttk.LabelFrame(right_panel, text="Bit-Pooling Hardware Registers")
        reg_frame.pack(fill="x", padx=5, pady=5)

        # Canvas for rendering register bit cells (Target, Guess, Pooled)
        self.reg_canvas = tk.Canvas(reg_frame, width=480, height=140, bg="#121414", highlightthickness=0)
        self.reg_canvas.pack(padx=5, pady=5)

        self.lbl_status = ttk.Label(reg_frame, text="Status: IDLE", font=("Consolas", 11), foreground="#00e5ff")
        self.lbl_status.pack(anchor="w", padx=10, pady=5)

        # Settings Configuration Panel
        set_frame = ttk.LabelFrame(right_panel, text="Pooling & Algorithm Settings")
        set_frame.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Label(set_frame, text="Pooling Operator:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.op_combo = ttk.Combobox(set_frame, values=["POPCOUNT", "AND", "OR", "XOR", "MAJORITY"], state="readonly")
        self.op_combo.set("POPCOUNT")
        self.op_combo.grid(row=0, column=1, padx=5, pady=5)
        self.op_combo.bind("<<ComboboxSelected>>", self._update_pool_op)

        ttk.Label(set_frame, text="Kernel Size:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.kernel_spin = ttk.Spinbox(set_frame, from_=2, to=8, increment=2, width=5, command=self._update_kernel)
        self.kernel_spin.set(4)
        self.kernel_spin.grid(row=1, column=1, sticky="w", padx=5, pady=5)

    def start_lab(self):
        self.ctl["stop"] = False
        self.thread = SimThread(0, self.ctl, self.q, self.engine)
        self.thread.start()

    def stop_lab(self):
        self.ctl["stop"] = True

    def step_search(self):
        self.engine.step()

    def reset_search(self):
        self.engine.reset()
        self.v_custom_target.set("[SECURE]")

    def set_custom_target(self):
        try:
            val = int(self.v_custom_target.get())
            if 0 <= val <= self.engine.max_val:
                self.engine.reset(val)
                messagebox.showinfo("Target Set", "New custom target securely configured.")
            else:
                messagebox.showerror("Invalid Range", f"Target must be between 0 and {self.engine.max_val}.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer for the custom target.")

    def randomize_target(self):
        self.engine.reset()
        self.v_custom_target.set("[SECURE]")

    def _update_pool_op(self, event):
        self.engine.pool_op = self.op_combo.get()

    def _update_kernel(self):
        try:
            self.engine.kernel_size = int(self.kernel_spin.get())
        except ValueError:
            pass

    def _draw_bit_rows(self, target_val, guess_val, pooled_val, revealed):
        self.reg_canvas.delete("all")
        bit_depth = self.engine.bit_depth
        
        # Row configurations: (label, value, color, is_hidden)
        rows = [
            ("TARGET", target_val if (revealed or self.engine.converged) else 0, "#00e676", not (revealed or self.engine.converged)),
            ("GUESS ", guess_val, "#ffb300", False),
            ("POOLED", pooled_val, "#b388ff", False)
        ]

        y_offset = 10
        for label, val, color, hidden in rows:
            self.reg_canvas.create_text(10, y_offset + 10, text=label, fill="#88a0c0", anchor="w", font=("Consolas", 10, "bold"))
            
            bx = 90
            for b in range(bit_depth - 1, -1, -1):
                bit_state = (val >> b) & 1
                rect = self.reg_canvas.create_rectangle(bx, y_offset, bx + 22, y_offset + 22, 
                                                        fill=color if (bit_state and not hidden) else "#1e2430", 
                                                        outline="#38445c")
                txt_val = "?" if hidden else str(bit_state)
                txt_color = "#000000" if (bit_state and not hidden) else "#708090"
                self.reg_canvas.create_text(bx + 11, y_offset + 11, text=txt_val, fill=txt_color, font=("Consolas", 9, "bold"))
                bx += 26
            
            # Numeric value display at the end of the row
            num_str = "[HIDDEN]" if hidden else f"[{val:4d}]"
            self.reg_canvas.create_text(bx + 5, y_offset + 11, text=num_str, fill="#d2e0f0", anchor="w", font=("Consolas", 9))
            y_offset += 40

    def _poll(self):
        try:
            for _ in range(100):
                ev = self.q.get_nowait()
                if ev[0] == "frame":
                    d = ev[1]
                    self.photo = ppm_photo(d["preview"])
                    self.cam_canvas.itemconfig(self.cam_img_id, image=self.photo)
                    
                    # Draw visual bit registers on the canvas
                    self._draw_bit_rows(d["target"], d["guess"], d["pooled"], self.v_reveal.get())
                    self.lbl_status.config(text=f"Status: {d['status']} | Steps: {d['steps']}")
                elif ev[0] == "error":
                    messagebox.showerror("Camera Error", ev[1])
        except queue.Empty:
            pass
        self.window.after(30, self._poll)

    def _close(self):
        self.ctl["stop"] = True
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.geometry("860x600")
    root.mainloop()
