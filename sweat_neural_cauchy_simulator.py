#!/usr/bin/env python3
"""
REAL-HARDWARE MOISTURE CAUCHY GUI — ARDUINO UNO / USB SERIAL

Install:
    pip install pyserial numpy matplotlib

Run:
    python moisture_cauchy_arduino_gui.py

Connect:
    Capacitive sensor AO -> Arduino A0
    Sensor GND -> Arduino GND
    Sensor VCC -> rated supply (many modules accept 3.3–5 V; check yours)
    Arduino Uno -> USB

The GUI reads the Arduino CSV stream, converts ADC values into a calibrated
relative moisture proxy, filters it, estimates dM/dt, and calculates:

    dM/dt = P(t) - lambda*M(t)
    P_est(t) = dM_est/dt + lambda*M_est(t)

The displayed energy is a dimensionless signal-deviation score, not physical
energy expenditure. The system detects moisture changes only; it cannot
infer thoughts, cognitive events, or sweat chemistry.

Typical soil-moisture capacitive probes are not designed for direct skin
contact. Do not attach exposed electronics/probe circuitry to skin. Use an
insulated collection interface and treat readings as experimental proxies.
"""

import csv
import queue
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import serial
from serial.tools import list_ports
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk


class MoistureHardwareGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Moisture Cauchy Acquisition — Arduino Uno")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 720)

        self.ser = None
        self.reader_thread = None
        self.stop_event = threading.Event()
        self.data_queue = queue.Queue()
        self.lock = threading.Lock()
        self.rows = []

        self.t0_host = None
        self.first_millis = None
        self.latest_time = 0.0
        self.last_raw = None
        self.connected = False

        self._build_ui()
        self.refresh_ports()
        self.root.after(100, self.process_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        panel = ttk.Frame(outer)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        plots = ttk.Frame(outer)
        plots.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(panel, text="Hardware connection",
                  font=("Segoe UI", 13, "bold")).pack(anchor="w")

        port_row = ttk.Frame(panel)
        port_row.pack(fill=tk.X, pady=(7, 3))
        self.port_var = tk.StringVar()
        self.port_box = ttk.Combobox(port_row, textvariable=self.port_var,
                                     width=15, state="readonly")
        self.port_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(port_row, text="Refresh", command=self.refresh_ports).pack(
            side=tk.LEFT, padx=(4, 0))

        self.connect_button = ttk.Button(panel, text="Connect", command=self.toggle_connection)
        self.connect_button.pack(fill=tk.X, pady=3)
        self.status_var = tk.StringVar(value="Disconnected")
        ttk.Label(panel, textvariable=self.status_var,
                  foreground="#9a4b00").pack(anchor="w", pady=(0, 8))

        ttk.Separator(panel).pack(fill=tk.X, pady=7)
        ttk.Label(panel, text="Calibration",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")

        self.dry_var = tk.IntVar(value=800)
        self.wet_var = tk.IntVar(value=350)
        self._entry(panel, "Dry reference ADC", self.dry_var)
        self._entry(panel, "Wet reference ADC", self.wet_var)
        ttk.Label(panel, text="Take readings in the intended sensor setup.\n"
                              "Enter representative dry and wet ADC values.",
                  wraplength=270, foreground="#555").pack(anchor="w", pady=4)

        ttk.Separator(panel).pack(fill=tk.X, pady=7)
        ttk.Label(panel, text="Cauchy model",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.lambda_var = tk.DoubleVar(value=0.035)
        self.filter_var = tk.DoubleVar(value=1.0)
        self.threshold_var = tk.DoubleVar(value=0.025)
        self.baseline_var = tk.DoubleVar(value=0.0)
        self._entry(panel, "Removal rate λ (1/s)", self.lambda_var)
        self._entry(panel, "Filter time constant (s)", self.filter_var)
        self._entry(panel, "Onset threshold (proxy/s)", self.threshold_var)
        self._entry(panel, "Baseline moisture proxy", self.baseline_var)

        ttk.Separator(panel).pack(fill=tk.X, pady=7)
        ttk.Label(panel, text="Live readout",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.readout = tk.Text(panel, width=36, height=15, wrap=tk.WORD,
                               font=("Consolas", 9), state=tk.DISABLED)
        self.readout.pack(fill=tk.X, pady=5)

        actions = ttk.Frame(panel)
        actions.pack(fill=tk.X, pady=4)
        ttk.Button(actions, text="Clear", command=self.clear_data).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        ttk.Button(actions, text="Save CSV", command=self.save_csv).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=3)

        ttk.Label(panel,
                  text="Safety / interpretation\n"
                       "A common capacitive soil probe is not a medical sweat "
                       "sensor and is not intended for direct skin contact. "
                       "Use an insulated collection setup. Moisture proxy "
                       "values are not sweat chemistry, metabolism, energy "
                       "expenditure, or thought timing.",
                  wraplength=275, justify=tk.LEFT, foreground="#8a3b20").pack(
                      anchor="w", pady=(8, 0))

        self.fig = Figure(figsize=(10, 8), dpi=100, constrained_layout=True)
        self.ax1 = self.fig.add_subplot(411)
        self.ax2 = self.fig.add_subplot(412)
        self.ax3 = self.fig.add_subplot(413)
        self.ax4 = self.fig.add_subplot(414)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plots)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, plots).update()
        self._draw_empty()

    def _entry(self, parent, label, variable):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable, width=9).pack(side=tk.RIGHT)

    def refresh_ports(self):
        ports = [p.device for p in list_ports.comports()]
        self.port_box["values"] = ports
        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])
        if not ports:
            self.port_var.set("")

    def toggle_connection(self):
        if self.connected:
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning("No port", "Select the Arduino Uno serial port.")
            return
        try:
            self.ser = serial.Serial(port, 115200, timeout=1)
            # Opening an Uno serial port normally resets it.
            time.sleep(2.0)
            self.ser.reset_input_buffer()
        except Exception as exc:
            self.ser = None
            messagebox.showerror("Serial connection failed", str(exc))
            return

        self.stop_event.clear()
        self.connected = True
        self.reader_thread = threading.Thread(target=self.serial_reader, daemon=True)
        self.reader_thread.start()
        self.connect_button.config(text="Disconnect")
        self.status_var.set(f"Connected: {port}")
        self.status_var.set(f"Connected: {port}")
        
    def disconnect(self):
        self.stop_event.set()
        self.connected = False
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        self.connect_button.config(text="Connect")
        self.status_var.set("Disconnected")

    def serial_reader(self):
        while not self.stop_event.is_set():
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("MOISTURE") or line.startswith("millis"):
                    continue
                parts = line.split(",")
                if len(parts) != 3:
                    continue
                millis = int(parts[0])
                raw = int(parts[1])
                voltage = float(parts[2])
                if not (0 <= raw <= 1023 and 0.0 <= voltage <= 5.1):
                    continue
                self.data_queue.put((millis, raw, voltage, time.time()))
            except (ValueError, serial.SerialException, OSError) as exc:
                if not self.stop_event.is_set():
                    self.data_queue.put(("ERROR", str(exc)))
                break

    def process_queue(self):
        changed = False
        while True:
            try:
                item = self.data_queue.get_nowait()
            except queue.Empty:
                break
            if item[0] == "ERROR":
                self.status_var.set("Serial read error")
                continue
            millis, raw, voltage, host_time = item
            if self.first_millis is None:
                self.first_millis = millis
                self.t0_host = host_time
            # Use Arduino elapsed millis; handle Uno millis wrap-around.
            elapsed_ms = (millis - self.first_millis) & 0xFFFFFFFF
            t = elapsed_ms / 1000.0
            self.latest_time = t
            self.last_raw = raw
            self.rows.append((t, millis, raw, voltage))
            changed = True

        if changed:
            self.update_plots_and_readout()
        self.root.after(100, self.process_queue)

    def calibrated_moisture(self, raw):
        dry = float(self.dry_var.get())
        wet = float(self.wet_var.get())
        if abs(dry - wet) < 1:
            raise ValueError("Dry and wet calibration ADC values must differ.")
        # Works whether the wet reading is lower or higher than the dry reading.
        return float(np.clip((raw - dry) / (wet - dry), 0.0, 1.0))

    def calculate(self):
        arr = np.asarray(self.rows, dtype=float)
        t, millis, raw, voltage = arr.T
        moisture = np.array([self.calibrated_moisture(v) for v in raw])

        # Remove duplicate/non-increasing timestamps before differentiation.
        keep = np.concatenate(([True], np.diff(t) > 0))
        t, millis, raw, voltage, moisture = (
            a[keep] for a in (t, millis, raw, voltage, moisture)
        )
        if len(t) < 2:
            return None

        dt = np.diff(t)
        dt_med = max(float(np.median(dt)), 1e-3)
        tau = max(float(self.filter_var.get()), 0.0)
        filtered = np.empty_like(moisture)
        filtered[0] = moisture[0]
        for i in range(1, len(moisture)):
            step = max(t[i] - t[i-1], 1e-4)
            alpha = 1.0 if tau <= 0 else step / (tau + step)
            filtered[i] = filtered[i-1] + alpha * (
                moisture[i] - filtered[i-1]
            )

        rate = np.gradient(filtered, t)
        lam = max(0.0, float(self.lambda_var.get()))
        production = np.clip(rate + lam * filtered, 0.0, None)

        baseline = float(self.baseline_var.get())
        delta = filtered - baseline
        energy = 0.5 * (delta ** 2 + 0.2 * production ** 2)
        return {
            "t": t, "millis": millis, "raw": raw, "voltage": voltage,
            "moisture": moisture, "filtered": filtered, "rate": rate,
            "production": production, "energy": energy,
            "median_dt": dt_med
        }

    def update_plots_and_readout(self):
        try:
            data = self.calculate()
        except Exception as exc:
            self.status_var.set(f"Calibration error: {exc}")
            return
        if data is None:
            return
        t = data["t"]
        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.clear()

        self.ax1.plot(t, data["raw"], label="Raw ADC")
        self.ax1.set_title("1. Live capacitive sensor ADC")
        self.ax1.set_ylabel("ADC (0–1023)")
        self.ax1.grid(alpha=0.25)
        self.ax1.legend(fontsize=8)

        self.ax2.plot(t, data["moisture"], label="Calibrated moisture proxy", alpha=0.5)
        self.ax2.plot(t, data["filtered"], label="Filtered moisture proxy", linewidth=1.7)
        self.ax2.set_title("2. Calibrated moisture state")
        self.ax2.set_ylabel("Relative proxy (0–1)")
        self.ax2.set_ylim(-0.05, 1.05)
        self.ax2.grid(alpha=0.25)
        self.ax2.legend(fontsize=8)

        self.ax3.plot(t, data["rate"], label="dM/dt")
        self.ax3.plot(t, data["production"], label="Estimated input P(t)")
        self.ax3.axhline(float(self.threshold_var.get()), linestyle=":",
                         label="Onset threshold")
        self.ax3.set_title("3. Moisture rate and model-reconstructed input")
        self.ax3.set_ylabel("Proxy units/s")
        self.ax3.grid(alpha=0.25)
        self.ax3.legend(fontsize=8)

        self.ax4.plot(t, data["energy"], label="Baseline-relative score")
        self.ax4.set_title("4. Moisture deviation score (dimensionless)")
        self.ax4.set_xlabel("Elapsed time (s)")
        self.ax4.set_ylabel("Score")
        self.ax4.grid(alpha=0.25)
        self.ax4.legend(fontsize=8)

        threshold = float(self.threshold_var.get())
        sustained = data["production"] > threshold
        onset = None
        required = max(1, int(round(1.0 / data["median_dt"])))
        count = 0
        for i, flag in enumerate(sustained):
            count = count + 1 if flag else 0
            if count >= required:
                onset = t[i-required+1]
                break
        if onset is not None:
            for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
                ax.axvline(onset, color="black", linestyle="--", alpha=0.5)

        self.canvas.draw_idle()
        latest = len(t) - 1
        lines = (
            "LIVE HARDWARE READOUT\n"
            "---------------------\n"
            f"Samples:             {len(t)}\n"
            f"Elapsed:             {t[-1]:.2f} s\n"
            f"Latest raw ADC:      {data['raw'][-1]:.0f}\n"
            f"Latest voltage:      {data['voltage'][-1]:.3f} V\n"
            f"Moisture proxy:      {data['filtered'][-1]:.4f}\n"
            f"Moisture rate:       {data['rate'][-1]:+.5f}/s\n"
            f"Estimated input:     {data['production'][-1]:.5f}/s\n"
            f"Energy score:        {data['energy'][-1]:.6f}\n\n"
            f"Detected onset:      {onset:.2f} s\n" if onset is not None else
            "LIVE HARDWARE READOUT\n"
            "---------------------\n"
            f"Samples:             {len(t)}\n"
            f"Elapsed:             {t[-1]:.2f} s\n"
            f"Latest raw ADC:      {data['raw'][-1]:.0f}\n"
            f"Latest voltage:      {data['voltage'][-1]:.3f} V\n"
            f"Moisture proxy:      {data['filtered'][-1]:.4f}\n"
            f"Moisture rate:       {data['rate'][-1]:+.5f}/s\n"
            f"Estimated input:     {data['production'][-1]:.5f}/s\n"
            f"Energy score:        {data['energy'][-1]:.6f}\n\n"
            "Detected onset:      not yet detected\n"
        )
        lines += (
            "\nModel:\n"
            "dM/dt = P(t) - lambda*M(t)\n"
            "P_est = max(0, dM/dt + lambda*M)\n\n"
            "This is a moisture proxy, not a sweat\n"
            "chemistry, metabolic-energy, or thought\n"
            "measurement. Sensor calibration and\n"
            "collection geometry affect the result."
        )
        self.readout.config(state=tk.NORMAL)
        self.readout.delete("1.0", tk.END)
        self.readout.insert(tk.END, lines)
        self.readout.config(state=tk.DISABLED)

    def _draw_empty(self):
        for ax, title in zip(
            (self.ax1, self.ax2, self.ax3, self.ax4),
            ("Raw ADC", "Moisture proxy", "Moisture dynamics", "Energy score")
        ):
            ax.set_title(title)
            ax.grid(alpha=0.2)
        self.canvas.draw_idle()

    def clear_data(self):
        self.rows.clear()
        self.first_millis = None
        self.latest_time = 0.0
        self._draw_empty()
        self.readout.config(state=tk.NORMAL)
        self.readout.delete("1.0", tk.END)
        self.readout.config(state=tk.DISABLED)

    def save_csv(self):
        if not self.rows:
            messagebox.showinfo("No data", "Connect the Arduino and collect data first.")
            return
        try:
            data = self.calculate()
        except Exception as exc:
            messagebox.showerror("Calibration error", str(exc))
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile="live_moisture_cauchy.csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "elapsed_s", "arduino_millis", "raw_adc", "voltage_v",
                "moisture_proxy", "filtered_moisture", "dM_dt",
                "estimated_input_P", "energy_score"
            ])
            for i in range(len(data["t"])):
                writer.writerow([
                    data["t"][i], data["millis"][i], data["raw"][i],
                    data["voltage"][i], data["moisture"][i],
                    data["filtered"][i], data["rate"][i],
                    data["production"][i], data["energy"][i]
                ])
        messagebox.showinfo("Saved", f"Saved {len(data['t'])} samples to:\n{path}")

    def close(self):
        self.disconnect()
        self.root.destroy()


def main():
    root = tk.Tk()
    MoistureHardwareGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
