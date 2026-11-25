#!/usr/bin/env python3
"""
SWEAT–NEURAL CAUCHY SIMULATOR
----------------------------
A synthetic, educational simulator inspired by:
  * Cauchy initial-value problems
  * Keller–Segel-style diffusion / chemotactic transport / logistic growth
  * delayed peripheral biomarker dynamics

IMPORTANT:
This is NOT a medical device and does not infer real thoughts from sweat.
The "neural event" is a known synthetic input. Sweat signals are delayed,
noisy, and nonspecific. All values are illustrative, not clinical ranges.

Requirements:
    pip install numpy matplotlib

Run:
    python sweat_neural_cauchy_simulator.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk


class SweatNeuralSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Sweat–Neural Cauchy Simulator")
        self.root.geometry("1280x850")
        self.root.minsize(1000, 700)

        self.running = False
        self.after_id = None
        self.time = np.array([])
        self.results = None

        self._build_ui()
        self.reset_simulation()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(outer)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        plot_frame = ttk.Frame(outer)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(controls, text="Simulation controls",
                  font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        self.vars = {
            "duration": tk.DoubleVar(value=180.0),
            "dt": tk.DoubleVar(value=0.1),
            "event_time": tk.DoubleVar(value=35.0),
            "event_strength": tk.DoubleVar(value=1.0),
            "diffusion": tk.DoubleVar(value=0.18),
            "chemotaxis": tk.DoubleVar(value=0.35),
            "growth": tk.DoubleVar(value=0.20),
            "sweat_lag": tk.DoubleVar(value=8.0),
            "marker_noise": tk.DoubleVar(value=0.025),
            "initial_neural": tk.DoubleVar(value=0.08),
        }

        self._add_scale(controls, "Duration (s)", "duration", 30, 300, 5)
        self._add_scale(controls, "Neural event time (s)", "event_time", 5, 150, 1)
        self._add_scale(controls, "Neural event strength", "event_strength", 0.1, 2.0, 0.05)
        self._add_scale(controls, "Diffusion coefficient", "diffusion", 0.0, 0.8, 0.01)
        self._add_scale(controls, "Chemotactic coupling", "chemotaxis", 0.0, 1.2, 0.01)
        self._add_scale(controls, "Logistic growth rate", "growth", 0.0, 0.8, 0.01)
        self._add_scale(controls, "Sweat response lag (s)", "sweat_lag", 1, 30, 0.5)
        self._add_scale(controls, "Marker noise", "marker_noise", 0.0, 0.12, 0.005)
        self._add_scale(controls, "Initial neural baseline", "initial_neural", 0.0, 0.5, 0.01)

        buttons = ttk.Frame(controls)
        buttons.pack(fill=tk.X, pady=10)
        ttk.Button(buttons, text="Run", command=self.run_simulation).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(buttons, text="Reset", command=self.reset_simulation).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        ttk.Button(buttons, text="Export CSV", command=self.export_csv).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        ttk.Separator(controls).pack(fill=tk.X, pady=8)
        ttk.Label(controls, text="Readout", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.readout = tk.Text(controls, width=35, height=13, wrap=tk.WORD,
                               font=("Consolas", 9), state=tk.DISABLED)
        self.readout.pack(fill=tk.X, pady=(5, 8))

        note = (
            "Interpretation\n"
            "• Neural activation is a synthetic event input.\n"
            "• Sweat markers are delayed proxy signals.\n"
            "• Energy is a model-state deviation measure.\n"
            "• Changing diffusion to zero tests a limiting case.\n\n"
            "The simulator cannot identify a real person's thoughts "
            "or their exact timing from sweat."
        )
        ttk.Label(controls, text=note, wraplength=260, justify=tk.LEFT,
                  foreground="#555").pack(anchor="w", pady=(2, 0))

        self.fig = Figure(figsize=(9, 7), dpi=100, constrained_layout=True)
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

    def _add_scale(self, parent, label, key, low, high, resolution):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=3)
        top = ttk.Frame(frame)
        top.pack(fill=tk.X)
        ttk.Label(top, text=label).pack(side=tk.LEFT)
        value_label = ttk.Label(top, width=7, anchor="e")
        value_label.pack(side=tk.RIGHT)

        def update_label(value):
            value_label.config(text=f"{float(value):.2f}")

        scale = tk.Scale(
            frame, variable=self.vars[key], from_=low, to=high,
            resolution=resolution, orient=tk.HORIZONTAL, showvalue=False,
            length=265, command=update_label
        )
        scale.pack(fill=tk.X)
        update_label(self.vars[key].get())

    def _simulate(self):
        p = {k: v.get() for k, v in self.vars.items()}
        dt = max(0.05, min(0.25, p["dt"]))
        duration = p["duration"]
        t = np.arange(0.0, duration + dt, dt)
        n = len(t)

        # One-dimensional Keller–Segel-inspired field:
        # u_t = D*u_xx - chi*d_x(u*v_x) + r*u*(1-u/K)
        # v_t = Dv*v_xx + a*u - b*v
        # This reduced PDE is only a toy analogy, not the paper's theorem.
        nx = 41
        x = np.linspace(0.0, 1.0, nx)
        dx = x[1] - x[0]
        u = np.zeros((n, nx), dtype=float)
        chemo = np.zeros((n, nx), dtype=float)
        centre = 0.5
        u[0] = p["initial_neural"] + 0.03 * np.exp(-((x-centre)/0.12)**2)
        chemo[0] = 0.02

        neural = np.zeros(n)
        neural[0] = p["initial_neural"]
        event = np.exp(-0.5 * ((t - p["event_time"]) / 1.8) ** 2)
        event *= p["event_strength"]

        # Peripheral markers are first-order delayed filters of a latent
        # autonomic/metabolic drive, with distinct time constants.
        lactate = np.zeros(n)
        glucose = np.zeros(n)
        cortisol = np.zeros(n)
        eda = np.zeros(n)
        sweat_drive = np.zeros(n)
        lactate[0], glucose[0], cortisol[0], eda[0] = 0.25, 0.55, 0.20, 0.15

        tau_neural = 2.5
        tau_lactate = max(4.0, p["sweat_lag"] * 0.75)
        tau_glucose = max(8.0, p["sweat_lag"] * 1.3)
        tau_cortisol = max(18.0, p["sweat_lag"] * 2.8)
        tau_eda = 1.8
        D = p["diffusion"]
        chi = p["chemotaxis"]
        r = p["growth"]
        Dv = 0.08
        K = 1.0

        for i in range(1, n):
            neural_target = p["initial_neural"] + event[i]
            neural[i] = neural[i-1] + dt * (
                (neural_target - neural[i-1]) / tau_neural
            )
            # Spatially localised synthetic source from neural activation.
            source = 0.12 * neural[i] * np.exp(-((x-centre)/0.16)**2)

            ui = u[i-1]
            vi = chemo[i-1]
            lap_u = np.zeros(nx)
            lap_v = np.zeros(nx)
            lap_u[1:-1] = (ui[2:] - 2*ui[1:-1] + ui[:-2]) / dx**2
            lap_v[1:-1] = (vi[2:] - 2*vi[1:-1] + vi[:-2]) / dx**2
            lap_u[0] = 2*(ui[1]-ui[0])/dx**2
            lap_u[-1] = 2*(ui[-2]-ui[-1])/dx**2
            lap_v[0] = 2*(vi[1]-vi[0])/dx**2
            lap_v[-1] = 2*(vi[-2]-vi[-1])/dx**2

            grad_v = np.gradient(vi, dx)
            flux = ui * grad_v
            chemotactic_div = np.gradient(flux, dx)
            du = D*lap_u - chi*chemotactic_div + r*ui*(1-ui/K) + source
            dv = Dv*lap_v + 0.35*ui - 0.12*vi
            u[i] = np.clip(ui + dt*du, 0.0, 2.0)
            chemo[i] = np.clip(vi + dt*dv, 0.0, 3.0)

            # Peripheral drive uses the spatial mean of the field and event
            # response. It is deliberately not a direct readout of neural state.
            field_activity = np.mean(u[i])
            sweat_drive[i] = 0.65*field_activity + 0.35*neural[i]
            noise = p["marker_noise"] * np.random.normal(size=4)

            def relax(previous, target, tau):
                return previous + dt*(target-previous)/tau

            lactate[i] = max(0.0, relax(
                lactate[i-1], 0.25 + 0.65*sweat_drive[i], tau_lactate
            ) + noise[0])
            glucose[i] = max(0.0, relax(
                glucose[i-1], 0.55 - 0.22*sweat_drive[i], tau_glucose
            ) + noise[1])
            cortisol[i] = max(0.0, relax(
                cortisol[i-1], 0.20 + 0.45*sweat_drive[i], tau_cortisol
            ) + noise[2])
            eda[i] = max(0.0, relax(
                eda[i-1], 0.15 + 0.75*neural[i], tau_eda
            ) + noise[3])

        # Weighted energy of deviation from baseline across the field and markers.
        baseline = np.array([0.25, 0.55, 0.20, 0.15])
        marker_matrix = np.column_stack([lactate, glucose, cortisol, eda])
        weights = np.diag([1.0, 0.7, 0.8, 0.9])
        marker_delta = marker_matrix - baseline
        marker_energy = 0.5*np.einsum(
            "ij,jk,ik->i", marker_delta, weights, marker_delta
        )
        field_energy = 0.5*np.mean((u - p["initial_neural"])**2, axis=1)
        energy = marker_energy + field_energy

        self.time = t
        self.results = {
            "t": t, "x": x, "u": u, "chemo": chemo,
            "neural": neural, "event": event, "lactate": lactate,
            "glucose": glucose, "cortisol": cortisol, "eda": eda,
            "energy": energy, "marker_energy": marker_energy,
            "field_energy": field_energy, "sweat_drive": sweat_drive,
        }

    def reset_simulation(self):
        self.pause()
        try:
            self._simulate()
            self._draw()
            self._update_readout()
        except Exception as exc:
            messagebox.showerror("Simulation error", str(exc))

    def run_simulation(self):
        self.pause()
        try:
            self._simulate()
            self._draw()
            self._update_readout()
        except Exception as exc:
            messagebox.showerror("Simulation error", str(exc))

    def pause(self):
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _draw(self):
        r = self.results
        t = r["t"]
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        self.ax1.plot(t, r["neural"], label="Synthetic neural activation", linewidth=1.8)
        self.ax1.plot(t, r["event"], label="Known simulated event input",
                      linestyle="--", alpha=0.8)
        self.ax1.set_ylabel("Neural proxy")
        self.ax1.set_title("1. Latent event and neural-response proxy")
        self.ax1.legend(loc="upper right", fontsize=8)
        self.ax1.grid(alpha=0.25)

        self.ax2.plot(t, r["lactate"], label="Lactate proxy")
        self.ax2.plot(t, r["glucose"], label="Glucose proxy")
        self.ax2.plot(t, r["cortisol"], label="Cortisol proxy")
        self.ax2.plot(t, r["eda"], label="Electrodermal activity proxy")
        self.ax2.set_ylabel("Arbitrary units")
        self.ax2.set_title("2. Delayed peripheral signals (synthetic)")
        self.ax2.legend(loc="upper right", fontsize=8, ncol=2)
        self.ax2.grid(alpha=0.25)

        self.ax3.plot(t, r["energy"], label="Total model energy", linewidth=1.8)
        self.ax3.plot(t, r["marker_energy"], label="Marker contribution", alpha=0.8)
        self.ax3.plot(t, r["field_energy"], label="Spatial-field contribution", alpha=0.8)
        self.ax3.set_xlabel("Time (seconds)")
        self.ax3.set_ylabel("Energy functional")
        self.ax3.set_title("3. Weighted deviation from baseline")
        self.ax3.legend(loc="upper right", fontsize=8)
        self.ax3.grid(alpha=0.25)
        self.canvas.draw_idle()

    def _update_readout(self):
        r = self.results
        t = r["t"]
        event_t = self.vars["event_time"].get()
        idx = int(np.argmin(np.abs(t-event_t)))
        peak_idx = int(np.argmax(r["energy"]))
        lag = t[peak_idx] - event_t
        text = (
            "Synthetic run summary\n"
            "---------------------\n"
            f"Input event time: {event_t:7.2f} s\n"
            f"Event amplitude:  {self.vars['event_strength'].get():7.2f}\n"
            f"Energy at event:  {r['energy'][idx]:7.4f}\n"
            f"Peak energy time: {t[peak_idx]:7.2f} s\n"
            f"Peak-event offset: {lag:7.2f} s\n\n"
            "Final marker values\n"
            f"Lactate proxy:    {r['lactate'][-1]:7.3f}\n"
            f"Glucose proxy:    {r['glucose'][-1]:7.3f}\n"
            f"Cortisol proxy:   {r['cortisol'][-1]:7.3f}\n"
            f"EDA proxy:        {r['eda'][-1]:7.3f}\n\n"
            "Model parameters\n"
            f"Diffusion D:      {self.vars['diffusion'].get():7.3f}\n"
            f"Chemotaxis chi:   {self.vars['chemotaxis'].get():7.3f}\n"
            f"Growth r:         {self.vars['growth'].get():7.3f}\n\n"
            "Peak-energy offset is NOT a recovered\n"
            "thought time. It is a property of this\n"
            "synthetic model and its chosen delays."
        )
        self.readout.config(state=tk.NORMAL)
        self.readout.delete("1.0", tk.END)
        self.readout.insert(tk.END, text)
        self.readout.config(state=tk.DISABLED)

    def export_csv(self):
        if self.results is None:
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="sweat_neural_simulation.csv"
        )
        if not path:
            return
        r = self.results
        data = np.column_stack((
            r["t"], r["event"], r["neural"], r["lactate"],
            r["glucose"], r["cortisol"], r["eda"], r["energy"],
            r["marker_energy"], r["field_energy"], r["sweat_drive"]
        ))
        header = (
            "time_s,event_input,neural_proxy,lactate_proxy,glucose_proxy,"
            "cortisol_proxy,eda_proxy,total_energy,marker_energy,"
            "field_energy,sweat_drive"
        )
        np.savetxt(path, data, delimiter=",", header=header, comments="")
        messagebox.showinfo("Export complete", f"Saved simulation data to:\n{path}")


def main():
    root = tk.Tk()
    app = SweatNeuralSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
