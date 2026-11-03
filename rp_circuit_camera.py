"""
RP-style optoelectronic decision circuit driven by a USB camera  (single file)

    pip install numpy opencv-python
    python rp_circuit_camera.py

Every camera frame is one trial of the circuit:

  1. Motion detection: the frame is shrunk to 128x96, blurred and differenced
     against the previous frame. Pixels that changed by more than a threshold
     form the motion mask.
  2. Instance x = number of moving pixels. (No motion -> no instance, Q = 0.)
     The language is L = { odd composite numbers }.
  3. Driver A (randomized witness test, Detector A = motion energy):
        the same camera frame feeds both detectors and no sensor noise is
        used. r is derived from the motion pattern itself (the size of the
        change at every moving pixel, hashed with SHA-256), so it is
        unpredictable the way the movement is. A = 1 if r is a Miller-Rabin
        witness that x is composite. A = 0 for every prime, so the error is
        one-sided.
  4. Driver B (deterministic check, Detector B = motion mask):
        c(x) = "x is odd and > 2". It drives capacitor C through R for one
        frame period; the NPN emitter follower gives V(B) = V(C) - 0.7 V and
        B is a valid logic HIGH at >= 3.5 V.
  5. Gates: OR and AND (tied outputs read as AND), 10 uF node (low-pass),
     time code Y (which frames may pass), 74HC74 latches D on the clock edge.
     Here the clock edge is the end of each camera frame.
  6. SELECT = "any Q = 1 in the last k frames" (the repetition step).

Driver analysis (button "Driver analysis"): every instance frame is scored against
the truth (is x really an odd composite?). For Driver A, Driver B and the final Q
it reports TPR/FPR/PPV/NPV/accuracy and mutual information with "x in L", checks
the RP properties the bounds rest on (A never fires outside L, A's witness rate
p >= 1/2, B never misses a member of L), calibrates A against the exact witness
density of every x, and tests the independence the (1-p)^N bound assumes.

Time code Y (proved in time_code_proof_check.py, shown live in the window):
  * Gating reading, D = A.B.Y: x not in L is never accepted for ANY Y (Thm 1).
    For x in L the false-reject probability is exactly (1-p)^N, N = open bins
    in the k-frame window (Thm 2), so Y sets the error exponent, not the sidedness.
  * Y accepts arithmetic: "1000|0011" OR-adds codes (N never drops, disjoint
    codes multiply their errors), "1111&1011" ANDs them (N never rises),
    "1011+0001" adds them as binary numbers: popcount(a+b) = popcount(a) +
    popcount(b) - carries, so each carry loses one open bin (Thm 3).
  * "forces HIGH" read as OR (checkbox) lets a forced bin accept any x, so the
    one-sided guarantee is lost (Thm 4). Use it to watch false accepts appear.

Concurrency: the camera read runs in a worker thread; Driver A and Driver B
process each frame concurrently as asyncio tasks; the flip-flop stage runs
once both are finished. The whole pipeline lives in a background thread with
its own asyncio loop and talks to Tk through a queue.Queue.

Needs a real USB camera and OpenCV: motion comes only from what the camera
sees. There is no simulated source; if the camera can't be opened, it says so.
"""
import asyncio
import hashlib
import math
import queue
import sys
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from tkinter import ttk, messagebox

import numpy as np

try:
    import cv2
except ImportError:                      # reported when you press Start
    cv2 = None

# ------------------------------------------------------------------ constants
SW, SH = 128, 96                  # processing size
PW, PH = 288, 216                 # live preview size
WARMUP = 8                        # frames ignored while auto-exposure settles
VS = 5.0                          # driver B output voltage
V_BE = 0.7                        # NPN turn-on
V_IH = 3.5                        # 74HC input-high at 5 V
RC_B = 0.004                      # s, IF/THEN link RC (must settle within a frame)
TAU_NODE = 0.004                  # s, 10 uF node low-pass
LOGIC_TIME = RC_B * math.log(VS / (VS - V_IH - V_BE))   # 1.83 RC for these values


# ------------------------------------------------------------------ maths
def miller_rabin_witness(a: int, n: int) -> bool:
    """True if a proves n composite. Never True for a prime n."""
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    y = pow(a, d, n)
    if y in (1, n - 1):
        return False
    for _ in range(s - 1):
        y = y * y % n
        if y == n - 1:
            return False
    return True


def is_odd_composite(x: int) -> bool:
    if x < 9 or x % 2 == 0:
        return False
    return any(x % d == 0 for d in range(3, int(x ** 0.5) + 1, 2))


# ------------------------------------------------------------------ time-code maths
def carries(a: int, b: int, w: int) -> int:
    """Number of carries (including the carry out of the top) when adding a + b."""
    c = n = 0
    for i in range(w):
        t = ((a >> i) & 1) + ((b >> i) & 1) + c
        c = t >> 1
        n += c
    return n


def parse_code(expr: str):
    """Evaluate a time-code expression: '1011', '1000|0011', '1111&1011', '1011+0001'.

    Operands are bit strings read as binary numbers (written order = bin order).
    |  OR-addition   &  AND   +  integer addition (wraps at the code width).
    Returns (Y, info). Raises ValueError on a malformed expression.
    """
    e = expr.replace(" ", "")
    toks, cur = [], ""
    for ch in e:
        if ch in "|&+":
            toks += [cur, ch]
            cur = ""
        else:
            cur += ch
    toks.append(cur)
    operands, ops = toks[0::2], toks[1::2]
    for o in operands:
        if not o or set(o) - {"0", "1"}:
            raise ValueError(f"bad operand '{o}'")
    W = max(len(o) for o in operands)
    mask = (1 << W) - 1
    val, car, wrapped = int(operands[0], 2), 0, 0
    for op, o in zip(ops, operands[1:]):
        b = int(o, 2)
        if op == "|":
            val |= b
        elif op == "&":
            val &= b
        else:
            car += carries(val, b, W)
            wrapped += int(val + b > mask)
            val = (val + b) & mask
    return format(val, f"0{W}b"), dict(carries=car, wrapped=wrapped, expr=e)


def open_in_window(Y: str, k: int, start: int) -> int:
    return sum(int(Y[(start + t) % len(Y)]) for t in range(k))


def window_stats(Y: str, k: int):
    """(worst, best) number of open bins over every k-frame window of the periodic code."""
    counts = [open_in_window(Y, k, s) for s in range(len(Y))]
    return min(counts), max(counts)


def frames_needed(Y: str, n_open: int):
    """Smallest k whose worst-phase window holds n_open open bins (None if Y is all 0)."""
    if "1" not in Y:
        return None
    k = n_open
    while window_stats(Y, k)[0] < n_open:
        k += 1
    return k


# ------------------------------------------------------------------ driver analysis
_WD: dict = {}


def witness_density(x: int) -> float:
    """Exact fraction of r in [2, x-2] that are Miller-Rabin witnesses for x
    (0 for primes, even numbers and x < 5). This is Driver A's true success
    probability on instance x. Rabin: >= 3/4 for every odd composite."""
    if x not in _WD:
        if x < 5 or x % 2 == 0:
            _WD[x] = 0.0
        else:
            _WD[x] = sum(miller_rabin_witness(a, x) for a in range(2, x - 1)) / (x - 3)
    return _WD[x]


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mutual_info(cm: dict) -> float:
    """I(driver output ; x in L) in bits from a 2x2 confusion matrix."""
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    N = tp + fp + fn + tn
    if N == 0:
        return 0.0
    pl = (tp + fn) / N
    pd = (tp + fp) / N
    mi = 0.0
    for cnt, d, l in ((tp, pd, pl), (fp, pd, 1 - pl), (fn, 1 - pd, pl), (tn, 1 - pd, 1 - pl)):
        if cnt and d > 0 and l > 0:
            mi += cnt / N * math.log2((cnt / N) / (d * l))
    return mi


def _div(a, b):
    return a / b if b else None


def _f(v, w=5):
    return " " * (w - 1) + "-" if v is None else f"{v:{w}.2f}"


class DriverAnalysis:
    """Predictive properties of Driver A, Driver B and the final Q, scored on
    every instance frame against the ground truth x in L."""

    NAMES = (("A", "Driver A (random)"), ("B", "Driver B (determ.)"), ("Q", "Q  (A.B.Y latch)"))

    def __init__(self):
        self.reset()

    def reset(self):
        self.cm = {n: dict(tp=0, fp=0, fn=0, tn=0) for n, _ in self.NAMES}
        self.n_noinst = 0
        self.link_miss = 0                  # c(x)=1 but B stayed LOW (timing)
        self.q_checked = self.q_mismatch = 0
        self.a_seq, self.w_seq, self.r_seq = [], [], []     # in-L frames, in order
        self.min_w = 1.0

    def update(self, rec: dict, w: float):
        A, B, Q, y = rec["A"], rec["B"], rec["Q"], rec["y_bin"]
        pred = ((A & B) | y) if rec["force"] else (A & B & y)    # what Q should be
        self.q_checked += 1
        self.q_mismatch += int(pred != Q)
        if not rec["motion"]:
            self.n_noinst += 1
            return
        pos = rec["in_L"]
        for name, val in (("A", A), ("B", B), ("Q", Q)):
            key = ("tp" if val else "fn") if pos else ("fp" if val else "tn")
            self.cm[name][key] += 1
        if rec["c_of_x"] and not B:
            self.link_miss += 1
        if pos:
            self.a_seq.append(A)
            self.w_seq.append(w)
            self.r_seq.append(rec["r"])
            self.min_w = min(self.min_w, w)
            if len(self.a_seq) > 4000:
                del self.a_seq[:1000], self.w_seq[:1000], self.r_seq[:1000]

    def lines(self, k: int, force: bool = False):
        a = self.cm["A"]
        n_in = a["tp"] + a["fn"]
        n_out = a["fp"] + a["tn"]
        n = n_in + n_out
        L = [f"DRIVER ANALYSIS   instances n={n}  (x in L: {n_in}, not in L: {n_out});"
             f" no-motion frames: {self.n_noinst}"]
        if n == 0:
            return L + ["waiting for motion ..."]
        base = n_in / n
        H = -sum(q * math.log2(q) for q in (base, 1 - base) if q > 0)
        L.append(f"truth: P(x in L)={base:.2f}   H(L)={H:.2f} bit   (L = odd composite)")
        L.append("")
        L.append(f"{'':20s}{'TPR':>6}{'FPR':>6}{'PPV':>6}{'NPV':>6}{'acc':>6}{'I(D;L)':>9}")
        for name, label in self.NAMES:
            c = self.cm[name]
            tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
            L.append(f"{label:20s}{_f(_div(tp, tp + fn), 6)}{_f(_div(fp, fp + tn), 6)}"
                     f"{_f(_div(tp, tp + fp), 6)}{_f(_div(tn, tn + fn), 6)}"
                     f"{_f(_div(tp + tn, n), 6)}{mutual_info(c):7.2f} b")
        L.append("")
        L.append("RP properties the bounds rely on")
        fa = a["fp"]
        lo, hi = wilson(fa, n_out)
        L.append(f" A fires only inside L (FP=0): {'PASS' if fa == 0 else 'FAIL'}   "
                 f"{fa}/{n_out}, FPR <= {hi:.1%} (95%)")
        lo, hi = wilson(a["tp"], n_in)
        verdict = "PASS" if lo >= 0.5 else ("FAIL" if hi < 0.5 else "inconclusive")
        L.append(f" A witness rate p >= 1/2     : {verdict}   p={_div(a['tp'], n_in) or 0:.3f}"
                 f"  [{lo:.3f}, {hi:.3f}]")
        b = self.cm["B"]
        L.append(f" B never misses L (FN=0)     : {'PASS' if b['fn'] == 0 else 'FAIL'}   "
                 f"{b['fn']}/{n_in}   link timing misses: {self.link_miss}")
        q = self.cm["Q"]
        L.append(f" Q one-sided (FP=0)          : {'PASS' if q['fp'] == 0 else 'FAIL'}   "
                 f"{q['fp']}/{n_out}" + ("   (force mode breaks it, Thm 4)" if force else ""))
        L.append("")
        L.append("Calibration of A (exact witness density w(x) of each member x)")
        m = len(self.a_seq)
        if m >= 10:
            obs = sum(self.a_seq)
            exp = sum(self.w_seq)
            var = sum(w * (1 - w) for w in self.w_seq)
            z = (obs - exp) / math.sqrt(var) if var > 0 else None
            tag = ("n/a" if z is None else ("calibrated" if abs(z) < 2.5 else "MISCALIBRATED"))
            L.append(f" predicted mean {exp / m:.3f}   observed {obs / m:.3f}"
                     f"   z={'-' if z is None else f'{z:+.1f}'}   {tag}")
            L.append(f" lowest density seen {self.min_w:.3f}   "
                     f"(Rabin bound 0.750: {'ok' if self.min_w >= 0.75 else 'VIOLATED'})")
        else:
            L.append(" need >= 10 members of L")
        L.append("")
        L.append("Independence of A across trials (assumed by (1-p)^N)")
        if m >= 10:
            pairs = list(zip(self.a_seq, self.a_seq[1:]))
            n11 = sum(1 for u, v in pairs if u and v)
            n10 = sum(1 for u, v in pairs if u and not v)
            n01 = sum(1 for u, v in pairs if not u and v)
            n00 = sum(1 for u, v in pairs if not u and not v)
            den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
            phi = (n11 * n00 - n10 * n01) / den if den else None
            L.append(f" lag-1 phi of A on successive members: "
                     f"{'n/a (A nearly constant)' if phi is None else f'{phi:+.2f}'}")
            L.append(f" distinct witnesses r: {len(set(self.r_seq)) / m:.0%} of members")
            nb = m // k
            if nb >= 1:
                zeros = exp_z = 0.0
                for i in range(nb):
                    seg = range(i * k, (i + 1) * k)
                    zeros += int(not any(self.a_seq[j] for j in seg))
                    pr = 1.0
                    for j in seg:
                        pr *= 1 - self.w_seq[j]
                    exp_z += pr
                L.append(f" blocks of k={k} members with no witness: observed {int(zeros)}, "
                         f"expected {exp_z:.2f}  (of {nb})")
        else:
            L.append(" need >= 10 members of L")
        L.append("")
        L.append(f"Logic check  Q == " + ("(A.B) OR Y" if force else "A.B.Y") +
                 f" : {self.q_mismatch} mismatches in {self.q_checked} frames")
        return L


# ------------------------------------------------------------------ frame source
class Source:
    """Real USB camera via OpenCV."""

    def __init__(self, index: int):
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed.  Run: pip install opencv-python")
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not (cap.isOpened() and cap.read()[0]):
            cap.release()
            raise RuntimeError(
                f"Cannot open camera {index}. Check that it is connected, not in use "
                f"by another app, and that the camera number is right.")
        self.cap = cap
        self.rgb = None                  # latest colour frame (for the preview)

    def read(self):
        """Return a full-resolution uint8 gray frame, or None."""
        ok, bgr = self.cap.read()
        if not ok:
            return None
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def close(self):
        self.cap.release()


def downscale(gray: np.ndarray) -> np.ndarray:
    """Shrink and blur so sensor grain does not count as motion."""
    small = cv2.resize(gray, (SW, SH), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0)


def make_preview(rgb: np.ndarray, mask: np.ndarray, overlay: bool) -> np.ndarray:
    """Colour camera frame at PW x PH, optionally with the motion mask in red."""
    h, w = rgb.shape[:2]
    pv = rgb[np.arange(PH) * h // PH][:, np.arange(PW) * w // PW].copy()
    if overlay and mask.any():
        big = mask[np.arange(PH) * SH // PH][:, np.arange(PW) * SW // PW]
        pv[big] = (pv[big] * 0.4 + np.array([255, 40, 40]) * 0.6).astype(np.uint8)
    return pv


# ------------------------------------------------------------------ pipeline
@dataclass
class Packet:
    idx: int
    dt: float
    x: int
    motion: bool
    rand_int: int


@dataclass
class St:
    A: int = 0
    r: int = 0
    B: int = 0
    c_of_x: int = 0
    vC: float = 0.0
    vB: float = 0.0
    traj: list = field(default_factory=list)
    node: float = 0.0
    orv: int = 0
    andv: int = 0
    gate: int = 0
    y_bin: int = 1
    Q: int = 0


class SimThread(threading.Thread):
    def __init__(self, cam_index, ctl, out):
        super().__init__(daemon=True)
        self.cam_index, self.ctl, self.out = cam_index, ctl, out

    def run(self):
        asyncio.run(self._run())

    async def _run(self):
        ctl, out = self.ctl, self.out
        try:
            src = Source(self.cam_index)
        except RuntimeError as e:
            out.put(("error", str(e)))
            out.put(("done", None))
            return
        out.put(("info", f"USB camera {self.cam_index} opened"))
        st = St()
        qa, qb, qc = asyncio.Queue(), asyncio.Queue(), asyncio.Queue()
        beta_tau = TAU_NODE

        async def driver_a():                          # randomized witness test
            while True:
                p = await qa.get()
                if p is None:
                    qa.task_done()
                    return
                st.r, st.A = 0, 0
                if p.motion and p.x > 4 and p.x % 2 == 1:
                    st.r = 2 + p.rand_int % (p.x - 3)           # r in [2, x-2]
                    st.A = int(miller_rabin_witness(st.r, p.x))
                qa.task_done()

        async def driver_b():                          # deterministic check + link
            while True:
                p = await qb.get()
                if p is None:
                    qb.task_done()
                    return
                st.c_of_x = int(p.motion and p.x > 2 and p.x % 2 == 1)
                traj, n = [], 24
                for j in range(n + 1):                 # C charges from 0 each frame
                    t = p.dt * j / n
                    vC = VS * (1 - math.exp(-t / RC_B)) if st.c_of_x else 0.0
                    traj.append((t * 1000, vC, max(0.0, vC - V_BE)))
                st.traj = traj
                st.vC, st.vB = traj[-1][1], traj[-1][2]
                st.B = int(st.vB >= V_IH)
                qb.task_done()

        async def clock_ff():                          # gates, node, Y, 74HC74
            while True:
                p = await qc.get()
                if p is None:
                    qc.task_done()
                    return
                st.orv = st.A | st.B
                st.andv = st.A & st.B
                st.gate = st.orv & st.andv                       # tied -> AND
                st.node = st.gate * (1 - math.exp(-p.dt / beta_tau))   # 10 uF node
                y = ctl["y"]
                st.y_bin = int(y[p.idx % len(y)])
                d_in = int(st.node > 0.5)
                if ctl.get("force"):                             # Y OR-ed in (Thm 4 demo)
                    st.Q = d_in | st.y_bin
                else:                                            # Y gates the result
                    st.Q = d_in & st.y_bin                       # latch on frame edge
                qc.task_done()

        qn = asyncio.Queue()
        AN = DriverAnalysis()

        async def analyst():                           # runs beside the drivers
            seen = 0
            while True:
                rec = await qn.get()
                if rec is None:
                    out.put(("analysis", AN.lines(ctl["k"], ctl.get("force", False))))
                    return
                if ctl.get("reset_an"):
                    AN.reset()
                    ctl["reset_an"] = False
                w = 0.0
                if rec["in_L"]:                        # exact density, off the loop
                    w = _WD.get(rec["x"])
                    if w is None:
                        w = await asyncio.to_thread(witness_density, rec["x"])
                AN.update(rec, w)
                seen += 1
                if seen % 5 == 0 or qn.empty():
                    out.put(("analysis", AN.lines(ctl["k"], rec["force"])))

        tasks = [asyncio.create_task(f()) for f in (driver_a, driver_b, clock_ff, analyst)]

        prev = None
        t_prev = time.perf_counter()
        idx = 0
        recent_q = deque(maxlen=60)
        S = dict(frames=0, motion=0, in_L=0, acc_in_L=0, false_acc=0, accepts=0,
                 open_inL=0, acc_open_inL=0)
        try:
            while not ctl["stop"]:
                gray = await asyncio.to_thread(src.read)
                if gray is None:
                    out.put(("info", "camera stopped delivering frames"))
                    break
                now = time.perf_counter()
                dt = min(max(now - t_prev, 1e-3), 0.25)
                t_prev = now

                # ---- motion detection
                small = downscale(gray)
                if prev is None or idx < WARMUP:
                    diff = np.zeros((SH, SW), np.int16)
                    mask = np.zeros((SH, SW), bool)
                else:
                    diff = np.abs(small.astype(np.int16) - prev.astype(np.int16))
                    mask = diff > ctl["thr"]
                prev = small
                count = int(mask.sum())
                motion = count >= ctl["min_px"]
                x = count if motion else 0

                # ---- witness source: the motion pattern (same camera, no noise)
                energy = np.where(mask, np.clip(diff, 0, 255), 0).astype(np.uint8)
                digest = hashlib.sha256(energy.tobytes()).digest()
                pkt = Packet(idx, dt, x, motion, int.from_bytes(digest, "big"))

                # ---- drivers A and B on this frame, concurrently; then flip-flop
                run_circuit = ctl.get("circuit", True)
                if run_circuit:
                    await qa.put(pkt)
                    await qb.put(pkt)
                    await asyncio.gather(qa.join(), qb.join())
                    await qc.put(pkt)
                    await qc.join()
                else:                                  # preview only: circuit idle
                    for name, val in vars(St()).items():
                        setattr(st, name, val)

                # ---- bookkeeping
                in_L = bool(motion and is_odd_composite(x))
                select = 0
                if run_circuit:
                    recent_q.append(st.Q)
                    select = int(any(list(recent_q)[-ctl["k"]:]))
                    S["frames"] += 1
                    if st.Q and not in_L:
                        S["false_acc"] += 1                # x not in L was accepted
                    if motion:
                        S["motion"] += 1
                        if in_L:
                            S["in_L"] += 1
                            S["acc_in_L"] += st.Q
                            if st.y_bin:                   # frames where Y is open
                                S["open_inL"] += 1
                                S["acc_open_inL"] += st.Q
                        S["accepts"] += st.Q

                if run_circuit:
                    qn.put_nowait(dict(x=x, motion=motion, in_L=in_L, A=st.A, B=st.B,
                                       c_of_x=st.c_of_x, Q=st.Q, y_bin=st.y_bin,
                                       r=st.r, force=bool(ctl.get("force"))))
                rgb = src.rgb if src.rgb is not None else np.stack([gray] * 3, axis=-1)
                preview = make_preview(rgb, mask, ctl.get("overlay", True))
                out.put(("frame", dict(
                    circuit=run_circuit, preview=preview, idx=idx, dt=dt, x=x, motion=motion, in_L=in_L, small=small,
                    mask=mask, energy=energy, A=st.A, r=st.r, B=st.B,
                    c_of_x=st.c_of_x, vC=st.vC, vB=st.vB, traj=st.traj,
                    node=st.node, orv=st.orv, andv=st.andv, gate=st.gate,
                    y_bin=st.y_bin, Q=st.Q, select=select, stats=dict(S))))
                idx += 1
        finally:
            for q in (qa, qb, qc, qn):
                await q.put(None)
            await asyncio.gather(*tasks)
            src.close()
            out.put(("done", None))


# ------------------------------------------------------------------ gui
OFF, ON, GOOD = "#f1efe8", "#ffd9a8", "#b9e6b9"
W_OFF, W_ON = "#999999", "#d9480f"


def ppm_photo(rgb: np.ndarray) -> tk.PhotoImage:
    h, w, _ = rgb.shape
    data = b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(rgb, np.uint8).tobytes()
    return tk.PhotoImage(data=data, format="PPM")


class App:
    def __init__(self, root):
        self.root = root
        root.title("RP-style circuit driven by USB camera motion")
        self.q = queue.Queue()
        self.thread = None
        self.ctl = dict(stop=False, thr=18, min_px=30, k=5, y="1111", circuit=True,
                        overlay=True, force=False)
        self.photos = [None, None, None]
        self.q_hist = deque(maxlen=84)          # (Q, y_bin, motion)
        self.x_hist = deque(maxlen=150)
        self.last_traj = []
        self.err = False
        self.an_win = None
        self.an_lines = ["Press Start circuit, then move in front of the camera."]
        self.code_info = dict(carries=0, wrapped=0, expr="1111")
        self.y_ok = True

        self._controls()
        self.c = tk.Canvas(root, width=710, height=410, bg="white",
                           highlightthickness=1, highlightbackground="#bbb")
        self.c.grid(row=1, column=0, padx=6, pady=4)
        self.boxes, self.wires = {}, {}
        self._diagram()
        self._right_panel()
        bottom = ttk.Frame(root)
        bottom.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        self.cam = tk.Canvas(bottom, width=PW, height=PH, bg="black",
                             highlightthickness=1, highlightbackground="#bbb")
        self.cam.pack(side="left")
        self.cam_img = self.cam.create_image(0, 0, anchor="nw")
        self.cam_text = self.cam.create_text(
            8, 8, anchor="nw", fill="#ffd54a", font=("Consolas", 10, "bold"),
            text="Live camera preview\n(press Preview only or Start circuit)")
        self.cam_photo = None
        self.analysis = ttk.Label(bottom, text="", font=("Consolas", 8), justify="left",
                                  wraplength=330)
        self.analysis.pack(side="left", anchor="n", padx=8)
        self.log = tk.Text(bottom, height=14, font=("Consolas", 9), state="disabled")
        self.log.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self._y_changed()
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(15, self._poll)

    # ---------------------------------------------------------- controls
    def _controls(self):
        outer = ttk.Frame(self.root)
        outer.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        f = ttk.Frame(outer)
        f.pack(anchor="w")
        f2 = ttk.Frame(outer)
        f2.pack(anchor="w", pady=(5, 0))
        ttk.Label(f, text="Camera:").pack(side="left")
        self.s_cam = ttk.Spinbox(f, from_=0, to=9, width=3)
        self.s_cam.set(0)
        self.s_cam.pack(side="left", padx=(2, 12))

        def scale(label, key, lo, hi, init):
            ttk.Label(f, text=label).pack(side="left")
            s = ttk.Scale(f, from_=lo, to=hi, value=init, length=90,
                          command=lambda v, key=key: (self.ctl.update({key: int(float(v))}),
                                                      self._refresh_analysis()))
            s.pack(side="left", padx=(2, 10))
        scale("pixel diff >", "thr", 4, 60, 18)
        scale("min moving px", "min_px", 5, 800, 30)
        scale("k frames", "k", 1, 30, 5)

        ttk.Label(f2, text="Time code Y:").pack(side="left")
        self.e_y = ttk.Entry(f2, width=22)
        self.e_y.insert(0, "1111")
        self.e_y.pack(side="left", padx=(2, 6))
        self.e_y.bind("<KeyRelease>", self._y_changed)
        self.l_y = ttk.Label(f2, text="", font=("Consolas", 9), width=26)
        self.l_y.pack(side="left")
        ttk.Label(f2, text="ops:  |  OR-add    &  AND    +  integer add",
                  foreground="#666").pack(side="left", padx=(0, 10))
        self.v_force = tk.BooleanVar(value=False)
        ttk.Checkbutton(f2, text="'forces HIGH' as OR (breaks one-sidedness)",
                        variable=self.v_force,
                        command=lambda: (self.ctl.update(force=self.v_force.get()),
                                         self._refresh_analysis())).pack(side="left")
        self.b_start = ttk.Button(f, text="Start circuit", command=lambda: self.start(True))
        self.b_start.pack(side="left", padx=2)
        self.b_prev = ttk.Button(f, text="Preview only", command=lambda: self.start(False))
        self.b_prev.pack(side="left", padx=2)
        self.b_stop = ttk.Button(f, text="Stop", command=self.stop, state="disabled")
        self.b_stop.pack(side="left", padx=2)
        self.b_an = ttk.Button(f, text="Driver analysis", command=self._open_analysis)
        self.b_an.pack(side="left", padx=(10, 2))
        self.v_ov = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="show motion", variable=self.v_ov,
                        command=lambda: self.ctl.update(overlay=self.v_ov.get())
                        ).pack(side="left", padx=8)

    def _y_changed(self, _e=None):
        try:
            y, info = parse_code(self.e_y.get())
        except ValueError as err:
            self.l_y.config(text=f"invalid: {err}", foreground="#c33")
            self.y_ok = False
            return
        self.y_ok = True
        self.ctl["y"] = y
        self.code_info = info
        note = ""
        if info["carries"]:
            note = f"  ({info['carries']} carries" + (", top wrapped)" if info["wrapped"] else ")")
        self.l_y.config(text=f"= {y}{note}", foreground="#222")
        self._refresh_analysis()

    def _open_analysis(self):
        if self.an_win is not None:
            self.an_win.lift()
            return
        w = tk.Toplevel(self.root)
        w.title("Driver analysis: predictive properties of A, B and Q")
        self.an_label = ttk.Label(w, text="", font=("Consolas", 9), justify="left")
        self.an_label.pack(padx=10, pady=8, anchor="w")
        ttk.Button(w, text="Reset counters",
                   command=lambda: self.ctl.update(reset_an=True)).pack(pady=(0, 8))
        self.an_win = w
        w.protocol("WM_DELETE_WINDOW", self._close_analysis)
        self._show_analysis()

    def _close_analysis(self):
        self.an_win.destroy()
        self.an_win = None

    def _show_analysis(self):
        if self.an_win is not None:
            self.an_label.config(text="\n".join(self.an_lines))

    def _refresh_analysis(self, S=None):
        """Show what the time code does to the error bound (Thm 1-4)."""
        if not hasattr(self, "analysis"):
            return
        Y, k = self.ctl["y"], self.ctl["k"]
        m, ones = len(Y), Y.count("1")
        L = [f"Time code Y = {Y}   open {ones}/{m}  (density {ones / m:.2f})"]
        ci = self.code_info
        if ci["carries"]:
            L.append(f"integer add: {ci['carries']} carry(ies), each loses one open bin (Thm 3c)")
        if self.ctl.get("force"):
            L += ["FORCE mode: D = (A.B) OR Y.", "A forced bin gives Q=1 for ANY x, so x not in L",
                  "is accepted: the one-sided guarantee is lost (Thm 4).",
                  "Watch 'false accepts' in the stats."]
        else:
            nmin, nmax = window_stats(Y, k)
            L.append(f"window k={k}: open bins worst N={nmin}, best {nmax}")
            L.append("false accept: impossible for any Y (Thm 1)")
            if nmin == 0:
                L.append("false reject: some window has NO open bin, no bound")
            else:
                L.append(f"false reject <= (1-p)^N; p=1/2: 2^-{nmin} = {0.5 ** nmin:.2g}")
                if S and S["open_inL"] >= 20:
                    ph = S["acc_open_inL"] / S["open_inL"]
                    L.append(f"measured p={ph:.3f} (n={S['open_inL']}): (1-p)^N = {(1 - ph) ** nmin:.2g}")
            fn = frames_needed(Y, 20)
            L.append(f"frames for error <= 2^-20 at p=1/2: {fn if fn else 'never'}")
        self.analysis.config(text="\n".join(L))

    # ---------------------------------------------------------- diagram
    def _box(self, key, x0, y0, x1, y1, title, sub=""):
        c = self.c
        r = c.create_rectangle(x0, y0, x1, y1, fill=OFF, outline="#666", width=1.5)
        c.create_text((x0 + x1) / 2, y0 + 11, text=title, font=("Arial", 9, "bold"))
        s = c.create_text((x0 + x1) / 2, y1 - 11, text=sub, font=("Arial", 8), fill="#444")
        self.boxes[key] = (r, s)

    def _wire(self, key, *pts, arrow=False):
        w = self.c.create_line(*pts, fill=W_OFF, width=2, arrow="last" if arrow else None)
        self.wires.setdefault(key, []).append(w)

    def _diagram(self):
        c = self.c
        c.create_text(250, 10, text="USB camera  \u2192  frame difference (motion)",
                      font=("Arial", 9, "bold"), fill="#a33")
        c.create_text(90, 24, text="Detector B: motion mask", font=("Arial", 8))
        c.create_text(250, 24, text="camera + motion", font=("Arial", 8))
        c.create_text(410, 24, text="Detector A: motion energy", font=("Arial", 8))
        self.img_b = c.create_image(26, 34, anchor="nw")
        self.img_c = c.create_image(186, 34, anchor="nw")
        self.img_a = c.create_image(346, 34, anchor="nw")
        c.create_line(186, 82, 154, 82, arrow="last", fill="#d9480f")
        c.create_line(314, 82, 346, 82, arrow="last", fill="#d9480f")

        self._box("drvB", 30, 180, 150, 230, "Driver B", "deterministic")
        self._box("drvA", 350, 180, 470, 230, "Driver A", "randomized")
        self.x_text = c.create_text(250, 198, text="x = 0", font=("Arial", 12, "bold"))
        c.create_text(250, 218, text="moving pixels", font=("Arial", 7), fill="#666")
        c.create_line(170, 205, 150, 205, arrow="last")
        c.create_line(330, 205, 350, 205, arrow="last")
        c.create_line(90, 130, 90, 180, arrow="last")
        c.create_line(410, 130, 410, 180, arrow="last")

        self._box("link", 30, 262, 190, 332, "IF/THEN  (C + NPN)", "V(C)=0.00")
        self._wire("B", 90, 230, 90, 262, arrow=True)
        self._box("or", 300, 262, 370, 296, "OR")
        self._box("and", 300, 310, 370, 344, "AND")
        self._wire("A", 410, 230, 410, 247, 285, 247, 285, 319)
        self._wire("A", 285, 272, 300, 272)
        self._wire("A", 285, 319, 300, 319)
        self._wire("B", 190, 297, 240, 297, 240, 286, 300, 286)
        self._wire("B", 240, 297, 240, 336, 300, 336)
        self._wire("gate", 370, 279, 392, 279, 392, 327, 370, 327)
        self._wire("gate", 392, 303, 415, 303)
        self._box("node", 415, 280, 515, 330, "10\u00b5F node", "0.00")
        self._wire("node", 465, 330, 465, 352, arrow=True)
        self._box("Y", 415, 352, 515, 400, "Y time code", "")
        self._wire("D", 515, 376, 540, 376, 540, 262, 560, 262, arrow=True)
        self._box("ff", 560, 232, 660, 332, "74HC74", "")
        c.create_text(575, 262, text="D", font=("Arial", 8))
        c.create_text(575, 312, text="CLK", font=("Arial", 8))
        self._box("clk", 560, 352, 660, 400, "Clock", "= camera frame")
        c.create_line(610, 352, 610, 332, arrow="last")
        c.create_text(676, 247, text="SELECT", font=("Arial", 8, "bold"))
        self.lamp = c.create_oval(668, 258, 686, 276, fill="#ddd", outline="#555")
        c.create_text(676, 290, text="Q", font=("Arial", 8))
        c.create_text(676, 312, text="Q\u0304", font=("Arial", 8), fill="#666")

    def _lit(self, key, on, sub=None, color=ON):
        r, s = self.boxes[key]
        self.c.itemconfig(r, fill=color if on else OFF)
        if sub is not None:
            self.c.itemconfig(s, text=sub)

    def _wire_state(self, key, on):
        for w in self.wires.get(key, []):
            self.c.itemconfig(w, fill=W_ON if on else W_OFF)

    # ---------------------------------------------------------- right panel
    def _right_panel(self):
        f = ttk.Frame(self.root)
        f.grid(row=1, column=1, sticky="n", padx=6, pady=4)
        ttk.Label(f, text="IF/THEN link during the latest frame",
                  font=("Arial", 9, "bold")).pack(anchor="w")
        self.plot1 = tk.Canvas(f, width=420, height=170, bg="white",
                               highlightthickness=1, highlightbackground="#bbb")
        self.plot1.pack()
        ttk.Label(f, text="Moving pixels per frame (x)",
                  font=("Arial", 9, "bold")).pack(anchor="w", pady=(6, 0))
        self.plot2 = tk.Canvas(f, width=420, height=90, bg="white",
                               highlightthickness=1, highlightbackground="#bbb")
        self.plot2.pack()
        ttk.Label(f, text="Q per frame: green=1, grey=0, hollow=no motion, red edge=Y blocked",
                  font=("Arial", 8, "bold")).pack(anchor="w", pady=(6, 0))
        self.strip = tk.Canvas(f, width=420, height=34, bg="white",
                               highlightthickness=1, highlightbackground="#bbb")
        self.strip.pack()
        self.verdict = ttk.Label(f, text="Idle", font=("Arial", 11, "bold"), wraplength=420)
        self.verdict.pack(anchor="w", pady=(6, 0))
        self.stats = ttk.Label(f, text="", font=("Consolas", 9), justify="left")
        self.stats.pack(anchor="w")

    def _draw_traj(self):
        p = self.plot1
        p.delete("all")
        W, H, m = 420, 170, 28
        top, bot = 8, H - 20
        sy = lambda v: bot - v / 5.3 * (bot - top)
        for v in range(6):
            p.create_line(m, sy(v), W - 6, sy(v), fill="#eee")
            p.create_text(14, sy(v), text=str(v), font=("Arial", 7))
        for v, col, lab in ((V_BE, "#888", "0.7 V NPN on"), (V_IH, "#c33", "3.5 V logic HIGH")):
            p.create_line(m, sy(v), W - 6, sy(v), fill=col, dash=(4, 3))
            p.create_text(W - 8, sy(v) - 6, text=lab, anchor="e", font=("Arial", 7), fill=col)
        tr = self.last_traj
        if len(tr) > 1:
            tmax = tr[-1][0]
            sx = lambda t: m + t / tmax * (W - m - 6)
            for k, col, lab in ((1, "#1f6fb5", "V(C)"), (2, "#2a9d3a", "V(B)")):
                pts = []
                for row in tr:
                    pts += [sx(row[0]), sy(row[k])]
                p.create_line(*pts, fill=col, width=2)
            p.create_text(m + 6, H - 8, text="\u25cf V(C)", anchor="w", fill="#1f6fb5", font=("Arial", 8))
            p.create_text(m + 70, H - 8, text="\u25cf V(B)", anchor="w", fill="#2a9d3a", font=("Arial", 8))
            p.create_text(W - 8, H - 8, text=f"frame period {tmax:.0f} ms; 1.83RC = {LOGIC_TIME*1000:.1f} ms",
                          anchor="e", font=("Arial", 8), fill="#555")

    def _draw_x(self):
        p = self.plot2
        p.delete("all")
        W, H, m = 420, 90, 6
        if len(self.x_hist) < 2:
            return
        ymax = max(100, max(self.x_hist) * 1.1)
        sy = lambda v: H - 6 - v / ymax * (H - 12)
        step = (W - 2 * m) / (self.x_hist.maxlen - 1)
        thr = self.ctl["min_px"]
        p.create_line(m, sy(thr), W - m, sy(thr), fill="#c33", dash=(4, 3))
        p.create_text(W - m, sy(thr) - 6, text=f"min {thr}", anchor="e", fill="#c33", font=("Arial", 7))
        pts = []
        for i, v in enumerate(self.x_hist):
            pts += [m + i * step, sy(v)]
        p.create_line(*pts, fill="#d9480f", width=2)
        p.create_text(m + 2, 8, text=f"max {int(ymax)}", anchor="w", font=("Arial", 7), fill="#666")

    def _draw_strip(self):
        s = self.strip
        s.delete("all")
        for i, (q, y, mo) in enumerate(self.q_hist):
            row, col = divmod(i, 42)
            x0, y0 = 2 + col * 10, 4 + row * 14
            if not mo:
                s.create_rectangle(x0, y0, x0 + 8, y0 + 8, outline="#aaa")
                continue
            s.create_rectangle(x0, y0, x0 + 8, y0 + 8,
                               fill="#3cb043" if q else "#d8d8d8",
                               outline="#c33" if y == 0 else "#888")

    # ---------------------------------------------------------- run control
    def start(self, circuit=True):
        self._y_changed()
        if not self.y_ok:
            messagebox.showerror("Time code", "Fix the time code first (bits 0/1 with | & + only).")
            return
        self.ctl.update(stop=False, circuit=circuit)
        self.q = queue.Queue()
        self.q_hist.clear()
        self.x_hist.clear()
        self.last_traj = []
        self.err = False
        self._log_clear()
        self.verdict.config(text="Starting ...")
        self.thread = SimThread(int(self.s_cam.get()), self.ctl, self.q)
        self.thread.start()
        self.b_start.config(state="disabled")
        self.b_prev.config(state="disabled")
        self.b_stop.config(state="normal")

    def stop(self):
        self.ctl["stop"] = True

    def _close(self):
        self.ctl["stop"] = True
        self.root.after(150, self.root.destroy)

    # ---------------------------------------------------------- events
    def _poll(self):
        last = None
        try:
            while True:
                ev = self.q.get_nowait()
                if ev[0] == "frame":
                    d = ev[1]
                    self.x_hist.append(d["x"])
                    if d["circuit"]:
                        self.q_hist.append((d["Q"], d["y_bin"], d["motion"]))
                        self.last_traj = d["traj"] or self.last_traj
                        if d["motion"]:
                            self._frame_log(d)
                    last = d
                elif ev[0] == "analysis":
                    self.an_lines = ev[1]
                    self._show_analysis()
                elif ev[0] == "error":
                    self.err = True
                    self._log("# ERROR: " + ev[1])
                    messagebox.showerror("Camera", ev[1])
                elif ev[0] == "info":
                    self._log("# " + ev[1])
                elif ev[0] == "done":
                    self.b_start.config(state="normal")
                    self.b_prev.config(state="normal")
                    self.b_stop.config(state="disabled")
                    self.verdict.config(text="Camera not available" if self.err else "Stopped")
        except queue.Empty:
            pass
        if last:
            self._render(last)
            self._draw_traj()
            self._draw_x()
            self._draw_strip()
        self.root.after(15, self._poll)

    def _render(self, d):
        g = d["small"]
        prev_rgb = np.stack([g, g, g], axis=-1).copy()
        prev_rgb[d["mask"]] = (255, 40, 40)
        m = d["mask"].astype(np.uint8)
        mask_rgb = np.stack([m * 255, m * 115, m * 40], axis=-1)
        e = np.clip(d["energy"].astype(np.int16) * 4, 0, 255).astype(np.uint8)
        energy_rgb = np.stack([e, e, e], axis=-1)
        self.photos = [ppm_photo(mask_rgb), ppm_photo(prev_rgb), ppm_photo(energy_rgb)]
        self.c.itemconfig(self.img_b, image=self.photos[0])
        self.c.itemconfig(self.img_c, image=self.photos[1])
        self.c.itemconfig(self.img_a, image=self.photos[2])
        self.c.itemconfig(self.x_text, text=f"x = {d['x']}" if d["motion"] else "no motion")

        self.cam_photo = ppm_photo(d["preview"])
        self.cam.itemconfig(self.cam_img, image=self.cam_photo)
        self.cam.itemconfig(
            self.cam_text,
            text=f"{1 / d['dt']:.0f} fps   " + (f"moving px: {d['x']}" if d["motion"] else "no motion"))
        if not d["circuit"]:
            self.verdict.config(text="Preview only: circuit idle. Adjust the sliders, then Start circuit.")
            return

        self._lit("drvA", d["A"], f"A={d['A']}   r={d['r']}")
        self._lit("drvB", d["c_of_x"], f"c(x) = {d['c_of_x']}")
        self._lit("link", d["B"], f"V(C)={d['vC']:.2f}  V(B)={d['vB']:.2f}", GOOD)
        self._lit("or", d["orv"])
        self._lit("and", d["andv"])
        self._lit("node", d["node"] > 0.5, f"{d['node']:.2f}")
        self._lit("Y", d["y_bin"], f"[{self.ctl['y']}]  bin {d['idx'] % len(self.ctl['y'])}={d['y_bin']}")
        self._lit("ff", d["Q"], f"Q = {d['Q']}", GOOD)
        self._lit("clk", False, f"frame {d['idx']}  {d['dt']*1000:.0f} ms")
        self.c.itemconfig(self.lamp, fill="#3cb043" if d["select"] else "#ddd")
        self._wire_state("A", d["A"])
        self._wire_state("B", d["B"])
        self._wire_state("gate", d["gate"])
        self._wire_state("node", d["node"] > 0.5)
        self._wire_state("D", d["node"] > 0.5 and d["y_bin"])

        S = d["stats"]
        if d["select"]:
            self.verdict.config(text=f"SELECT = 1  ACCEPT  (witness found within last {self.ctl['k']} frames)")
        elif d["motion"]:
            self.verdict.config(text=f"SELECT = 0  (x = {d['x']}: no witness in last {self.ctl['k']} frames)")
        else:
            self.verdict.config(text="SELECT = 0  (no motion, no instance)")
        self._refresh_analysis(S)
        rate = S["acc_in_L"] / S["in_L"] if S["in_L"] else 0.0
        self.stats.config(text=(
            f"frames {S['frames']}  with motion {S['motion']}\n"
            f"x in L (odd composite): {S['in_L']}  accepted: {S['acc_in_L']}  ({rate:.0%})\n"
            f"x not in L accepted (false accepts): {S['false_acc']}"))

    def _frame_log(self, d):
        kind = "odd composite" if d["in_L"] else ("even" if d["x"] % 2 == 0 else "prime/small")
        self._log(f"frame {d['idx']:5d} x={d['x']:<6d} ({kind:<13s}) r={d['r']:<6d} "
                  f"A={d['A']} B={d['B']} node={d['node']:.2f} Y={d['y_bin']} -> Q={d['Q']}")

    # ---------------------------------------------------------- log helpers
    def _log(self, s):
        self.log.config(state="normal")
        self.log.insert("end", s + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 400:
            self.log.delete("1.0", "100.0")
        self.log.see("end")
        self.log.config(state="disabled")

    def _log_clear(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
