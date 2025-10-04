import numpy as np
import streamlit as st

import matplotlib.pyplot as plt
import re

def u(t, shift=0):
    return (t >= shift).astype(int)

def delta_gauss(t, shift=0, eps=1e-3):
    return np.exp(-(t-shift)**2/eps) / np.sqrt(np.pi*eps)

def impulse_on_grid(t, a, dt):
    h = np.zeros_like(t)
    idx = int(np.argmin(np.abs(t - a)))
    h[idx] = 1.0 / dt
    return h

def preprocess(expr: str) -> str:
    if not expr:
        return ""
    expr = re.sub(r"e\^\(([^)]+)\)", r"np.exp(\1)", expr)
    expr = re.sub(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*t\b", r"\1*t", expr)
    return expr

_delta_signed_ascii = re.compile(r"delta\s*\(\s*t\s*(?P<sign>[+-])\s*(?P<val>[0-9.]+)\s*\)")
_delta_zero_ascii   = re.compile(r"delta\s*\(\s*t\s*\)")
_delta_signed_greek = re.compile(r"[δ]\s*\(\s*t\s*(?P<sign>[+-])\s*(?P<val>[0-9.]+)\s*\)")
_delta_zero_greek   = re.compile(r"[δ]\s*\(\s*t\s*\)")

def extract_impulses(raw_expr: str):
    expr = raw_expr or ""
    shifts = []
    def _take(m):
        sign, val = m.group('sign'), float(m.group('val'))
        shifts.append(val if sign == '-' else -val)
        return "0"
    expr = _delta_signed_ascii.sub(_take, expr)
    expr = _delta_signed_greek.sub(_take, expr)
    if _delta_zero_ascii.search(expr):
        expr = _delta_zero_ascii.sub(lambda m: (shifts.append(0.0) or "0"), expr)
    if _delta_zero_greek.search(expr):
        expr = _delta_zero_greek.sub(lambda m: (shifts.append(0.0) or "0"), expr)
    return expr, shifts

def evaluate_expr(expr: str, t: np.ndarray) -> np.ndarray:
    expr = preprocess(expr)
    if expr.strip() == "":
        return np.zeros_like(t, dtype=float)
    try:
        res = eval(expr, {"np": np, "t": t, "u": u, "delta": delta_gauss})
        res = np.asarray(res)
        if res.ndim == 0 or res.size == 1:
            return np.full_like(t, float(res), dtype=float)
        if res.shape == t.shape:
            return res.astype(float, copy=False)
        if res.ndim == 1 and res.size == t.size:
            return res.astype(float, copy=False)
        raise ValueError(f"Expression produced shape {res.shape}, expected {(t.shape,)}")
    except Exception as e:
        st.error(f"Error in expression '{expr}': {e}")
        return np.zeros_like(t, dtype=float)

def add_impulse_contributions(y, t, x, h, x_impulses, h_impulses):
    if x_impulses:
        for a in x_impulses:
            y += np.interp(t - a, t, h, left=0.0, right=0.0)
    if h_impulses:
        for a in h_impulses:
            y += np.interp(t - a, t, x, left=0.0, right=0.0)
    return y

st.title("Convolution Visualizer")

mode = st.selectbox("Choose signal type:", ["Continuous", "Discrete", "Wave"])

if mode == "Continuous":
    st.subheader("Choose bounds and Samples")
    c1, c2, c3 = st.columns(3)
    with c1:
        t1_expr = st.text_input("t1 =", "-5")
    with c2:
        t2_expr = st.text_input("t2 =", "25")
    with c3:
        samples_expr = st.text_input("samples =", "4000")

    st.subheader("Define Signals")
    c1, c2 = st.columns(2)
    with c1:
        x_expr_raw = st.text_input("x(t) =", "u(t) - u(t-2)")
    with c2:
        h_expr_raw = st.text_input("h(t) =", "delta(t-5) + np.exp(-t)*u(t)")

    if st.button("Calculate"):
        try:
            t1 = float(t1_expr)
            t2 = float(t2_expr)
            samples = int(samples_expr)
            if samples < 4:
                raise ValueError("samples must be >= 4")

            t = np.linspace(t1, t2, samples)
            dt = t[1] - t[0]

            x_no_imp, x_impulses = extract_impulses(x_expr_raw)
            h_no_imp, h_impulses = extract_impulses(h_expr_raw)

            x = evaluate_expr(x_no_imp, t)
            h = evaluate_expr(h_no_imp, t)

            y = np.convolve(x, h, mode="same") * dt
            y = add_impulse_contributions(y, t, x, h, x_impulses, h_impulses)

            st.latex(r"y(t) = (x * h)(t) = \int_{-\infty}^{\infty} x(\tau)\,h(t-\tau)\,d\tau")

            plt.style.use("default")
            fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            fig.patch.set_alpha(0)
            for a in ax:
                a.set_facecolor("none")

            ax[0].plot(t, x, color="royalblue", linewidth=3, label="x(t) (no impulses)")
            if x_impulses:
                x_spike = np.zeros_like(t)
                for a in x_impulses:
                    x_spike += impulse_on_grid(t, a, dt)
                ax[0].plot(t, x_spike, color="royalblue", alpha=0.5, label="impulses in x")
            ax[0].set_title("x(t)", fontsize=24, fontweight="bold", color="royalblue")
            ax[0].legend(fontsize=12)
            ax[0].grid(True, linestyle="--", alpha=0.6)

            ax[1].plot(t, h, color="darkorange", linewidth=3, label="h(t) (no impulses)")
            if h_impulses:
                h_spike = np.zeros_like(t)
                for a in h_impulses:
                    h_spike += impulse_on_grid(t, a, dt)
                ax[1].plot(t, h_spike, color="darkorange", alpha=0.5, label="impulses in h")
            ax[1].set_title("h(t)", fontsize=24, fontweight="bold", color="darkorange")
            ax[1].legend(fontsize=12)
            ax[1].grid(True, linestyle="--", alpha=0.6)

            ax[2].plot(t, y, color="forestgreen", linewidth=3, label="y(t) = x*h")
            ax[2].set_title("y(t) = x*h", fontsize=24, fontweight="bold", color="forestgreen")
            ax[2].legend(fontsize=12)
            ax[2].grid(True, linestyle="--", alpha=0.6)

            for a in ax:
                for spine in a.spines.values():
                    spine.set_color("white"); spine.set_linewidth(1.8)
                a.tick_params(colors="white", labelsize=14)
                a.title.set_color("white")
                a.xaxis.label.set_color("white"); a.yaxis.label.set_color("white")
                a.relim(); a.autoscale_view()

            plt.tight_layout()
            st.pyplot(fig)

            if x_impulses or h_impulses:
                st.caption(
                    f"Impulse shifts used exactly — x impulses: {x_impulses or []}, "
                    f"h impulses: {h_impulses or []}"
                )

        except ValueError as e:
            st.error(f"Input error: {e}")

elif mode == "Discrete":
    st.subheader("Discrete Sequences")
    n = np.arange(-5, 6)
    x = (n >= 0).astype(int)
    h = (n == 0).astype(int) + (n == 1).astype(int)
    y = np.convolve(x, h, mode="full")
    n_out = np.arange(len(y)) + n[0] + n[0]

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    fig.patch.set_alpha(0)
    ax[0].stem(n, x, basefmt=" ", linefmt="C0-", markerfmt="C0o"); ax[0].set_title("x[n]")
    ax[1].stem(n, h, basefmt=" ", linefmt="C1-", markerfmt="C1s"); ax[1].set_title("h[n]")
    ax[2].stem(n_out, y, basefmt=" ", linefmt="C2-", markerfmt="C2^"); ax[2].set_title("y[n] = x*h")
    for a in ax:
        a.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

elif mode == "Wave":
    st.subheader("Wave Signals")
    t = np.linspace(-5, 5, 1200)
    freq = st.slider("Frequency", 1, 10, 2)
    x = np.sin(2 * np.pi * freq * t)
    h = np.sin(2 * np.pi * (freq/2) * t)
    dt = t[1] - t[0]
    y = np.convolve(x, h, mode="same") * dt

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_alpha(0)
    ax[0].plot(t, x, linewidth=3, color="royalblue");    ax[0].set_title("x(t)")
    ax[1].plot(t, h, linewidth=3, color="darkorange");   ax[1].set_title("h(t)")
    ax[2].plot(t, y, linewidth=3, color="forestgreen");  ax[2].set_title("y(t) = x*h")
    for a in ax:
        a.grid(True, ls="--", alpha=0.6); a.relim(); a.autoscale_view()
    plt.tight_layout()
    st.pyplot(fig)
