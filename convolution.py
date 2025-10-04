import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import re

# ==============================
# Utilities
# ==============================

def u(t, shift=0.0):
    """Unit step u(t - shift) for vector t."""
    return (t >= shift).astype(int)

def delta_gauss(t, shift=0.0, eps=1e-3):
    """Numerical stand-in for Dirac delta, for visualization-only expressions."""
    return np.exp(-(t - shift) ** 2 / eps) / np.sqrt(np.pi * eps)

def impulse_on_grid(t, a, dt):
    """Draw a single 'impulse' spike at t=a with area 1, as 1/dt at nearest grid index."""
    h = np.zeros_like(t)
    idx = int(np.argmin(np.abs(t - a)))
    h[idx] = 1.0 / dt
    return h

def preprocess(expr: str) -> str:
    """Normalize a few common syntaxes to valid numpy/Python."""
    if not expr:
        return ""
    s = expr
    # e^(...) -> np.exp(...)
    s = re.sub(r"e\^\(([^)]+)\)", r"np.exp(\1)", s)
    # '3 t' -> '3*t' (implicit multiplication with plain numbers)
    s = re.sub(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*t\b", r"\1*t", s)
    return s

# Patterns for delta(...) or δ(...)
_delta_signed_ascii = re.compile(r"delta\s*\(\s*t\s*(?P<sign>[+-])\s*(?P<val>[0-9.]+)\s*\)")
_delta_zero_ascii   = re.compile(r"delta\s*\(\s*t\s*\)")
_delta_signed_greek = re.compile(r"[δ]\s*\(\s*t\s*(?P<sign>[+-])\s*(?P<val>[0-9.]+)\s*\)")
_delta_zero_greek   = re.compile(r"[δ]\s*\(\s*t\s*\)")

def extract_impulses(raw_expr: str):
    """
    Returns (expr_without_impulses, impulses)
    where impulses is a list of (shift, coefficient) tuples for δ(t-shift).
    
    Handles:
    - delta(t - a) or δ(t - a) => impulse at t=+a
    - delta(t + a) or δ(t + a) => impulse at t=-a  
    - delta(t) or δ(t) => impulse at t=0
    - 3*delta(t-2) => impulse with coefficient 3
    - -delta(t) => impulse with coefficient -1
    """
    import re
    
    expr = raw_expr or ""
    impulses = []  # List of (shift, coefficient) tuples
    
    # Pattern to match: [coefficient*]delta(t [+/-] value) or [coefficient*]δ(t [+/-] value)
    # Captures: optional coefficient, delta/δ, optional sign and value
    pattern = r'(?P<coef>-?\d*\.?\d*)\s*\*?\s*(?P<delta>delta|δ)\s*\(\s*t\s*(?:(?P<sign>[+-])\s*(?P<val>[0-9.]+))?\s*\)'
    
    def replace_impulse(match):
        coef_str = match.group('coef')
        sign = match.group('sign')
        val_str = match.group('val')
        
        # Parse coefficient
        if coef_str == '' or coef_str == '+':
            coef = 1.0
        elif coef_str == '-':
            coef = -1.0
        else:
            coef = float(coef_str)
        
        # Parse shift
        if val_str is None:
            # delta(t) or δ(t)
            shift = 0.0
        else:
            val = float(val_str)
            # delta(t - a) => shift = +a
            # delta(t + a) => shift = -a
            shift = val if sign == '-' else -val
        
        impulses.append((shift, coef))
        return "0"  # Replace impulse with 0 in expression
    
    expr = re.sub(pattern, replace_impulse, expr)
    
    return expr, impulses

def evaluate_expr(expr: str, t: np.ndarray) -> np.ndarray:
    """Safely evaluate an expression of t into a 1D array matching t.shape."""
    expr = preprocess(expr)
    if expr.strip() == "":
        return np.zeros_like(t, dtype=float)
    try:
        # available names inside eval:
        res = eval(expr, {"np": np, "t": t, "u": u, "delta": delta_gauss})
        res = np.asarray(res)
        if res.ndim == 0 or res.size == 1:
            return np.full_like(t, float(res), dtype=float)
        if res.shape == t.shape:
            return res.astype(float, copy=False)
        if res.ndim == 1 and res.size == t.size:
            return res.astype(float, copy=False)
        raise ValueError(f"Expression produced shape {res.shape}, expected {t.shape}")
    except Exception as e:
        st.error(f"Error in expression '{expr}': {e}")
        return np.zeros_like(t, dtype=float)

def add_impulse_contributions_fullgrid(y_full, t_full,
                                       t_xpad, x_pad,
                                       t_hpad, h_pad,
                                       x_impulses, h_impulses,
                                       dt):
    """
    Add the contributions due to impulses with coefficients.
    
    If x includes c1*δ(t - a): y(t) += c1*h(t - a).
    If h includes c2*δ(t - b): y(t) += c2*x(t - b).
    If both have impulses: δ(t-a) * δ(t-b) = δ(t - (a+b))
    
    x_impulses and h_impulses are lists of (shift, coefficient) tuples.
    """
    # Impulse-to-impulse convolution: δ(t-a) * δ(t-b) = δ(t-(a+b))
    if x_impulses and h_impulses:
        for x_shift, x_coef in x_impulses:
            for h_shift, h_coef in h_impulses:
                # Resulting impulse at (x_shift + h_shift) with amplitude (x_coef * h_coef)
                total_shift = x_shift + h_shift
                total_coef = x_coef * h_coef
                # Add impulse spike at the appropriate location
                idx = np.argmin(np.abs(t_full - total_shift))
                if 0 <= idx < len(y_full):
                    y_full[idx] += total_coef / dt
    
    # Impulse in x convolves with continuous h
    if x_impulses:
        for shift, coef in x_impulses:
            # Add coef * h(t - shift) sampled on t_full
            h_shifted = np.interp(t_full - shift, t_hpad, h_pad, left=0.0, right=0.0)
            y_full += coef * h_shifted
            
    # Impulse in h convolves with continuous x
    if h_impulses:
        for shift, coef in h_impulses:
            # Add coef * x(t - shift) sampled on t_full
            x_shifted = np.interp(t_full - shift, t_xpad, x_pad, left=0.0, right=0.0)
            y_full += coef * x_shifted
    
    return y_full

# ==============================
# Streamlit UI
# ==============================

st.title("Convolution Visualizer")

mode = st.selectbox("Choose signal type:", ["Continuous", "Discrete", "Wave"])

# ------------------------------
# Continuous-time mode
# ------------------------------
if mode == "Continuous":
    st.subheader("Choose bounds and samples")
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

            # Build time grid
            t = np.linspace(t1, t2, samples)
            dt = t[1] - t[0]

            # Split impulses and evaluate regular parts
            x_no_imp, x_impulses = extract_impulses(x_expr_raw)
            h_no_imp, h_impulses = extract_impulses(h_expr_raw)

            x = evaluate_expr(x_no_imp, t)
            h = evaluate_expr(h_no_imp, t)

            # Zero-pad to reduce truncation at edges
            pad_n = samples // 2
            x_pad = np.pad(x, (pad_n, pad_n))
            h_pad = np.pad(h, (pad_n, pad_n))

            # Time bases for padded signals
            t_x0 = t[0] - pad_n * dt
            t_h0 = t[0] - pad_n * dt
            t_xpad = t_x0 + np.arange(x_pad.size) * dt
            t_hpad = t_h0 + np.arange(h_pad.size) * dt

            # Full linear convolution with Riemann scaling
            y_full = np.convolve(x_pad, h_pad, mode="full") * dt

            # Full convolution time base
            # start time = start_x + start_h
            t_full_start = t_x0 + t_h0
            t_full = t_full_start + np.arange(y_full.size) * dt

            # Add impulse contributions on the full grid
            y_full = add_impulse_contributions_fullgrid(
                y_full, t_full,
                t_xpad, x_pad,
                t_hpad, h_pad,
                x_impulses, h_impulses,
                dt
            )

            # Resample back to original t for plotting
            y = np.interp(t, t_full, y_full, left=0.0, right=0.0)

            st.latex(r"y(t) = (x * h)(t) = \int_{-\infty}^{\infty} x(\tau)\,h(t-\tau)\,d\tau")

            plt.style.use("default")

            # Global-ish tweaks for dark bg
            import matplotlib as mpl
            mpl.rcParams.update({
                "xtick.color": "white",
                "ytick.color": "white",
                "axes.labelcolor": "white",
                "text.color": "white",
                "axes.edgecolor": "white",
                "grid.color": "white",
                "grid.alpha": 0.15,         # softer grid
                "legend.edgecolor": "white"
            })

            fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            fig.patch.set_alpha(0)          # keep fig transparent for Streamlit dark
            for a in ax:
                a.set_facecolor("none")     # transparent axes
                # crisp white spines
                for sp in a.spines.values():
                    sp.set_color("white")
                # ticks + minor ticks
                a.tick_params(axis='both', colors='white', labelsize=11, length=6, width=1)
                a.minorticks_on()
                a.tick_params(axis='both', which='minor', length=3, width=0.8)
                # light dashed grid on major ticks
                a.grid(True, linestyle="--")

            # x(t)
            ax[0].plot(t, x, linewidth=3, label="x(t) (no impulses)")
            if x_impulses:
                x_spike = np.zeros_like(t)
                for shift, coef in x_impulses:
                    x_spike += coef * impulse_on_grid(t, shift, dt)
                ax[0].plot(t, x_spike, alpha=0.6, label="impulses in x")
            ax[0].set_title("x(t)", color="white", fontsize=14)
            ax[0].legend(fontsize=12, framealpha=0.12, facecolor="black")

            # h(t)
            ax[1].plot(t, h, linewidth=3, label="h(t) (no impulses)")
            if h_impulses:
                h_spike = np.zeros_like(t)
                for shift, coef in h_impulses:
                    h_spike += coef * impulse_on_grid(t, shift, dt)
                ax[1].plot(t, h_spike, alpha=0.6, label="impulses in h")
            ax[1].set_title("h(t)", color="white", fontsize=14)
            ax[1].legend(fontsize=12, framealpha=0.12, facecolor="black")

            # y(t)
            ax[2].plot(t, y, linewidth=3, label="y(t) = x*h")
            ax[2].set_title("y(t) = x*h", color="white", fontsize=14)
            ax[2].legend(fontsize=12, framealpha=0.12, facecolor="black")

            for a in ax:
                a.relim()
                a.autoscale_view()

            plt.tight_layout()
            st.pyplot(fig)

            if x_impulses or h_impulses:
                st.caption(
                    f"Impulse shifts used — x impulses: {x_impulses or []}, "
                    f"h impulses: {h_impulses or []}"
                )

        except ValueError as e:
            st.error(f"Input error: {e}")

# ------------------------------
# Discrete-time mode
# ------------------------------
elif mode == "Discrete":
    st.subheader("Discrete Sequences (demo)")
    n = np.arange(-5, 6)
    x = (n >= 0).astype(int)
    h = (n == 0).astype(int) + (n == 1).astype(int)
    y = np.convolve(x, h, mode="full")
    n_out = np.arange(len(y)) + 2*n[0]

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    fig.patch.set_alpha(0)

    ax[0].stem(n, x, basefmt=" "); ax[0].set_title("x[n]")
    ax[1].stem(n, h, basefmt=" "); ax[1].set_title("h[n]")
    ax[2].stem(n_out, y, basefmt=" "); ax[2].set_title("y[n] = x*h")

    for a in ax:
        a.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)

# ------------------------------
# Wave (smooth demo)
# ------------------------------
elif mode == "Wave":
    st.subheader("Wave Signals (demo)")
    t = np.linspace(-5, 5, 2000)
    freq = st.slider("Frequency", 1, 10, 2)
    x = np.sin(2 * np.pi * freq * t)
    h = np.sin(2 * np.pi * (freq / 2) * t)
    dt = t[1] - t[0]
    # "same" is fine here since both are defined on same grid and we only visualize
    y = np.convolve(x, h, mode="same") * dt

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_alpha(0)

    ax[0].plot(t, x, linewidth=3); ax[0].set_title("x(t)")
    ax[1].plot(t, h, linewidth=3); ax[1].set_title("h(t)")
    ax[2].plot(t, y, linewidth=3); ax[2].set_title("y(t) = x*h")

    for a in ax:
        a.grid(True, ls="--", alpha=0.6)
        a.relim(); a.autoscale_view()

    plt.tight_layout()
    st.pyplot(fig)
