# app.py
import io
import os
import time
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Data Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def load_file(file) -> pd.DataFrame:
    """Load CSV/XLSX from an uploaded file-like object."""
    if file is None:
        return pd.DataFrame()
    name = file.name.lower()
    if name.endswith(".csv"):
        # try to detect delimiter
        try:
            df = pd.read_csv(file, low_memory=False)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, sep=";", low_memory=False)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file)
    else:
        raise ValueError("Please upload a .csv or .xlsx file")
    return df

@st.cache_data(show_spinner=False)
def load_sample(sample_name: str) -> pd.DataFrame:
    """Provide a few built-in small datasets."""
    if sample_name == "Iris":
        from sklearn import datasets
        iris = datasets.load_iris(as_frame=True)
        df = iris.frame
        df.rename(columns={"target": "species"}, inplace=True)
        return df
    if sample_name == "Tips":
        # small synthetic "tips" style dataset
        rng = np.random.default_rng(42)
        n = 244
        df = pd.DataFrame({
            "total_bill": rng.normal(20, 8, n).round(2),
            "tip": rng.normal(3, 1.1, n).round(2),
            "sex": rng.choice(["Female", "Male"], n),
            "smoker": rng.choice(["Yes", "No"], n, p=[0.2, 0.8]),
            "day": rng.choice(["Thur", "Fri", "Sat", "Sun"], n, p=[0.2, 0.2, 0.35, 0.25]),
            "time": rng.choice(["Lunch", "Dinner"], n, p=[0.3, 0.7]),
            "size": rng.integers(1, 6, n),
        })
        return df
    if sample_name == "Sales (dates)":
        dates = pd.date_range("2024-01-01", periods=140, freq="D")
        stores = ["North", "East", "South", "West"]
        cats = ["A", "B", "C"]
        rows = []
        rng = np.random.default_rng(0)
        for d in dates:
            for s in stores:
                for c in cats:
                    rows.append({
                        "date": d,
                        "store": s,
                        "category": c,
                        "units": int(rng.integers(0, 50)),
                        "price": float(np.clip(rng.normal(10, 2), 3, 25)),
                    })
        df = pd.DataFrame(rows)
        df["revenue"] = (df["units"] * df["price"]).round(2)
        return df
    return pd.DataFrame()

def try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Try to cast columns that look like dates."""
    if df.empty:
        return df
    for col in df.columns:
        if df[col].dtype == object:
            # heuristic: if > 60% parseable, convert
            sample = df[col].dropna().astype(str).head(200)
            success = 0
            for v in sample:
                try:
                    pd.to_datetime(v)
                    success += 1
                except Exception:
                    pass
            if len(sample) > 0 and success / len(sample) >= 0.6:
                with pd.option_context("mode.chained_assignment", None):
                    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def memory_usage_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / (1024 ** 2)

def df_info(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    info = []
    for c in df.columns:
        s = df[c]
        info.append({
            "column": c,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "nulls": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
            "example": s.dropna().iloc[0] if s.notna().any() else None,
        })
    return pd.DataFrame(info)

def numeric_cols(df): return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
def categorical_cols(df): return [c for c in df.columns if pd.api.types.is_categorical_dtype(df[c]) or df[c].dtype == object]
def datetime_cols(df): return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

def filter_block(df: pd.DataFrame) -> pd.DataFrame:
    """Build a column-wise filter UI in the sidebar."""
    if df.empty:
        return df
    st.sidebar.markdown("### 🔎 Filters")
    filtered = df.copy()
    with st.sidebar.expander("Toggle & set filters", expanded=False):
        cols_to_filter = st.multiselect(
            "Columns to filter",
            options=list(df.columns),
            help="Pick columns to filter; controls appear below."
        )
        for col in cols_to_filter:
            s = df[col]
            st.markdown(f"**{col}**")
            if pd.api.types.is_numeric_dtype(s):
                min_v, max_v = float(np.nanmin(s)), float(np.nanmax(s))
                a, b = st.slider(
                    f"Range for {col}", min_value=min_v, max_value=max_v,
                    value=(min_v, max_v), step=(max_v - min_v)/100 or 1.0, label_visibility="collapsed"
                )
                filtered = filtered[(filtered[col] >= a) & (filtered[col] <= b)]
            elif pd.api.types.is_datetime64_any_dtype(s):
                min_d, max_d = s.min(), s.max()
                d1, d2 = st.date_input(
                    f"Date range for {col}",
                    value=(min_d.date(), max_d.date()) if pd.notna(min_d) and pd.notna(max_d) else None,
                    help="Pick inclusive date range", label_visibility="collapsed"
                )
                if isinstance(d1, tuple) or isinstance(d2, tuple):
                    # streamlit quirk safety
                    pass
                else:
                    d1 = pd.to_datetime(d1)
                    d2 = pd.to_datetime(d2)
                    filtered = filtered[(filtered[col] >= d1) & (filtered[col] <= d2)]
            else:
                opts = sorted([x for x in s.dropna().unique().tolist()][:500])
                pick = st.multiselect(
                    f"Values for {col}", options=opts, default=opts[: min(10, len(opts))],
                    label_visibility="collapsed"
                )
                if pick:
                    filtered = filtered[filtered[col].isin(pick)]
    return filtered

def csv_download(df: pd.DataFrame, filename: str) -> None:
    if df.empty:
        st.info("No data to download.")
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, file_name=filename, mime="text/csv")

# -----------------------------
# Sidebar: data source
# -----------------------------
st.sidebar.title("📦 Data")
source = st.sidebar.radio("Choose data source", ["Upload", "Sample data"], horizontal=True)

df_raw = pd.DataFrame()
if source == "Upload":
    up = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], accept_multiple_files=False)
    if up:
        with st.spinner("Loading file…"):
            df_raw = load_file(up)
else:
    sample = st.sidebar.selectbox("Pick a sample", ["Sales (dates)", "Iris", "Tips"])
    with st.spinner("Building sample…"):
        df_raw = load_sample(sample)

# optional preprocessing
if not df_raw.empty:
    df_raw = try_parse_dates(df_raw)

# filters
df_filtered = filter_block(df_raw)

# -----------------------------
# Header + key metrics
# -----------------------------
st.title("📊 Data Studio")
st.caption("Upload or pick a sample dataset, explore it across tabs, and export results.")

left, right = st.columns([2, 1])
with left:
    if df_filtered.empty:
        st.info("Load a dataset to get started.")
    else:
        st.success(f"Loaded **{len(df_raw):,}** rows × **{df_raw.shape[1]}** columns "
                   f"(filtered to **{len(df_filtered):,}** rows).")
with right:
    if not df_raw.empty:
        st.metric("Memory (raw)", f"{memory_usage_mb(df_raw):.2f} MB")
        st.metric("Memory (filtered)", f"{memory_usage_mb(df_filtered):.2f} MB")

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_explore, tab_visual, tab_pivot, tab_profile, tab_settings = st.tabs(
    ["Overview", "Explore", "Visualize", "Pivot", "Profile", "Settings"]
)

# ---- Overview
with tab_overview:
    st.subheader("Data preview")
    st.dataframe(df_filtered.head(200), use_container_width=True)
    st.divider()
    st.subheader("Schema")
    info = df_info(df_filtered)
    if not info.empty:
        st.dataframe(info, use_container_width=True)
    else:
        st.write("No schema to display.")
    st.divider()
    st.subheader("Missing values heatmap (top 40 cols)")
    if not df_filtered.empty:
        show_df = df_filtered.iloc[:300].copy()
        miss = show_df[show_df.columns[: min(40, show_df.shape[1])]].isna()
        miss = miss.reset_index().melt("index", var_name="column", value_name="is_na")
        chart = (
            alt.Chart(miss)
            .mark_rect()
            .encode(
                x=alt.X("column:N", sort=None),
                y=alt.Y("index:O", title="row"),
                color=alt.Color("is_na:N", legend=alt.Legend(title="Missing")),
                tooltip=["column", "index", "is_na"]
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)

# ---- Explore
with tab_explore:
    st.subheader("Column statistics")
    if df_filtered.empty:
        st.info("Load and/or filter data to see stats.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.write("Numerical summary")
            num = df_filtered[numeric_cols(df_filtered)]
            if not num.empty:
                st.dataframe(num.describe().T.round(3), use_container_width=True)
            else:
                st.write("No numeric columns.")
        with c2:
            st.write("Categorical top counts")
            cats = categorical_cols(df_filtered)
            if cats:
                col = st.selectbox("Pick a categorical column", options=cats, index=0)
                st.dataframe(
                    df_filtered[col].value_counts(dropna=False).head(30).rename_axis(col).to_frame("count"),
                    use_container_width=True
                )
            else:
                st.write("No categorical columns.")

        st.divider()
        st.subheader("Row search")
        q = st.text_input("Contains text (case-insensitive):", placeholder="e.g., north, alice, widget…")
        if q:
            mask = pd.Series(False, index=df_filtered.index)
            for c in categorical_cols(df_filtered):
                mask = mask | df_filtered[c].astype(str).str.contains(q, case=False, na=False)
            res = df_filtered[mask]
            st.write(f"Matches: {len(res)}")
            st.dataframe(res.head(200), use_container_width=True)

# ---- Visualize
with tab_visual:
    st.subheader("Build a quick chart")
    if df_filtered.empty:
        st.info("Load data to chart.")
    else:
        cols = df_filtered.columns.tolist()
        num_cols = numeric_cols(df_filtered)
        cat_cols = categorical_cols(df_filtered) + datetime_cols(df_filtered)

        chart_type = st.radio("Chart", ["Bar (aggregate)", "Line (aggregate)", "Scatter"], horizontal=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("X axis", options=cols)
        with c2:
            if chart_type == "Scatter":
                y_col = st.selectbox("Y axis", options=num_cols if num_cols else cols)
            else:
                y_col = st.selectbox("Value / Y (aggregated)", options=num_cols if num_cols else cols)
        with c3:
            color = st.selectbox("Color / Group (optional)", options=[None] + cat_cols)

        agg = "mean"
        if chart_type in {"Bar (aggregate)", "Line (aggregate)"}:
            agg = st.selectbox("Aggregation", ["sum", "mean", "median", "min", "max", "count"], index=0)

        df_plot = df_filtered.copy()

        # Prepare aggregation if needed
        if chart_type in {"Bar (aggregate)", "Line (aggregate)"} and y_col:
            group_cols = [x_col] + ([color] if color else [])
            if agg == "count":
                plot_df = df_plot.groupby(group_cols, dropna=False, as_index=False).size()
                val_col = "size"
            else:
                plot_df = df_plot.groupby(group_cols, dropna=False, as_index=False)[y_col].agg(agg)
                val_col = y_col
        else:
            plot_df = df_plot
            val_col = y_col

        # Build chart
        if chart_type == "Scatter":
            chart = (
                alt.Chart(plot_df)
                .mark_circle(opacity=0.8)
                .encode(
                    x=alt.X(x_col, title=x_col),
                    y=alt.Y(val_col, title=y_col),
                    color=color if color else alt.value("#4c78a8"),
                    tooltip=list(set([x_col, val_col] + ([color] if color else [])))[:8],
                )
                .interactive()
            )
        elif chart_type == "Line (aggregate)":
            chart = (
                alt.Chart(plot_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X(x_col, title=x_col, sort=None),
                    y=alt.Y(val_col, title=f"{agg}({y_col})"),
                    color=color if color else alt.value("#4c78a8"),
                    tooltip=list(set([x_col, val_col] + ([color] if color else [])))[:8],
                )
                .interactive()
            )
        else:  # Bar
            chart = (
                alt.Chart(plot_df)
                .mark_bar()
                .encode(
                    x=alt.X(x_col, title=x_col, sort="-y"),
                    y=alt.Y(val_col, title=f"{agg}({y_col})"),
                    color=color if color else alt.value("#4c78a8"),
                    tooltip=list(set([x_col, val_col] + ([color] if color else [])))[:8],
                )
                .interactive()
            )

        st.altair_chart(chart, use_container_width=True)

# ---- Pivot
with tab_pivot:
    st.subheader("Pivot table")
    if df_filtered.empty:
        st.info("Load data to pivot.")
    else:
        cols = df_filtered.columns.tolist()
        idx = st.multiselect("Index (rows)", options=cols)
        cols_sel = st.multiselect("Columns", options=[c for c in cols if c not in idx])
        vals = st.multiselect("Values", options=[c for c in cols if c not in idx + cols_sel and pd.api.types.is_numeric_dtype(df_filtered[c])])

        agg_map = {"sum": np.sum, "mean": np.mean, "median": np.median, "min": np.min, "max": np.max, "count": "count"}
        agg = st.selectbox("Aggregation", list(agg_map.keys()), index=0)

        if idx and vals:
            try:
                pvt = pd.pivot_table(
                    df_filtered,
                    index=idx,
                    columns=cols_sel if cols_sel else None,
                    values=vals,
                    aggfunc=agg_map[agg],
                    observed=True,
                )
                pvt = pvt.reset_index()
                st.dataframe(pvt, use_container_width=True)
                csv_download(pvt, "pivot.csv")
            except Exception as e:
                st.error(f"Pivot error: {e}")
        else:
            st.info("Select at least one Index and one Value column.")

# ---- Profile
with tab_profile:
    st.subheader("Quick profile")
    if df_filtered.empty:
        st.info("Load data to profile.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.write("Numeric columns")
            num = df_filtered[numeric_cols(df_filtered)]
            if num.empty:
                st.write("No numeric columns.")
            else:
                st.dataframe(num.describe().T.round(3), use_container_width=True)
        with col2:
            st.write("Top 10 missing by column")
            miss = df_filtered.isna().sum().sort_values(ascending=False).head(10)
            st.dataframe(miss.rename("nulls"), use_container_width=True)

        st.divider()
        st.write("Correlation (numeric only)")
        if not df_filtered[numeric_cols(df_filtered)].empty:
            corr = df_filtered[numeric_cols(df_filtered)].corr(numeric_only=True)
            corr_long = corr.reset_index().melt("index", var_name="col", value_name="corr")
            chart = (
                alt.Chart(corr_long)
                .mark_rect()
                .encode(
                    x=alt.X("index:N", title=""),
                    y=alt.Y("col:N", title=""),
                    color=alt.Color("corr:Q", scale=alt.Scale(domain=[-1, 1])),
                    tooltip=["index", "col", alt.Tooltip("corr:Q", format=".2f")]
                )
                .properties(height=360)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No numeric columns to correlate.")

# ---- Settings
with tab_settings:
    st.subheader("Settings")
    st.write("Theme, display & export options")
    show_raw = st.checkbox("Show raw (unfiltered) head", value=False)
    if show_raw and not df_raw.empty:
        st.dataframe(df_raw.head(50), use_container_width=True)

    st.markdown("**Export filtered data**")
    csv_download(df_filtered, "data_filtered.csv")

    st.caption("Tip: set default theme in `.streamlit/config.toml` (optional).")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, pandas, and Altair.")
