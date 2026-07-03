# =============================================================================
# Metacryst: Alloy Tensile Strength Predictor
# =============================================================================
# A Streamlit application that serves a pre-trained Random Forest regression
# model to predict the Ultimate Tensile Strength (UTS) of alloys from their
# chemical compositions.
#
# Usage:
#   streamlit run app.py
#
# Dependencies:
#   pip install streamlit scikit-learn joblib pandas numpy
#
# Required file in the same directory:
#   random_forest_model.pkl  (pre-trained sklearn RandomForestRegressor)
# =============================================================================

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# PAGE CONFIGURATION
# Must be the very first Streamlit call in the script.
# =============================================================================
st.set_page_config(
    page_title="Metacryst — UTS Predictor",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# Injects minimal styling: a steel-blue/silver palette to evoke metallurgy,
# a monospace data-display font for the metric readout, and a subtle header
# rule.  Everything else defers to Streamlit's own spacing and component
# defaults so the UI stays clean without fighting the framework.
# =============================================================================
st.markdown(
    """
    <style>
    /* ── Global tokens ──────────────────────────────────────────────────── */
    :root {
        --steel:   #3A7CA5;   /* primary accent — deep steel blue            */
        --silver:  #D0D6DC;   /* secondary — cold silver                     */
        --forge:   #1C2733;   /* dark background tint for cards              */
        --warn:    #E8A838;   /* amber for the composition warning           */
        --pass:    #2ECC71;   /* green for a valid 100 % composition         */
    }

    /* ── Header rule beneath the main title ─────────────────────────────── */
    .metacryst-header {
        border-bottom: 2px solid var(--steel);
        padding-bottom: 0.4rem;
        margin-bottom: 0.2rem;
    }

    /* ── Composition gauge bar ───────────────────────────────────────────── */
    .gauge-track {
        background: #2C3E50;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
        margin-top: 6px;
        margin-bottom: 2px;
    }
    .gauge-fill {
        height: 10px;
        border-radius: 6px;
        transition: width 0.3s ease;
    }

    /* ── Metric card override: larger, monospace UTS readout ─────────────── */
    [data-testid="stMetricValue"] {
        font-family: "Courier New", Courier, monospace;
        font-size: 2.6rem !important;
        color: var(--steel) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--silver);
    }

    /* ── Sidebar section heading ──────────────────────────────────────────── */
    .sidebar-section {
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--silver);
        margin-top: 1rem;
        margin-bottom: 0.25rem;
        border-bottom: 1px solid #2C3E50;
        padding-bottom: 3px;
    }

    /* ── Predict button: full-width, steel accent ───────────────────────── */
    div[data-testid="stButton"] > button {
        width: 100%;
        background-color: var(--steel);
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 0.65rem 1.2rem;
        transition: background-color 0.2s ease;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #2E6389;
    }

    /* ── Info / explainer box ────────────────────────────────────────────── */
    .info-box {
        background: var(--forge);
        border-left: 4px solid var(--steel);
        border-radius: 4px;
        padding: 0.85rem 1rem;
        font-size: 0.92rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# CONSTANTS
# The exact feature order the model was trained on.  DO NOT reorder.
# =============================================================================
FEATURE_COLS = ["Fe", "Ni", "Co", "Cr", "Mn", "C", "Mo", "Si", "Cu", "Al", "W", "V", "Ti", "Nb"]

# Default composition representing a generic stainless-steel-like alloy
DEFAULTS = {
    "Fe": 85.0, "Cr": 10.0, "Ni": 2.0,  "Mn": 1.0,
    "C":   0.5, "Mo":  0.5, "Si":  0.5,  "Ti": 0.5,
    # all others default to 0.0
    "Co": 0.0, "Cu": 0.0, "Al": 0.0, "W": 0.0, "V": 0.0, "Nb": 0.0,
}

# Periodic-table groupings for visual organisation in the sidebar
SIDEBAR_GROUPS = {
    "Iron Group (Base & Transition)": ["Fe", "Ni", "Co", "Cr", "Mn"],
    "Interstitials & Light Alloying":  ["C",  "Si", "Al"],
    "Refractory & Carbide Formers":    ["Mo", "W",  "V",  "Ti", "Nb"],
    "Specialty Additions":             ["Cu"],
}

# =============================================================================
# MODEL LOADING
# Cached so it is read from disk only once per session, not on every rerun.
# =============================================================================
@st.cache_resource(show_spinner="Loading Random Forest model …")
def load_model():
    """
    Load the pre-trained scikit-learn pipeline/model from disk.
    Returns the model object, or None if the file is missing.
    """
    try:
        model = joblib.load("random_forest_model.pkl")
        return model
    except FileNotFoundError:
        return None

model = load_model()

# =============================================================================
# MAIN AREA — Title & Description
# =============================================================================
st.markdown('<div class="metacryst-header">', unsafe_allow_html=True)
st.title("Metacryst: Alloy Tensile Strength Predictor 🚀")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="info-box">
        <strong>What this tool does:</strong> Enter the chemical composition of a metallic alloy
        (in weight %) and the model will predict its
        <strong>Ultimate Tensile Strength (UTS)</strong> — the maximum stress the material can
        withstand before fracture. Predictions are generated by a
        <em>Random Forest regressor</em> trained on experimental alloy data.<br><br>
        <strong>How to use it:</strong> Adjust the 14 elemental sliders in the
        <em>Alloy Configuration</em> sidebar so that the composition sums to exactly 100 %,
        then press <strong>Predict Tensile Strength</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)

# Surface a hard error early if the model file is absent so the user knows
# immediately rather than after filling in all the inputs.
if model is None:
    st.error(
        "**Model file not found.**  "
        "Place `random_forest_model.pkl` in the same directory as `app.py` and restart the app."
    )

# =============================================================================
# SIDEBAR — Alloy Configuration
# =============================================================================
with st.sidebar:
    st.header("⚙️ Alloy Configuration")
    st.caption("Adjust each element's weight percentage (wt %).")

    element_values: dict[str, float] = {}

    # Render inputs grouped by metallurgical role for readability
    for group_label, elements in SIDEBAR_GROUPS.items():
        st.markdown(f'<div class="sidebar-section">{group_label}</div>', unsafe_allow_html=True)

        for elem in elements:
            element_values[elem] = st.number_input(
                label=f"{elem}  (wt %)",
                min_value=0.0,
                max_value=100.0,
                value=DEFAULTS.get(elem, 0.0),
                step=0.1,
                format="%.2f",
                key=f"elem_{elem}",
                help=f"Weight percentage of {elem} in the alloy.  Range: 0 – 100 wt %.",
            )

    # -------------------------------------------------------------------------
    # COMPOSITION GAUGE — dynamic sum counter + validation
    # -------------------------------------------------------------------------
    st.divider()

    composition_sum = sum(element_values.values())
    delta_from_100  = composition_sum - 100.0
    pct_clamped     = min(max(composition_sum, 0.0), 100.0)  # clamp for the bar width

    # Colour logic: green at 100 %, amber otherwise
    bar_colour = "#2ECC71" if abs(delta_from_100) < 1e-6 else "#E8A838"

    st.markdown("**Composition Total**")
    st.markdown(
        f"""
        <div class="gauge-track">
            <div class="gauge-fill"
                 style="width:{pct_clamped:.1f}%; background:{bar_colour};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Numeric readout with delta colouring
    total_col, delta_col = st.columns([2, 1])
    with total_col:
        st.metric(
            label="Total (wt %)",
            value=f"{composition_sum:.2f}",
            delta=f"{delta_from_100:+.2f} from 100",
            delta_color="off" if abs(delta_from_100) < 1e-6 else "inverse",
        )

    # Warning banner if composition does not sum to 100 %
    if abs(delta_from_100) > 1e-6:
        st.warning(
            f"⚠️ **Composition is {composition_sum:.2f} wt %** — metallurgical compositions "
            f"must total exactly 100 %.  "
            f"Please adjust the elements by **{-delta_from_100:+.2f} wt %**.",
            icon=None,
        )
    else:
        st.success("✅ Composition sums to exactly 100 wt %.")

# =============================================================================
# MAIN AREA — Results Layout
# Two columns: left for the predict button + metric, right for feature breakdown
# =============================================================================
col_predict, col_breakdown = st.columns([1, 1], gap="large")

# ── Left column: inference engine ────────────────────────────────────────────
with col_predict:
    st.subheader("Inference Engine")

    # Assemble inputs in the exact order the model expects
    # Build as a dict first to guarantee column alignment, then convert to DF
    input_dict  = {col: [element_values[col]] for col in FEATURE_COLS}
    input_df    = pd.DataFrame(input_dict, columns=FEATURE_COLS)  # shape (1, 14)

    predict_btn = st.button(
    "🔬 Predict Tensile Strength",
    # This locks the button if the model is missing OR if the sum is not exactly 100%
    disabled=(model is None or abs(delta_from_100) > 1e-6),
    use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if predict_btn:
        if model is None:
            # Belt-and-suspenders guard; the button is already disabled
            st.error("Cannot run inference: model file is missing.")
        else:
            # ----------------------------------------------------------------
            # INFERENCE
            # Pass the 1×14 DataFrame directly to the model.  RandomForest
            # .predict() returns a 1-D ndarray; we take index 0 for the scalar.
            # ----------------------------------------------------------------
            with st.spinner("Running Random Forest inference …"):
                prediction_array = model.predict(input_df)
                uts_psi          = float(prediction_array[0])

            # ── Primary result metric ────────────────────────────────────── #
            st.metric(
                label="Predicted UTS",
                value=f"{uts_psi:,.2f} psi",
            )

            # ── Contextual interpretation ────────────────────────────────── #
            # Soft benchmarks for common structural alloy families (illustrative)
            if uts_psi < 60_000:
                grade, colour = "Low-strength alloy", "#E8A838"
                note = "Typical of annealed low-carbon steels or soft aluminium alloys."
            elif uts_psi < 120_000:
                grade, colour = "Medium-strength alloy", "#3A7CA5"
                note = "Characteristic of structural steels, normalised stainless steels."
            elif uts_psi < 200_000:
                grade, colour = "High-strength alloy", "#2ECC71"
                note = "Common in heat-treated alloy steels and Ni-based superalloys."
            else:
                grade, colour = "Ultra-high-strength alloy", "#E74C3C"
                note = "Maraging steels, precipitation-hardened grades, or speciality composites."

            st.markdown(
                f"""
                <div style="
                    border-left: 4px solid {colour};
                    background: var(--forge, #1C2733);
                    border-radius: 4px;
                    padding: 0.75rem 1rem;
                    margin-top: 0.5rem;
                    font-size: 0.9rem;
                ">
                    <strong style="color:{colour};">{grade}</strong><br>
                    <span style="color:#B0BEC5;">{note}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Raw model input echoed back for auditability ─────────────── #
            with st.expander("📋 Input vector sent to model", expanded=False):
                st.dataframe(input_df, use_container_width=True, hide_index=True)

    elif model is not None:
        # Placeholder before the user clicks Predict
        st.info(
            "Configure the alloy composition in the sidebar, verify the total "
            "equals 100 wt %, then click **Predict Tensile Strength**.",
            icon="ℹ️",
        )

# ── Right column: composition breakdown ──────────────────────────────────────
with col_breakdown:
    st.subheader("Composition Breakdown")

    # Build a display DataFrame sorted by weight % descending
    breakdown_data = (
        pd.DataFrame(
            {"Element": FEATURE_COLS,
             "Weight %": [element_values[c] for c in FEATURE_COLS]}
        )
        .query("`Weight %` > 0.0")          # hide absent elements
        .sort_values("Weight %", ascending=False)
        .reset_index(drop=True)
    )

    if breakdown_data.empty:
        st.warning("All elements are set to 0.0 wt %.  Adjust the sidebar sliders.")
    else:
        # Horizontal bar chart — concise at a glance
        st.bar_chart(
            breakdown_data.set_index("Element")["Weight %"],
            use_container_width=True,
            height=320,
            color="#3A7CA5",
        )

        # Table with percentage contribution column
        breakdown_data["Share of total (%)"] = (
            breakdown_data["Weight %"] / composition_sum * 100
            if composition_sum > 0 else 0.0
        ).round(2)

        st.dataframe(
            breakdown_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Weight %": st.column_config.NumberColumn(format="%.2f wt %"),
                "Share of total (%)": st.column_config.ProgressColumn(
                    min_value=0,
                    max_value=100,
                    format="%.1f %%",
                ),
            },
        )

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.caption(
    "Metacryst v1.0  ·  Random Forest Regression  ·  "
    "Predictions are for research guidance only and should be validated "
    "against experimental data before engineering use."
)