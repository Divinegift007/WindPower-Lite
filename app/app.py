# app/app.py
import os
import sys
from datetime import datetime, timedelta, timezone, date

# ensure src is importable when running from project root
ROOT = os.path.dirname(os.path.dirname(__file__))  # project root (windpower_lite/)
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# map/support libraries for wizard
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# local imports (from src)
try:
    from main import run_pipeline
    from utils import load_turbine_specs
except Exception as e:
    st.error(f"Failed to import local modules. Make sure `src/` is on PYTHONPATH. Error: {e}")
    raise

# ----- UI defaults -----
DEFAULT_LAT = 8.4966
DEFAULT_LON = 4.5421

# initialize session state values that persist across tabs
if "map_lat" not in st.session_state:
    st.session_state.map_lat = DEFAULT_LAT
if "map_lon" not in st.session_state:
    st.session_state.map_lon = DEFAULT_LON
if "map_address" not in st.session_state:
    st.session_state.map_address = "Ilorin, Nigeria"
if "marker_added" not in st.session_state:
    st.session_state.marker_added = False

if "start_date" not in st.session_state:
    st.session_state.start_date = None
if "end_date" not in st.session_state:
    st.session_state.end_date = None
if "turbine_sel" not in st.session_state:
    st.session_state.turbine_sel = None
if "apply_hybrid" not in st.session_state:
    st.session_state.apply_hybrid = True

if "df_results" not in st.session_state:
    st.session_state.df_results = None
if "summary" not in st.session_state:
    st.session_state.summary = None

st.set_page_config(
    page_title="Windpower Lite — Farm-Scale Estimator", layout="wide"
)

st.markdown(
    '''
    <style>
    .stApp, .streamlit-expanderHeader {
        max-width: 100% !important;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)

st.title("Windpower Lite — Farm-Scale Wind Output Estimator")
st.markdown(
    "Estimate hourly power, AEP and capacity factor for a turbine at a chosen site. "
    "Data fetched from NASA POWER (hourly). Hybrid correction applied if a trained model exists."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_datetime(dt_date: date) -> datetime:
    return datetime(dt_date.year, dt_date.month, dt_date.day, tzinfo=timezone.utc)


def recommendation_text(capacity_factor: float) -> str:
    if capacity_factor is None or (capacity_factor != capacity_factor):
        return "Capacity factor could not be computed."
    cf_pct = capacity_factor * 100.0
    if capacity_factor < 0.15:
        return (
            f"Low wind potential (capacity factor ≈ {cf_pct:.1f}%). "
            "Not recommended for standalone wind projects; consider hybrid "
            "solar-wind systems or taller hubs."
        )
    if 0.15 <= capacity_factor < 0.25:
        return (
            f"Marginal wind potential (capacity factor ≈ {cf_pct:.1f}%). "
            "May be viable for small-scale or supportive roles (pumping, "
            "battery-charged systems). Optimise hub height and turbine selection."
        )
    if 0.25 <= capacity_factor < 0.35:
        return (
            f"Moderate wind potential (capacity factor ≈ {cf_pct:.1f}%). "
            "Potentially viable for farm-scale projects; perform more focused "
            "on-site measurement and economic analysis."
        )
    return (
        f"Good wind potential (capacity factor ≈ {cf_pct:.1f}%). "
        "Site looks promising for dedicated wind deployment; proceed to detailed "
        "site assessment and wake-loss modelling."
    )


def build_sankey(
    aep_physics_kwh: float,
    aep_hybrid_kwh: float,
    rated_power_kw: float,
    n_hours: int,
) -> go.Figure:
    """
    Build a Sankey diagram showing energy flow through the Windpower Lite pipeline.

    Nodes (left to right):
      0  Available Wind Energy
      1  Betz Limit Loss
      2  Aerodynamic / Cp Loss
      3  Physics Baseline Output
      4  ML Correction (Hybrid Gain)
      5  Final Hybrid Output
      6  Unused Capacity
    """
    # ── Energy quantities ─────────────────────────────────────────────────────
    # Maximum possible output if turbine ran at rated power all year
    max_possible_kwh    = rated_power_kw * n_hours

    # Physics baseline and hybrid in Wh then kWh already supplied
    physics_kwh         = float(aep_physics_kwh)
    hybrid_kwh          = float(aep_hybrid_kwh)
    ml_correction_kwh   = hybrid_kwh - physics_kwh  # can be negative

    # Available wind energy — back-calculate from Betz limit
    # P_available = P_physics / (Cp_base * Betz_efficiency_ratio)
    # We use a simplified model: available = physics / (0.40 / 0.593)
    betz_limit          = 0.593
    cp_base             = 0.40
    available_kwh       = physics_kwh / cp_base  # total kinetic energy that passed through rotor

    betz_loss_kwh       = available_kwh * (1.0 - betz_limit)
    extractable_kwh     = available_kwh * betz_limit
    cp_loss_kwh         = extractable_kwh - physics_kwh
    unused_capacity_kwh = max(0.0, max_possible_kwh - hybrid_kwh)

    # Clamp ML correction for display — negative means hybrid < physics
    ml_gain_kwh         = max(0.0, ml_correction_kwh)
    ml_loss_kwh         = max(0.0, -ml_correction_kwh)

    # ── Colours ───────────────────────────────────────────────────────────────
    node_colors = [
        "#12B5C8",  # 0 Available Wind Energy  — teal
        "#DC2626",  # 1 Betz Loss              — red
        "#F5A623",  # 2 Cp / Aerodynamic Loss  — amber
        "#0A7E8C",  # 3 Physics Baseline       — dark teal
        "#16A34A",  # 4 ML Correction Gain     — green
        "#0D2137",  # 5 Final Hybrid Output    — navy
        "#64748B",  # 6 Unused Capacity        — grey
    ]

    # ── Build links ───────────────────────────────────────────────────────────
    sources = [0, 0, 2, 3]
    targets = [1, 2, 5, 5]
    values  = [
        betz_loss_kwh,
        extractable_kwh,
        cp_loss_kwh,
        physics_kwh,
    ]
    link_labels = [
        "Betz Limit Loss",
        "Extractable Energy",
        "Cp & Aerodynamic Loss",
        "Physics Baseline",
    ]
    link_colors = [
        "rgba(220,38,38,0.4)",
        "rgba(18,181,200,0.4)",
        "rgba(245,166,35,0.4)",
        "rgba(10,126,140,0.4)",
    ]

    # Add ML gain link if positive
    if ml_gain_kwh > 0:
        sources.append(4)
        targets.append(5)
        values.append(ml_gain_kwh)
        link_labels.append("ML Correction Gain")
        link_colors.append("rgba(22,163,74,0.4)")

    # Add unused capacity link
    if unused_capacity_kwh > 0:
        sources.append(5)
        targets.append(6)
        values.append(unused_capacity_kwh)
        link_labels.append("Unused Capacity")
        link_colors.append("rgba(100,116,139,0.3)")

    node_labels = [
        f"Available Wind<br>{available_kwh:,.0f} kWh",
        f"Betz Loss<br>{betz_loss_kwh:,.0f} kWh",
        f"Cp / Aero Loss<br>{cp_loss_kwh:,.0f} kWh",
        f"Physics Baseline<br>{physics_kwh:,.0f} kWh",
        "ML Correction",
        f"Hybrid Output<br>{hybrid_kwh:,.0f} kWh",
        f"Unused Capacity<br>{unused_capacity_kwh:,.0f} kWh",
    ]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=node_labels,
            color=node_colors,
            pad=20,
            thickness=25,
            line=dict(color="white", width=0.5),
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color=link_colors,
        ),
    ))

    fig.update_layout(
        title_text="Energy Flow — Windpower Lite Pipeline",
        title_font_size=16,
        font_size=12,
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_pie(
    aep_physics_kwh: float,
    aep_hybrid_kwh: float,
    rated_power_kw: float,
    n_hours: int,
) -> go.Figure:
    """
    Pie chart showing proportional energy distribution across the pipeline stages.
    """
    betz_limit       = 0.593
    cp_base          = 0.40
    available_kwh    = float(aep_physics_kwh) / cp_base
    betz_loss_kwh    = available_kwh * (1.0 - betz_limit)
    extractable_kwh  = available_kwh * betz_limit
    cp_loss_kwh      = extractable_kwh - float(aep_physics_kwh)
    ml_gain_kwh      = max(0.0, float(aep_hybrid_kwh) - float(aep_physics_kwh))
    final_kwh        = float(aep_hybrid_kwh)

    labels = ["Betz Limit Loss", "Cp & Aerodynamic Loss", "ML Correction Gain", "Final Hybrid Output"]
    values = [betz_loss_kwh, cp_loss_kwh, ml_gain_kwh, final_kwh]
    colors = ["#DC2626", "#F5A623", "#16A34A", "#0A7E8C"]

    # Remove zero or negative slices
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not filtered:
        return None
    labels, values, colors = zip(*filtered)

    fig = go.Figure(go.Pie(
        labels=labels,
        values=list(values),
        marker=dict(colors=list(colors), line=dict(color="white", width=2)),
        textinfo="label+percent",
        hovertemplate="%{label}<br>%{value:,.0f} kWh<br>%{percent}<extra></extra>",
        hole=0.35,
    ))

    fig.update_layout(
        title_text="Energy Distribution (kWh)",
        title_font_size=16,
        height=420,
        legend=dict(orientation="v", x=1.02, y=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Wizard tabs ───────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["1. Site selection", "2. Configuration", "3. Analysis"])

# ── Tab 1 — Site selection ────────────────────────────────────────────────────
with tab1:
    st.header("Select site on map")
    with st.form(key="search_form"):
        search_input     = st.text_input("Search for a location (e.g., 'Sokoto')", key="search_input")
        search_submitted = st.form_submit_button("Search")

        if search_submitted and search_input:
            geolocator = Nominatim(user_agent="windpower_lite_app")
            try:
                location = geolocator.geocode(search_input)
                if location:
                    st.session_state.map_lat     = location.latitude
                    st.session_state.map_lon     = location.longitude
                    st.session_state.map_address = location.address
                    st.session_state.marker_added = True
                    st.experimental_rerun()
                else:
                    st.error("Location not found.")
            except Exception as e:
                st.error(f"Error: {e}")

    m = folium.Map(
        location=[st.session_state.map_lat, st.session_state.map_lon],
        zoom_start=12,
    )
    if st.session_state.marker_added:
        folium.Marker(
            [st.session_state.map_lat, st.session_state.map_lon],
            popup=st.session_state.map_address,
            icon=folium.Icon(color="green", icon="bolt", prefix="fa"),
        ).add_to(m)

    map_data = st_folium(m, width=1000, height=800)

    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]
        if (clicked_lat, clicked_lon) != (
            st.session_state.map_lat, st.session_state.map_lon
        ):
            st.session_state.map_lat = clicked_lat
            st.session_state.map_lon = clicked_lon
            try:
                geolocator = Nominatim(user_agent="windpower_lite_app")
                loc = geolocator.reverse((clicked_lat, clicked_lon), language="en")
                st.session_state.map_address = (
                    loc.address if loc else "Unknown Location"
                )
            except GeocoderTimedOut:
                st.session_state.map_address = "Error: Geocoding timed out"
            except Exception as e:
                st.session_state.map_address = f"Error: {e}"
            st.session_state.marker_added = True
            st.rerun()

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Address:** {st.session_state.map_address}")
    with col2:
        st.success(
            f"**Coordinates:** {st.session_state.map_lat:.5f}, "
            f"{st.session_state.map_lon:.5f}"
        )

# ── Tab 2 — Configuration ─────────────────────────────────────────────────────
with tab2:
    st.header("Configuration and inputs")

    lat = st.number_input(
        "Latitude", value=float(st.session_state.map_lat), format="%.6f"
    )
    lon = st.number_input(
        "Longitude", value=float(st.session_state.map_lon), format="%.6f"
    )
    st.session_state.map_lat = lat
    st.session_state.map_lon = lon

    today         = date.today()
    default_end   = today
    default_start = today - timedelta(days=365)
    start_date    = st.date_input(
        "Start date",
        value=st.session_state.start_date or default_start,
    )
    end_date = st.date_input(
        "End date",
        value=st.session_state.end_date or default_end,
    )
    st.session_state.start_date = start_date
    st.session_state.end_date   = end_date

    try:
        specs        = load_turbine_specs()
        turbine_names = list(specs.keys())
    except Exception:
        turbine_names = []

    turbine_sel = st.selectbox(
        "Turbine model",
        options=turbine_names,
        index=(
            turbine_names.index(st.session_state.turbine_sel)
            if st.session_state.turbine_sel in turbine_names
            else 0
        ),
    )
    st.session_state.turbine_sel = turbine_sel

    apply_hybrid = st.checkbox(
        "Apply hybrid correction (if model available)",
        value=st.session_state.apply_hybrid,
    )
    st.session_state.apply_hybrid = apply_hybrid

# ── Tab 3 — Analysis ──────────────────────────────────────────────────────────
with tab3:
    st.header("Run analysis")
    st.markdown("Click the button below to fetch data and generate estimates.")

    if st.button("Run estimation", key="run_analysis"):
        start_dt = to_datetime(st.session_state.start_date)
        end_dt   = to_datetime(st.session_state.end_date)
        with st.spinner(
            "Fetching data and running pipeline "
            "(this may take 30–90 s for 12 months hourly)..."
        ):
            try:
                bundle_path      = os.path.join(ROOT, "models", "hybrid_all.joblib")
                model_path_to_use = bundle_path if os.path.exists(bundle_path) else None
                if model_path_to_use:
                    st.info(f"Using hybrid bundle: {os.path.basename(model_path_to_use)}")
                df, summ = run_pipeline(
                    lat=float(st.session_state.map_lat),
                    lon=float(st.session_state.map_lon),
                    start_dt=start_dt,
                    end_dt=end_dt,
                    turbine_name=(
                        st.session_state.turbine_sel
                        if st.session_state.turbine_sel
                        else None
                    ),
                    turbine_specs_csv=None,
                    apply_hybrid_if_available=st.session_state.apply_hybrid,
                    hybrid_model_path=model_path_to_use,
                )
                st.session_state.df_results = df
                st.session_state.summary    = summ
            except Exception as e:
                st.exception(f"Pipeline failed: {e}")
                st.stop()
        st.success("Pipeline completed.")

    if st.session_state.summary is not None:
        summary    = st.session_state.summary
        df_results = st.session_state.df_results

        # ── KPI metrics ───────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        c1.metric("AEP (Physics) [kWh]", f"{summary.get('aep_physics_kwh', 0):,.0f}")
        aep_hybrid = summary.get("aep_hybrid_kwh")
        c2.metric(
            "AEP (Hybrid) [kWh]",
            f"{aep_hybrid:,.0f}" if aep_hybrid else "N/A",
        )
        cf_phys   = summary.get("capacity_factor_physics")
        cf_hybrid = summary.get("capacity_factor_hybrid")
        c3.metric(
            "Capacity factor (Physics)",
            f"{cf_phys*100:.2f}%" if cf_phys else "N/A",
        )
        c4.metric(
            "Capacity factor (Hybrid)",
            f"{cf_hybrid*100:.2f}%" if cf_hybrid else "N/A",
        )

        # ── Recommendation ────────────────────────────────────────────────────
        st.markdown("### Recommendation")
        rec = recommendation_text(cf_hybrid if cf_hybrid else cf_phys)
        st.info(rec)

        # ── Time series ───────────────────────────────────────────────────────
        st.markdown("### Time series")
        df_plot = df_results.copy()
        df_plot.replace({-999: pd.NA, -9999: pd.NA}, inplace=True)
        if "timestamp" in df_plot.columns:
            df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
        else:
            df_plot = df_plot.reset_index().rename(columns={"index": "timestamp"})
            df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
        if "wind_speed_50m_mps" in df_plot.columns:
            df_plot = df_plot[df_plot["wind_speed_50m_mps"].notna()]
        if "v_hub_mps" in df_plot.columns:
            df_plot = df_plot[df_plot["v_hub_mps"].notna()]

        fig_power = px.line(title="Power output (kW) over time")
        fig_power.add_scatter(
            x=df_plot["timestamp"],
            y=df_plot["P_physics_w"] / 1000.0,
            mode="lines",
            name="Physics (kW)",
        )
        if "P_hybrid_w" in df_plot.columns:
            fig_power.add_scatter(
                x=df_plot["timestamp"],
                y=df_plot["P_hybrid_w"] / 1000.0,
                mode="lines",
                name="Hybrid (kW)",
            )
        fig_power.update_xaxes(title_text="Time")
        fig_power.update_yaxes(title_text="Power (kW)")
        st.plotly_chart(fig_power, use_container_width=True)

        # ── NEW — Energy flow Sankey ───────────────────────────────────────────
        st.markdown("### Energy flow — pipeline losses and corrections")
        st.caption(
            "Shows how available wind kinetic energy is reduced through "
            "Betz limit and aerodynamic losses to produce the physics baseline, "
            "then corrected upward by the hybrid ML layer to produce final output."
        )
        try:
            turbine_spec = load_turbine_specs().get(
                st.session_state.turbine_sel, {}
            )
            rated_kw  = float(turbine_spec.get("rated_power_kw", 2000))
            n_hours   = int(summary.get("hours", 8760))
            aep_phys  = float(summary.get("aep_physics_kwh", 0))
            aep_hyb   = float(summary.get("aep_hybrid_kwh") or aep_phys)

            fig_sankey = build_sankey(aep_phys, aep_hyb, rated_kw, n_hours)
            st.plotly_chart(fig_sankey, use_container_width=True)
        except Exception as e:
            st.warning(f"Sankey diagram could not be generated: {e}")

        # ── NEW — Energy distribution pie chart ───────────────────────────────
        st.markdown("### Energy distribution")
        st.caption(
            "Proportional breakdown of where the available wind energy goes — "
            "losses vs useful output."
        )
        try:
            fig_pie = build_pie(aep_phys, aep_hyb, rated_kw, n_hours)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
        except Exception as e:
            st.warning(f"Pie chart could not be generated: {e}")

        # ── Wind speed charts ─────────────────────────────────────────────────
        st.markdown("### Wind speed at 50 m")
        fig_ws = px.line(
            df_plot, x="timestamp", y="wind_speed_50m_mps",
            title="Wind speed at 50 m (m/s)",
        )
        fig_ws.update_xaxes(title_text="Time")
        fig_ws.update_yaxes(title_text="Wind speed (m/s)")
        st.plotly_chart(fig_ws, use_container_width=True)

        st.markdown("### Hub-height wind speed distribution")
        fig_hist = px.histogram(
            df_plot, x="v_hub_mps", nbins=40,
            title="Hub-height wind speed distribution (m/s)",
        )
        fig_hist.update_xaxes(title_text="Hub-height wind speed (m/s)")
        fig_hist.update_yaxes(title_text="Hours per year")
        fig_hist.update_traces(marker_line_width=2, marker_line_color="black")
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Sample data and download ──────────────────────────────────────────
        st.markdown("### Sample data (first 10 rows)")
        display_cols = ["timestamp", "wind_speed_50m_mps", "v_hub_mps", "P_physics_w"]
        if "P_hybrid_w" in df_plot.columns:
            display_cols.append("P_hybrid_w")
        st.dataframe(df_plot[display_cols].head(10))

        st.markdown("### Pipeline summary")
        st.json(summary)

        csv = df_plot.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results CSV",
            csv,
            file_name="windpower_results.csv",
            mime="text/csv",
        )