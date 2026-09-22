"""
Kerala Weather Forecast — Streamlit App
========================================
LSTM-based 7/14-day temperature forecast for Trivandrum, Kerala.

Artifacts expected in model_artifacts/:
  model_14day.keras
  scaler_X.pkl
  scaler_y.pkl
  features.pkl

These are produced by notebooks/model_training.ipynb — run that notebook
once before starting the app.

Run locally : streamlit run app.py
Deploy      : Push to GitHub -> Streamlit Community Cloud
"""

import streamlit as st
import numpy as np
import pandas as pd
import requests
import joblib
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta, date
from tensorflow.keras.models import load_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kerala Weather Forecast",
    page_icon="🌦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── TEMPORARY DEBUG — remove once the artifact-loading issue is fixed ────────
import os
st.write("CWD:", os.getcwd())
st.write("Root files:", os.listdir("."))
if os.path.exists("model_artifacts"):
    st.write("model_artifacts contents:", os.listdir("model_artifacts"))
else:
    st.write("model_artifacts folder NOT found here")
# ──────────────────────────────────────────────────────────────────────────────

# ── Constants ─────────────────────────────────────────────────────────────────
LAT, LON      = 8.5241, 76.9366
ARTIFACTS_DIR = "model_artifacts"

CONDITION_EMOJI = {
    "Hot & Sunny"                    : "☀️",
    "Warm & Mostly Sunny"            : "🌤",
    "Clear & Sunny"                  : "☀️",
    "Partly Cloudy & Pleasant"       : "🌥",
    "Pleasant"                       : "😊",
    "Very Hot, Risk of Thunderstorms": "🌩",
    "Hot with Chance of Showers"     : "🌦",
    "Heavy Rain / Overcast"          : "🌧",
    "Rainy & Humid"                  : "🌧",
    "Intermittent Showers"           : "🌦",
    "Rainy / Humid"                  : "🌧",
    "Chance of Showers"              : "🌦",
    "Partly Cloudy"                  : "⛅",
}

CONDITION_COLOR = {
    "Hot & Sunny"                    : "#F97316",
    "Warm & Mostly Sunny"            : "#FACC15",
    "Clear & Sunny"                  : "#FDE047",
    "Partly Cloudy & Pleasant"       : "#86EFAC",
    "Pleasant"                       : "#34D399",
    "Very Hot, Risk of Thunderstorms": "#EF4444",
    "Hot with Chance of Showers"     : "#FB923C",
    "Heavy Rain / Overcast"          : "#3B82F6",
    "Rainy & Humid"                  : "#6366F1",
    "Intermittent Showers"           : "#818CF8",
    "Rainy / Humid"                  : "#4F46E5",
    "Chance of Showers"              : "#A78BFA",
    "Partly Cloudy"                  : "#94A3B8",
}

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model    = load_model(f"{ARTIFACTS_DIR}/model_14day.keras")
    scaler_X = joblib.load(f"{ARTIFACTS_DIR}/scaler_X.pkl")
    scaler_y = joblib.load(f"{ARTIFACTS_DIR}/scaler_y.pkl")
    features = joblib.load(f"{ARTIFACTS_DIR}/features.pkl")
    return model, scaler_X, scaler_y, features

# ── Fetch recent weather ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching weather data...")
def fetch_recent_weather(seed_date=None, past_days=30):
    end_date = seed_date if seed_date else datetime.now().date()

    if end_date < datetime.now().date():
        url    = "https://archive-api.open-meteo.com/v1/archive"
        start  = (pd.Timestamp(end_date) - pd.Timedelta(days=past_days)).date()
        params = {
            "latitude"  : LAT, "longitude": LON,
            "daily"     : ["temperature_2m_max", "temperature_2m_min",
                           "temperature_2m_mean", "relative_humidity_2m_mean",
                           "windspeed_10m_max", "precipitation_sum",
                           "dewpoint_2m_mean"],
            "start_date": str(start),
            "end_date"  : str(end_date),
            "timezone"  : "Asia/Kolkata",
        }
    else:
        url    = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude"      : LAT, "longitude": LON,
            "daily"         : ["temperature_2m_max", "temperature_2m_min",
                               "temperature_2m_mean", "relative_humidity_2m_mean",
                               "windspeed_10m_max", "precipitation_sum",
                               "dewpoint_2m_mean"],
            "past_days"     : past_days,
            "forecast_days" : 1,
            "timezone"      : "Asia/Kolkata",
        }

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()["daily"]

    df = pd.DataFrame({
        "DATE"       : pd.to_datetime(data["time"]),
        "T2M"        : data["temperature_2m_mean"],
        "T2M_MAX"    : data["temperature_2m_max"],
        "T2M_MIN"    : data["temperature_2m_min"],
        "RH2M"       : data["relative_humidity_2m_mean"],
        "WS2M"       : data["windspeed_10m_max"],
        "PRECTOTCORR": data["precipitation_sum"],
        "T2MDEW"     : data.get("dewpoint_2m_mean", [None] * len(data["time"])),
        "PS"         : [100.0] * len(data["time"]),
    }).set_index("DATE")

    df.replace([None], np.nan, inplace=True)
    df.interpolate(method="time", inplace=True)
    return df

# ── Weather classifier (matches training notebook exactly) ──────────────────
def classify_weather(t2m, t2m_max, t2m_min, month):
    tr = t2m_max - t2m_min
    if month in (12, 1, 2, 3):
        if t2m_max >= 35:                      return "Hot & Sunny"
        if t2m_max >= 32:                      return "Warm & Mostly Sunny"
        if tr >= 7:                            return "Clear & Sunny"
        if 27 <= t2m <= 31:                    return "Partly Cloudy & Pleasant"
        return "Pleasant"
    if month in (4, 5):
        if t2m_max >= 36:                      return "Very Hot, Risk of Thunderstorms"
        if t2m_max >= 33:                      return "Hot & Sunny"
        if tr >= 8:                            return "Hot with Chance of Showers"
        return "Warm & Mostly Sunny"
    if month in (6, 7, 8, 9):
        if tr <= 4:                            return "Heavy Rain / Overcast"
        if tr <= 6:                            return "Rainy & Humid"
        return "Intermittent Showers"
    if month in (10, 11):
        if t2m_min >= 24 and tr <= 6:          return "Rainy / Humid"
        if tr >= 7:                            return "Chance of Showers"
        if t2m_max >= 32:                      return "Warm & Mostly Sunny"
        return "Partly Cloudy"
    return "Partly Cloudy"

# ── Build feature row ─────────────────────────────────────────────────────────
def build_row(current, recent_t2m, dt, features):
    m, d, w = dt.month, dt.dayofyear, dt.dayofweek
    T     = current.get("T2M",     recent_t2m[-1])
    RH    = current.get("RH2M",    75.0)
    T_max = current.get("T2M_MAX", T + 4)
    T_min = current.get("T2M_MIN", T - 3)
    T_dew = current.get("T2MDEW",  T - 3)
    hi    = max(T, -8.78 + 1.61 * T + 2.34 * RH - 0.15 * T * RH
                - 0.012 * T * T - 0.016 * RH * RH)

    lag1  = recent_t2m[-1]
    lag2  = recent_t2m[-2]  if len(recent_t2m) >= 2  else lag1
    lag3  = recent_t2m[-3]  if len(recent_t2m) >= 3  else lag1
    lag7  = recent_t2m[-7]  if len(recent_t2m) >= 7  else lag1
    lag14 = recent_t2m[-14] if len(recent_t2m) >= 14 else lag1
    r7    = recent_t2m[-7:]  if len(recent_t2m) >= 7  else recent_t2m
    r14   = recent_t2m[-14:] if len(recent_t2m) >= 14 else recent_t2m
    r30   = recent_t2m[-30:] if len(recent_t2m) >= 30 else recent_t2m

    base = {
        "T2M"             : T,
        "T2M_MAX"         : T_max,
        "T2M_MIN"         : T_min,
        "RH2M"            : RH,
        "PRECTOTCORR"     : current.get("PRECTOTCORR", 0.0),
        "PS"              : current.get("PS", 100.0),
        "WS2M"            : current.get("WS2M", 3.0),
        "T2MDEW"          : T_dew,
        "month_sin"       : np.sin(2 * np.pi * m / 12),
        "month_cos"       : np.cos(2 * np.pi * m / 12),
        "doy_sin"         : np.sin(2 * np.pi * d / 365),
        "doy_cos"         : np.cos(2 * np.pi * d / 365),
        "dow_sin"         : np.sin(2 * np.pi * w / 7),
        "dow_cos"         : np.cos(2 * np.pi * w / 7),
        "T2M_lag1"        : lag1,
        "T2M_lag2"        : lag2,
        "T2M_lag3"        : lag3,
        "T2M_lag7"        : lag7,
        "T2M_lag14"       : lag14,
        "T2M_MAX_lag1"    : T_max,
        "T2M_MAX_lag2"    : T_max,
        "T2M_MAX_lag3"    : T_max,
        "T2M_MAX_lag7"    : T_max,
        "T2M_MAX_lag14"   : T_max,
        "T2M_roll7_mean"  : float(np.mean(r7)),
        "T2M_roll7_std"   : float(np.std(r7)),
        "T2M_roll14_mean" : float(np.mean(r14)),
        "T2M_roll14_std"  : float(np.std(r14)),
        "T2M_roll30_mean" : float(np.mean(r30)),
        "T2M_roll30_std"  : float(np.std(r30)),
        "rain_roll7_sum"  : current.get("PRECTOTCORR", 0.0) * 7,
        "rain_roll14_sum" : current.get("PRECTOTCORR", 0.0) * 14,
        "diurnal_range"   : T_max - T_min,
        "dew_depression"  : T - T_dew,
        "heat_index"      : hi,
    }

    missing = [f for f in features if f not in base]
    if missing:
        raise KeyError(f"Features in features.pkl not in build_row(): {missing}")

    return pd.DataFrame([[base[f] for f in features]], columns=features)

# ── Forecast ──────────────────────────────────────────────────────────────────
def forecast_n_days(recent_df, model, scaler_X, scaler_y, features, horizon=14):
    recent_t2m   = list(recent_df["T2M"].values[-30:])
    current      = recent_df.iloc[-1].to_dict()
    current_date = recent_df.index[-1]
    results      = []

    for _ in range(horizon):
        current_date = current_date + pd.Timedelta(days=1)
        row    = build_row(current, recent_t2m, current_date, features)
        scaled = scaler_X.transform(row).reshape(1, 1, len(features))
        vals   = scaler_y.inverse_transform(
                     model.predict(scaled, verbose=0))[0]
        t2m, t2m_max, t2m_min = vals[0], vals[1], vals[2]
        cond = classify_weather(t2m, t2m_max, t2m_min, current_date.month)

        results.append({
            "Date"     : current_date.date(),
            "T2M"      : round(t2m,     2),
            "T2M_MAX"  : round(t2m_max, 2),
            "T2M_MIN"  : round(t2m_min, 2),
            "Condition": cond,
        })

        recent_t2m.append(t2m)
        if len(recent_t2m) > 30:
            recent_t2m.pop(0)
        current = {
            "T2M": t2m, "T2M_MAX": t2m_max, "T2M_MIN": t2m_min,
            "RH2M"       : current.get("RH2M", 75.0),
            "PRECTOTCORR": current.get("PRECTOTCORR", 0.0),
            "PS"         : current.get("PS", 100.0),
            "WS2M"       : current.get("WS2M", 3.0),
            "T2MDEW"     : t2m - 3,
        }

    return pd.DataFrame(results)

# ── Temperature chart ─────────────────────────────────────────────────────────
def make_temp_chart(recent_df, forecast_df):
    """Builds a combined observed + forecast DataFrame for st.line_chart."""
    obs = pd.DataFrame({
        "Observed max" : recent_df["T2M_MAX"],
        "Observed min" : recent_df["T2M_MIN"],
    }, index=recent_df.index)

    fc_idx = pd.to_datetime(forecast_df["Date"])
    fc = pd.DataFrame({
        "Forecast max"  : forecast_df["T2M_MAX"].values,
        "Forecast mean" : forecast_df["T2M"].values,
        "Forecast min"  : forecast_df["T2M_MIN"].values,
    }, index=fc_idx)

    combined = pd.concat([obs, fc], axis=0).sort_index()
    return combined

# ── Weather cards ─────────────────────────────────────────────────────────────
def render_weather_cards(forecast_df):
    cols = st.columns(len(forecast_df))
    for col, (_, row) in zip(cols, forecast_df.iterrows()):
        cond  = row["Condition"]
        emoji = CONDITION_EMOJI.get(cond, "🌡")
        color = CONDITION_COLOR.get(cond, "#94A3B8")
        d     = pd.to_datetime(row["Date"])
        with col:
            st.markdown(
                f"""
                <div style="background:{color}28;border:2px solid {color}88;
                    border-radius:14px;padding:14px 8px;text-align:center;
                    height:200px;display:flex;flex-direction:column;
                    justify-content:space-between;">
                    <div style="font-size:12px;color:#94a3b8;font-weight:600;
                                letter-spacing:0.5px;">
                        {d.strftime("%a").upper()}<br>
                        <span style="font-size:13px;color:#cbd5e1;">
                            {d.strftime("%b %d")}
                        </span>
                    </div>
                    <div style="font-size:32px;line-height:1;">{emoji}</div>
                    <div style="font-size:22px;font-weight:800;color:#f1f5f9;
                                letter-spacing:-0.5px;">
                        {row['T2M']:.1f}°C
                    </div>
                    <div style="font-size:12px;font-weight:600;color:#cbd5e1;">
                        ↑{row['T2M_MAX']:.1f}° &nbsp;·&nbsp; ↓{row['T2M_MIN']:.1f}°
                    </div>
                    <div style="font-size:10px;color:#94a3b8;margin-top:2px;
                                line-height:1.3;">
                        {cond}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():

    with st.sidebar:
        st.markdown("## 🌦 Kerala Weather Forecast")
        st.caption("Trivandrum · LSTM Deep Learning")
        st.markdown("---")

        horizon = st.radio(
            "Forecast horizon",
            options=[7, 14],
            format_func=lambda x: f"{x} days",
            horizontal=True,
        )

        st.markdown("---")
        st.markdown("**Forecast start date**")
        st.caption(
            "Model uses real observed weather up to this date as seed. "
            "Default is March 1 — first day after training cutoff."
        )
        forecast_start = st.date_input(
            "Start from",
            value=date(2026, 3, 1),
            min_value=date(2026, 3, 1),
            max_value=datetime.now().date(),
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Data**")
        st.markdown(
            "🗓 Training cutoff: **28 Feb 2026**  \n"
            "📡 Live seed: [Open-Meteo](https://open-meteo.com/) (free)  \n"
            f"📍 Trivandrum `{LAT}°N, {LON}°E`"
        )

        st.markdown("---")
        show_metrics = st.toggle("Show model metrics", value=False)
        show_table   = st.toggle("Show forecast table", value=False)

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        model, scaler_X, scaler_y, features = load_artifacts()
    except Exception as e:
        st.error(
            f"Could not load artifacts from `{ARTIFACTS_DIR}/`.  \n"
            f"Run notebooks/model_training.ipynb and save artifacts first.  \n`{e}`"
        )
        st.stop()

    # ── Fetch seed data ───────────────────────────────────────────────────────
    try:
        seed_date = forecast_start - timedelta(days=1)
        recent_df = fetch_recent_weather(seed_date=seed_date, past_days=30)
    except Exception as e:
        st.error(f"Could not fetch weather data: `{e}`")
        st.stop()

    # ── Generate forecast ─────────────────────────────────────────────────────
    with st.spinner(f"Generating {horizon}-day forecast..."):
        forecast_df = forecast_n_days(
            recent_df, model, scaler_X, scaler_y, features, horizon=horizon
        )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("# 🌦 Kerala Weather Forecast")
    is_live = forecast_start >= datetime.now().date()
    mode    = "Live forecast" if is_live else \
              f"Forecast from {forecast_start.strftime('%d %b %Y')}"
    st.caption(
        f"Trivandrum (Thiruvananthapuram) · {mode} · "
        f"Training cutoff: 28 Feb 2026"
    )

    # ── Day-1 card ────────────────────────────────────────────────────────────
    st.markdown("---")
    row1      = forecast_df.iloc[0]
    day1_temp = row1["T2M"]
    day1_cond = row1["Condition"]
    day1_date = pd.Timestamp(row1["Date"])
    emoji     = CONDITION_EMOJI.get(day1_cond, "🌡")
    color     = CONDITION_COLOR.get(day1_cond, "#94A3B8")
    last      = recent_df.iloc[-1]

    day1_label = (
        f"Tomorrow — {day1_date.strftime('%A, %d %b %Y')}"
        if is_live else
        f"Day 1 forecast — {day1_date.strftime('%A, %d %b %Y')}"
    )
    obs_label = "Today" if is_live else \
                f"Last observed ({recent_df.index[-1].strftime('%d %b')})"

    c1, c2, c3, c4, c5 = st.columns([2.5, 1, 1, 1, 1])
    with c1:
        st.markdown(
            f"""
            <div style="background:{color}18;border:2px solid {color}55;
                        border-radius:16px;padding:20px 24px;">
                <div style="font-size:12px;color:#64748b;margin-bottom:6px;">
                    {day1_label}
                </div>
                <div style="display:flex;align-items:center;gap:16px;">
                    <span style="font-size:48px;">{emoji}</span>
                    <div>
                        <div style="font-size:38px;font-weight:700;
                                    color:#1e293b;line-height:1.1;">
                            {day1_temp:.1f}°C
                        </div>
                        <div style="font-size:14px;color:#475569;">
                            {day1_cond}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.metric(f"{obs_label} max", f"{last['T2M_MAX']:.1f}°C")
    with c3:
        st.metric(f"{obs_label} min", f"{last['T2M_MIN']:.1f}°C")
    with c4:
        st.metric("Humidity", f"{last['RH2M']:.0f}%")
    with c5:
        st.metric("Wind",     f"{last['WS2M']:.1f} m/s")

    # ── Forecast cards ────────────────────────────────────────────────────────
    st.markdown("---")
    fc_start_str = pd.Timestamp(forecast_df.iloc[0]["Date"]).strftime("%d %b")
    fc_end_str   = pd.Timestamp(forecast_df.iloc[-1]["Date"]).strftime("%d %b %Y")
    st.markdown(
        f"### {horizon}-Day Forecast &nbsp;"
        f"<span style='font-size:14px;color:#64748b;font-weight:400;'>"
        f"{fc_start_str} → {fc_end_str}</span>",
        unsafe_allow_html=True,
    )
    render_weather_cards(forecast_df)

    # ── Chart ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Temperature trend")
    chart_data = make_temp_chart(recent_df, forecast_df)
    st.line_chart(chart_data, height=420, use_container_width=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    if show_metrics:
        st.markdown("---")
        st.markdown("### Model performance")
        st.caption(
            "14-day ahead forecast · Targets: T2M, T2M_MAX, T2M_MIN · "
            "80/20 chronological split"
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²",   "0.8628",   help="Proportion of variance explained")
        m2.metric("MAE",  "0.7431°C", help="Mean Absolute Error across all 3 targets")
        m3.metric("RMSE", "0.9597°C", help="Root Mean Squared Error")
        m4.metric("MAPE", "2.704%",   help="Mean Absolute Percentage Error")
        st.caption(
            "Trained on NASA/POWER MERRA-2 · Jan 1981 – Feb 2026 · "
            "45 years · 16,495 rows"
        )

    # ── Raw table ─────────────────────────────────────────────────────────────
    if show_table:
        st.markdown("---")
        st.markdown("### Raw forecast data")
        st.dataframe(
            forecast_df.style.format({
                "T2M"    : "{:.2f}°C",
                "T2M_MAX": "{:.2f}°C",
                "T2M_MIN": "{:.2f}°C",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.download_button(
        label="⬇️ Download forecast as CSV",
        data=forecast_df.to_csv(index=False),
        file_name=f"trivandrum_forecast_{forecast_start.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    st.caption(
        "Built with LSTM · Training: NASA/POWER MERRA-2 (1981–2026) · "
        "Live seed: Open-Meteo API · "
        f"Last updated {datetime.now().strftime('%d %b %Y')}"
    )


if __name__ == "__main__":
    main()
