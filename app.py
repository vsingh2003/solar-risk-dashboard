"""
Solar Risk Monitor — Dashboard v4 (Client Deliverable Layout)
Dark-themed professional UI using Plotly Express
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Solar Risk Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global background */
    .stApp { background-color: #0d1117; color: #e6edf3; }
    /* Sidebar */
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #161b22 0%, #0d1117 100%); border-right: 1px solid #30363d; }
    section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stDateInput label { color: #8b949e !important; font-size: 0.78rem !important; }
    /* KPI cards */
    .kpi-card { background: linear-gradient(135deg, #161b22, #1c2128); border: 1px solid #30363d; border-radius: 12px; padding: 22px 18px; text-align: center; transition: border-color 0.2s; }
    .kpi-card:hover { border-color: #58a6ff; }
    .kpi-icon  { font-size: 1.6rem; margin-bottom: 6px; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #58a6ff; line-height: 1.1; }
    .kpi-label { font-size: 0.75rem; color: #8b949e; margin-top: 4px; letter-spacing: 0.05em; text-transform: uppercase; }
    .kpi-card.red   .kpi-value { color: #f85149; }
    .kpi-card.green .kpi-value { color: #3fb950; }
    .kpi-card.amber .kpi-value { color: #d29922; }
    /* Section headings & Dividers */
    h1 { color: #58a6ff !important; }
    h2, h3 { color: #e6edf3 !important; }
    hr { border-color: #30363d !important; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    [data-testid="stMetricDelta"] { color: #3fb950; }
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK  = "plotly_dark"
PLOTLY_BG    = "#0d1117"
PLOTLY_PAPER = "#161b22"

# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("final_anomaly_results.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_feat_imp():
    if os.path.exists("feature_importance.csv"):
        return pd.read_csv("feature_importance.csv")
    return None

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ **final_anomaly_results.csv not found.** Please run `python pipeline.py` first.")
    st.stop()

feat_imp_df = load_feat_imp()

# ══════════════════════════════════════════════════════════════
# SIDEBAR — FILTERS
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Solar Risk Monitor")
    st.caption("Anomaly Detection & Risk Monitoring")
    st.markdown("---")
    st.markdown("### 🔍 Filters")

    unit_options = ['All Units'] + sorted(df['unit_id'].unique().tolist())
    selected_unit = st.selectbox("Select Unit / Inverter", unit_options)

    date_min = df['date'].min().date()
    date_max = df['date'].max().date()
    date_range = st.date_input("Date Range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
    show_anomalies_only = st.checkbox("Show anomalies only", value=False)

    st.markdown("---")
    st.markdown("### 📌 Data Sources")
    sources = df['source'].value_counts()
    for src, cnt in sources.items():
        st.markdown(f"- **{src}** — {cnt} records")

    st.markdown("---")
    st.caption("Built with Python · Streamlit · Plotly")

# Apply filters
fdf = df.copy()
if selected_unit != 'All Units':
    fdf = fdf[fdf['unit_id'] == selected_unit]
if len(date_range) == 2:
    fdf = fdf[(fdf['date'] >= pd.Timestamp(date_range[0])) & (fdf['date'] <= pd.Timestamp(date_range[1]))]
if show_anomalies_only:
    fdf = fdf[fdf['anomaly_flag'] == 1]

# ══════════════════════════════════════════════════════════════
# GLOBAL HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("# ⚡ Solar Risk Monitor")
st.markdown(
    f"**{selected_unit}** &nbsp;|&nbsp; "
    f"{fdf['date'].min().strftime('%d %b %Y') if len(fdf) else '—'} → "
    f"{fdf['date'].max().strftime('%d %b %Y') if len(fdf) else '—'} &nbsp;|&nbsp; "
    f"{len(fdf):,} records in view"
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard & Monitoring",
    "🔍 Exploratory Data Analysis (EDA)",
    "🧠 Explainable AI (XAI) & Flags"
])

# ==============================================================
# TAB 1: DASHBOARD & MONITORING
# ==============================================================
with tab1:

    # ── Section 1: KPI Metrics Row ─────────────────────────────
    total_anomalies   = int(fdf['anomaly_flag'].sum())
    total_units       = int(fdf['unit_id'].nunique())
    avg_daily_yield   = fdf['daily_yield_kwh'].mean() if len(fdf) else 0

    inv_anom_full     = df.groupby('unit_id')['anomaly_flag'].sum()
    highest_risk_unit = inv_anom_full.idxmax()
    highest_risk_cnt  = int(inv_anom_full.max())

    c1, c2, c3, c4 = st.columns(4)

    def kpi(col, icon, value, label, card_class=""):
        col.markdown(
            f'<div class="kpi-card {card_class}">'
            f'  <div class="kpi-icon">{icon}</div>'
            f'  <div class="kpi-value">{value}</div>'
            f'  <div class="kpi-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    kpi(c1, "⚠️", total_anomalies, "Total Anomalies", "red" if total_anomalies > 50 else "amber")
    kpi(c2, "🔌", total_units, "Active Units Monitored", "")
    kpi(c3, "⚡", f"{avg_daily_yield:.1f} kWh", "Avg Daily Yield", "green")
    kpi(c4, "🚨", highest_risk_unit, f"Highest Risk Unit ({highest_risk_cnt} flags)", "red")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Section 2: Daily Yield Trend ──────────────────────────
    st.markdown("## 📈 Daily Yield Trend")
    if len(fdf) == 0:
        st.info("No records match the current filter selection.")
    else:
        if selected_unit == 'All Units':
            trend_df = (
                fdf.groupby(['date', 'unit_id'], as_index=False)
                   .agg(daily_yield_kwh=('daily_yield_kwh', 'mean'))
            )
            fig_trend = px.line(
                trend_df, x='date', y='daily_yield_kwh', color='unit_id',
                template=PLOTLY_DARK,
                labels={'daily_yield_kwh': 'Daily Yield (kWh)', 'date': 'Date', 'unit_id': 'Unit'}
            )
        else:
            anom_df   = fdf[fdf['anomaly_flag'] == 1]
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=fdf['date'], y=fdf['daily_yield_kwh'],
                mode='lines', name='Daily Yield',
                line=dict(color='#58a6ff', width=2)
            ))
            fig_trend.add_trace(go.Scatter(
                x=fdf['date'], y=fdf['rolling_mean_7'],
                mode='lines', name='7-Day Rolling Mean',
                line=dict(color='#d29922', width=1.5, dash='dash')
            ))
            fig_trend.add_trace(go.Scatter(
                x=anom_df['date'], y=anom_df['daily_yield_kwh'],
                mode='markers', name='⚠ Anomaly',
                marker=dict(color='#f85149', size=10, symbol='x', line=dict(width=2, color='#f85149'))
            ))

        fig_trend.update_layout(
            paper_bgcolor=PLOTLY_PAPER, plot_bgcolor=PLOTLY_BG,
            height=400, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    # ── Section 3: Unit Comparison Charts ─────────────────────
    st.markdown("## 🔌 Unit Performance Comparison")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Average Daily Yield by Unit")
        unit_avg = (
            df.groupby('unit_id', as_index=False)
              .agg(avg_yield=('daily_yield_kwh', 'mean'))
              .sort_values('avg_yield', ascending=True)
        )
        fig_bar = px.bar(
            unit_avg, x='avg_yield', y='unit_id', orientation='h',
            color='avg_yield', color_continuous_scale='Blues',
            template=PLOTLY_DARK,
            labels={'avg_yield': 'Avg Daily Yield (kWh)', 'unit_id': 'Unit'}
        )
        fig_bar.update_layout(
            paper_bgcolor=PLOTLY_PAPER, plot_bgcolor=PLOTLY_BG,
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("### Anomaly Count by Unit")
        unit_anom = (
            df.groupby('unit_id', as_index=False)
              .agg(anomalies=('anomaly_flag', 'sum'))
              .sort_values('anomalies', ascending=True)
        )
        fig_anom_bar = px.bar(
            unit_anom, x='anomalies', y='unit_id', orientation='h',
            color='anomalies', color_continuous_scale='Reds',
            template=PLOTLY_DARK,
            labels={'anomalies': 'Anomaly Count', 'unit_id': 'Unit'}
        )
        fig_anom_bar.update_layout(
            paper_bgcolor=PLOTLY_PAPER, plot_bgcolor=PLOTLY_BG,
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_anom_bar, use_container_width=True)

    st.markdown("---")

    # ── Section 7: Simple Q&A Chatbot ─────────────────────────
    st.markdown("## 💬 Quick Insights — Ask a Question")
    st.caption("Try: 'most anomalies' · 'best unit' · 'average yield' · 'total anomalies' · 'how many units' · 'date range' · 'worst unit'")

    query = st.text_input("Type your question:", placeholder="e.g. which unit has the most anomalies?")

    if query:
        q = query.lower().strip()
        inv_anom_all = df.groupby('unit_id')['anomaly_flag'].sum()
        inv_avg_all  = df.groupby('unit_id')['daily_yield_kwh'].mean()

        if any(w in q for w in ['most anomalies', 'worst unit', 'worst', 'highest risk']):
            u = inv_anom_all.idxmax(); c = int(inv_anom_all.max())
            st.error(f"🚨 **{u}** has the most anomalies — **{c} flagged days**.")
        elif any(w in q for w in ['best unit', 'best performing', 'highest yield', 'top unit']):
            u = inv_avg_all.idxmax(); v = inv_avg_all.max()
            st.success(f"🏆 **{u}** is the best performer — avg **{v:.2f} kWh/day**.")
        elif any(w in q for w in ['average yield', 'avg yield', 'mean yield', 'average power']):
            v = df['daily_yield_kwh'].mean()
            st.info(f"⚡ Overall average daily yield across all units: **{v:.2f} kWh**.")
        elif any(w in q for w in ['total anomalies', 'how many anomalies', 'count anomalies']):
            n = int(df['anomaly_flag'].sum())
            st.info(f"⚠️ **{n} anomalies** detected across all units and dates.")
        elif any(w in q for w in ['how many units', 'total units', 'number of units', 'units monitored']):
            n = df['unit_id'].nunique()
            st.info(f"🔌 **{n} units** are monitored in this dashboard.")
        elif any(w in q for w in ['date range', 'time period', 'when', 'dates covered']):
            d1 = df['date'].min().strftime('%d %b %Y')
            d2 = df['date'].max().strftime('%d %b %Y')
            st.info(f"📅 Data covers **{d1}** to **{d2}**.")
        elif any(w in q for w in ['fleet', 'fleet average', 'fleet avg']):
            v = df['fleet_avg_yield'].mean()
            st.info(f"🌐 Fleet-wide average daily yield: **{v:.2f} kWh**.")
        else:
            st.warning("I can answer: **most anomalies · worst/best unit · average yield · total anomalies · how many units · date range · fleet average**")

# ==============================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================
with tab2:
    st.markdown("## 🔍 Exploratory Data Analysis (EDA)")
    st.markdown(
        "This tab explores the relationships between the engineered features used for anomaly detection. "
        "Use the scatter plot below to examine how metrics like z-score, fleet deviation, and rolling averages "
        "interact — and how those interactions separate normal operating days from flagged anomalies."
    )
    st.markdown("---")

    # ── Section 4: Correlation Scatter Plot ───────────────────
    st.markdown("## 🔬 Correlation Analysis — Key Metrics")
    st.caption("Each dot is one unit-day. Red dots = flagged anomalies. Hover for full details.")

    scatter_df = fdf.copy()
    scatter_df['Status'] = scatter_df['anomaly_flag'].map({0: 'Normal', 1: 'Anomaly'})

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        x_axis = st.selectbox(
            "X-axis metric",
            ['rolling_mean_7', 'z_score', 'pct_dev_from_fleet', 'day_change_pct', 'performance_ratio', 'ratio_to_rollmean'],
            index=0
        )
    with col_s2:
        y_axis = st.selectbox(
            "Y-axis metric",
            ['daily_yield_kwh', 'z_score', 'pct_dev_from_fleet', 'day_change_pct', 'performance_ratio', 'rolling_std_7'],
            index=0
        )

    if len(scatter_df) > 1:
        fig_scatter = px.scatter(
            scatter_df, x=x_axis, y=y_axis,
            color='Status',
            color_discrete_map={'Normal': '#3fb950', 'Anomaly': '#f85149'},
            hover_data=['date', 'unit_id', 'daily_yield_kwh', 'explanation'],
            symbol='unit_id',
            template=PLOTLY_DARK,
            labels={'Status': 'Classification'}
        )
        fig_scatter.update_traces(marker=dict(size=7, opacity=0.8))
        fig_scatter.update_layout(
            paper_bgcolor=PLOTLY_PAPER, plot_bgcolor=PLOTLY_BG,
            height=430, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Not enough data points in the current filter to render the scatter plot.")

# ==============================================================
# TAB 3: EXPLAINABLE AI (XAI) & FLAGS
# ==============================================================
with tab3:
    st.markdown("## 🧠 Explainable AI (XAI) & Flags")
    st.markdown(
        "This tab presents the Isolation Forest model's **feature importance** — showing which engineered signals "
        "drove anomaly detection — alongside the plain-English **XAI explanation** generated for every flagged record. "
        "Use this to communicate model reasoning clearly to stakeholders and validate that flags are meaningful."
    )
    st.markdown("---")

    # ── Section 5: XAI Feature Importance ─────────────────────
    st.markdown("## 🧬 Model Overview & Feature Importance")
    st.info(
        "**Model Overview:** This pipeline uses a hybrid anomaly detection system combining **Rule-Based Heuristics** "
        "(to catch severe, obvious drops) and an **Isolation Forest** (an unsupervised ML algorithm that detects "
        "complex, multi-dimensional anomalies in sensor data)."
    )

    col_fi, col_guide = st.columns([1, 1])

    with col_fi:
        if feat_imp_df is not None:
            fig_fi = px.bar(
                feat_imp_df.sort_values('importance', ascending=True),
                x='importance', y='feature', orientation='h',
                color='importance', color_continuous_scale='Purples',
                template=PLOTLY_DARK,
                title="Which signals matter most?",
                labels={'importance': 'Importance Score', 'feature': 'Feature'}
            )
            fig_fi.update_layout(
                paper_bgcolor=PLOTLY_PAPER, plot_bgcolor=PLOTLY_BG,
                height=340, margin=dict(l=10, r=10, t=50, b=10),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Run `pipeline.py` to generate `feature_importance.csv` and populate this chart.")

    with col_guide:
        st.markdown("### Signal Reference Guide")
        st.markdown("""
        | Signal | Threshold | Meaning |
        |--------|-----------|---------|
        | **z_score** | < −1.8 | Output far below this unit's own history |
        | **pct_dev_from_fleet** | < −20% | Performing 20%+ worse than peers |
        | **day_change_pct** | < −30% | Sharp overnight drop |
        | **ratio_to_rollmean** | < 0.65 | Below 65% of recent 7-day average |
        | **ML flag** | — | Isolation Forest detected unusual pattern |
        """)

    st.markdown("---")

    # ── Section 6: Flagged Records Table (with Explanation) ───
    st.markdown("## 🚨 Flagged Anomaly Records — XAI Explanations")
    anomaly_view = fdf[fdf['anomaly_flag'] == 1].copy()

    if anomaly_view.empty:
        st.success("✅ No anomalies in the current selection.")
    else:
        st.warning(f"⚠️  **{len(anomaly_view)} anomalous records** in current view.")
        display_df = anomaly_view[[
            'date', 'unit_id', 'source',
            'daily_yield_kwh', 'pct_dev_from_fleet',
            'z_score', 'day_change_pct', 'explanation'
        ]].copy()
        display_df.columns = [
            'Date', 'Unit', 'Source',
            'Yield (kWh)', 'Fleet Dev (%)',
            'Z-Score', 'Day Chg (%)', 'Explanation'
        ]
        display_df['Yield (kWh)']   = display_df['Yield (kWh)'].round(2)
        display_df['Fleet Dev (%)'] = display_df['Fleet Dev (%)'].round(1)
        display_df['Z-Score']       = display_df['Z-Score'].round(2)
        display_df['Day Chg (%)']   = display_df['Day Chg (%)'].round(1)
        display_df = display_df.sort_values('Date', ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# GLOBAL FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#30363d; font-size:0.78rem; padding:8px'>"
    "⚡ Solar Risk Monitor &nbsp;·&nbsp; Built with Python, Streamlit &amp; Plotly"
    "</div>",
    unsafe_allow_html=True
)