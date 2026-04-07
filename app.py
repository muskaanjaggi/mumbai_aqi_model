# ============================================================
# Mumbai AQI — Streamlit Web App
# File: app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import itertools
import warnings

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import linalg

warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Mumbai AQI — Stochastic Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #D3D3D3;
        text-align: center;
        padding: 1rem 0 0.2rem 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #D3D3D3;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #4e79a7;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.3rem 0;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #D3D3D3;
        border-bottom: 2px solid #4e79a7;
        padding-bottom: 0.3rem;
        margin: 1.5rem 0 1rem 0;
    }
    .insight-box {
        background: #1a1a2e;
        border-left: 4px solid #2196F3;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .warning-box {
        background: #1a1a2e;
        border-left: 4px solid #FFC107;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def aqi_category(val):
    if val <= 50:    return 'Good'
    elif val <= 100: return 'Satisfactory'
    elif val <= 200: return 'Moderate'
    else:            return 'Poor'

def aqi_color(val):
    if val <= 50:    return '#2e7d32'
    elif val <= 100: return '#f9a825'
    elif val <= 200: return '#e65100'
    else:            return '#b71c1c'

# ============================================================
# LOAD & CACHE DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv('mumbai_aqi_fixed.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    aqi = df['AQI'].dropna()
    return aqi

@st.cache_data
def run_stl(aqi_values, dates_list):
    s = pd.Series(aqi_values, index=pd.DatetimeIndex(dates_list))
    stl = STL(s, period=365, robust=True)
    res = stl.fit()
    return res.trend.values, res.seasonal.values, res.resid.values

@st.cache_data
def run_arima(aqi_values, dates_list, d_val):
    s = pd.Series(aqi_values, index=pd.DatetimeIndex(dates_list))
    train = s[:-30]
    test  = s[-30:]
    results = []
    for p, q in itertools.product(range(0, 4), range(0, 4)):
        try:
            m = ARIMA(train, order=(p, d_val, q)).fit()
            rmse = float(np.sqrt(np.mean(m.resid**2)))
            results.append({'order': (p, d_val, q),
                            'AIC': float(m.aic),
                            'BIC': float(m.bic),
                            'RMSE': rmse})
        except:
            pass
    best = min(results, key=lambda x: x['AIC'])
    return best['order'], pd.DataFrame(results)

@st.cache_data
def get_forecast(aqi_values, dates_list, order, steps):
    s = pd.Series(aqi_values, index=pd.DatetimeIndex(dates_list))
    fit = ARIMA(s, order=order).fit()
    fc  = fit.get_forecast(steps=steps)
    return (fc.predicted_mean,
            fc.conf_int(alpha=0.05),
            fit.resid)

@st.cache_data
def run_markov(aqi_values, dates_list):
    s = pd.Series(aqi_values, index=pd.DatetimeIndex(dates_list))
    STATES = ['Good', 'Satisfactory', 'Moderate', 'Poor']
    state_idx = {s_: i for i, s_ in enumerate(STATES)}
    cats = s.apply(aqi_category).map(state_idx)
    N = 4
    counts = np.zeros((N, N), dtype=int)
    for i in range(len(cats) - 1):
        s0, s1 = cats.iloc[i], cats.iloc[i+1]
        if pd.notna(s0) and pd.notna(s1):
            counts[int(s0), int(s1)] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = counts / row_sums
    eigenvalues, eigenvectors = linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi_raw = eigenvectors[:, idx].real
    pi = pi_raw / pi_raw.sum()
    return P, pi, STATES, counts

# ============================================================
# MAIN APP
# ============================================================

# Title
st.markdown('<div class="main-title">Mumbai AQI Fluctuation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Autoregressive Stochastic Model — ARIMA + Markov Chain Analysis</div>', unsafe_allow_html=True)

# Load data
aqi = load_data()
aqi_vals  = aqi.values
aqi_dates = aqi.index
aqi_dates_list = aqi_dates.tolist()  # hashable for st.cache_data

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")

    forecast_days = st.slider("Forecast horizon (days)", 7, 30, 7)

    st.markdown("### 📅 Date Filter (EDA)")
    year_range = st.select_slider(
        "Year range",
        options=list(range(2018, 2025)),
        value=(2018, 2024)
    )

    st.markdown("### 🔬 ARIMA Order Override")
    auto_order = st.checkbox("Auto-select best order (AIC)", value=True)
    if not auto_order:
        p_manual = st.number_input("p (AR order)", 0, 5, 1)
        d_manual = st.number_input("d (differencing)", 0, 2, 1)
        q_manual = st.number_input("q (MA order)", 0, 5, 1)

    st.markdown("---")
    st.markdown("**Dataset:** Mumbai AQI 2018–2024")
    st.markdown("**Model:** ARIMA + Markov Chain")
    st.markdown("**Source:** CPCB via GitHub + Kaggle")

# ── Top KPI row ───────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Observations", f"{len(aqi):,}")
with col2:
    st.metric("Date Range", f"{aqi.index[0].year}–{aqi.index[-1].year}")
with col3:
    st.metric("Mean AQI", f"{aqi.mean():.1f}")
with col4:
    st.metric("Max AQI", f"{aqi.max():.0f}",
              delta=f"{aqi_category(aqi.max())}", delta_color="inverse")
with col5:
    worst_pct = (aqi > 200).sum() / len(aqi) * 100
    st.metric("Days AQI > 200", f"{worst_pct:.1f}%")

# ============================================================
# TAB LAYOUT
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 EDA & Decomposition",
    "🔬 Stationarity",
    "📉 ACF & PACF",
    "🤖 ARIMA Forecast",
    "🔗 Markov Chain"
])

# ────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Time Series & Decomposition</div>',
                unsafe_allow_html=True)

    # Filter by year
    aqi_filtered = aqi[(aqi.index.year >= year_range[0]) &
                       (aqi.index.year <= year_range[1])]

    # Main time series plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=aqi_filtered.index, y=aqi_filtered.values,
        mode='lines', name='Daily AQI',
        line=dict(color='#4e79a7', width=1),
        hovertemplate='%{x|%d %b %Y}<br>AQI: %{y:.0f}<extra></extra>'
    ))
    # Rolling mean
    rm = aqi_filtered.rolling(30).mean()
    fig.add_trace(go.Scatter(
        x=rm.index, y=rm.values,
        mode='lines', name='30-day Rolling Mean',
        line=dict(color='#e15759', width=2.5)
    ))
    # Category bands
    for level, label, color in [(50,'Good','rgba(46,125,50,0.08)'),
                                  (100,'Satisfactory','rgba(249,168,37,0.08)'),
                                  (200,'Moderate','rgba(230,81,0,0.08)')]:
        fig.add_hline(y=level, line_dash='dash', line_color=color.replace('0.08','0.5'),
                      line_width=1,
                      annotation_text=label, annotation_position='right')
    fig.update_layout(title='Mumbai Daily AQI with 30-Day Rolling Mean',
                      xaxis_title='Date', yaxis_title='AQI',
                      hovermode='x unified', height=400,
                      legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    # STL Decomposition
    st.markdown("#### STL Decomposition (period = 365 days)")
    with st.spinner("Running STL decomposition..."):
        trend, seasonal, resid = run_stl(aqi_vals, aqi_dates_list)

    fig2 = make_subplots(rows=4, cols=1, shared_xaxes=True,
                         subplot_titles=['Observed','Trend','Seasonal','Residual'],
                         vertical_spacing=0.06)
    components = [
        (aqi.values,  '#4e79a7', 'Observed'),
        (trend,       '#e15759', 'Trend'),
        (seasonal,    '#59a14f', 'Seasonal'),
        (resid,       '#f28e2b', 'Residual')
    ]
    for i, (data, color, name) in enumerate(components, 1):
        fig2.add_trace(go.Scatter(x=aqi_dates, y=data, mode='lines',
                                  line=dict(color=color, width=1),
                                  name=name, showlegend=False), row=i, col=1)
    fig2.update_layout(height=600, title='STL Decomposition of Mumbai AQI')
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Insight:</b> The seasonal component shows a clear annual cycle — AQI peaks in November–January (winter inversion traps pollutants) and drops in June–August (monsoon washes the air). The trend component captured the 2020 COVID lockdown dip.</div>',
                unsafe_allow_html=True)

    # Monthly boxplot
    st.markdown("#### Monthly AQI Distribution")
    df_month = pd.DataFrame({'AQI': aqi.values, 'Month': aqi.index.month,
                              'MonthName': aqi.index.strftime('%b')},
                             index=aqi.index)
    month_order = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    fig3 = px.box(df_month, x='MonthName', y='AQI',
                  category_orders={'MonthName': month_order},
                  color='MonthName',
                  title='Monthly AQI Distribution — Seasonal Pattern',
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig3.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig3, use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 2 — STATIONARITY
# ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Stationarity Testing — ADF & KPSS</div>',
                unsafe_allow_html=True)

    st.markdown("""
    **What is stationarity?** A time series is stationary if its mean, variance, and 
    autocorrelation structure don't change over time. ARIMA models require stationarity.
    We use two complementary tests:
    - **ADF Test** — H₀: series is *non-stationary*. Reject if p < 0.05 → stationary ✓
    - **KPSS Test** — H₀: series is *stationary*. Reject if p < 0.05 → non-stationary ✗
    """)

    col1, col2 = st.columns(2)

    for series, label, col in [(aqi, "Original AQI", col1),
                                (aqi.diff().dropna(), "1st Differenced AQI", col2)]:
        with col:
            st.markdown(f"##### {label}")
            adf_r  = adfuller(series, autolag='AIC')
            kpss_r = kpss(series, regression='c', nlags='auto')

            adf_stat = "✅ Stationary" if adf_r[1] < 0.05 else "❌ Non-Stationary"
            kpss_stat = "❌ Non-Stationary" if kpss_r[1] < 0.05 else "✅ Stationary"

            st.markdown(f"""
            | Test | Statistic | p-value | Verdict |
            |------|-----------|---------|---------|
            | ADF  | {adf_r[0]:.4f} | {adf_r[1]:.4f} | {adf_stat} |
            | KPSS | {kpss_r[0]:.4f} | {kpss_r[1]:.4f} | {kpss_stat} |
            """)

    # Determine d
    adf_p_orig = adfuller(aqi, autolag='AIC')[1]
    d_val = 0 if adf_p_orig < 0.05 else 1
    st.markdown(f'<div class="insight-box">💡 <b>Conclusion:</b> d = <b>{d_val}</b> — {"No differencing needed (already stationary)" if d_val == 0 else "1st order differencing applied to achieve stationarity"}</div>',
                unsafe_allow_html=True)

    # Rolling stats plot
    st.markdown("#### Rolling Mean & Variance (Visual Stationarity Check)")
    fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=['Rolling Mean (window=30)',
                                         'Rolling Variance (window=30)'])
    fig4.add_trace(go.Scatter(x=aqi.index, y=aqi.rolling(30).mean(),
                              line=dict(color='#e15759'), name='Rolling Mean'), row=1, col=1)
    fig4.add_trace(go.Scatter(x=aqi.index, y=aqi.rolling(30).std()**2,
                              line=dict(color='#f28e2b'), name='Rolling Variance'), row=2, col=1)
    fig4.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 3 — ACF & PACF
# ────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">ACF & PACF Analysis</div>',
                unsafe_allow_html=True)

    st.markdown("""
    - **ACF** (Autocorrelation Function): How correlated is today's AQI with AQI from *k* days ago?
    - **PACF** (Partial ACF): After removing intermediate effects, how much does lag *k* directly influence today?
    - **Reading the plots:** Bars outside the shaded band are statistically significant (95% CI)
    """)

    max_lags = st.slider("Number of lags to display", 10, 60, 40)

    aqi_d = aqi.diff().dropna() if d_val == 1 else aqi
    ci_val = 1.96 / np.sqrt(len(aqi_d))

    acf_orig  = acf(aqi.values,    nlags=max_lags)
    pacf_orig = pacf(aqi.values,   nlags=max_lags, method='ywm')
    acf_diff  = acf(aqi_d.values,  nlags=max_lags)
    pacf_diff = pacf(aqi_d.values, nlags=max_lags, method='ywm')

    lags = list(range(max_lags + 1))

    fig5 = make_subplots(rows=2, cols=2,
                         subplot_titles=['ACF — Original AQI',
                                         'PACF — Original AQI',
                                         'ACF — Differenced AQI',
                                         'PACF — Differenced AQI'])

    for row, acf_v, pacf_v in [(1, acf_orig, pacf_orig),
                                 (2, acf_diff, pacf_diff)]:
        for col, vals in [(1, acf_v), (2, pacf_v)]:
            colors = ['#e15759' if abs(v) > ci_val and i > 0 else '#4e79a7'
                      for i, v in enumerate(vals)]
            fig5.add_trace(go.Bar(x=lags, y=vals,
                                  marker_color=colors,
                                  showlegend=False,
                                  hovertemplate='Lag %{x}: %{y:.4f}<extra></extra>'),
                           row=row, col=col)
            # CI bands
            for sign in [1, -1]:
                fig5.add_hline(y=sign * ci_val, line_dash='dash',
                               line_color='gray', line_width=1, row=row, col=col)

    fig5.update_layout(height=550,
                       title='ACF & PACF — Red bars are statistically significant')
    st.plotly_chart(fig5, use_container_width=True)

    # Significant lags summary
    sig_pacf = [i for i, v in enumerate(pacf_diff[1:], 1) if abs(v) > ci_val]
    sig_acf  = [i for i, v in enumerate(acf_diff[1:],  1) if abs(v) > ci_val]
    st.markdown(f'<div class="insight-box">💡 <b>Significant PACF lags (→ AR order p):</b> {sig_pacf[:8]}<br><b>Significant ACF lags (→ MA order q):</b> {sig_acf[:8]}</div>',
                unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# TAB 4 — ARIMA FORECAST
# ────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">ARIMA Model & Forecast</div>',
                unsafe_allow_html=True)

    with st.spinner("Running ARIMA grid search (p,d,q) ∈ {0..3}³ ..."):
        best_order, results_df = run_arima(aqi_vals, aqi_dates_list, d_val)

    if not auto_order:
        best_order = (p_manual, d_manual, q_manual)

    st.success(f"✅ Best model: **ARIMA{best_order}** (selected by AIC)")

    # Model comparison table
    with st.expander("📋 Full model comparison (AIC / BIC / RMSE)"):
        display_df = results_df.copy()
        display_df['order'] = display_df['order'].astype(str)
        display_df = display_df.sort_values('AIC').round(2)
        st.dataframe(display_df, use_container_width=True)

    # Get forecast
    with st.spinner("Generating forecast..."):
        fc_mean, fc_ci, residuals = get_forecast(
            aqi_vals, aqi_dates_list, best_order, forecast_days)

    fc_dates = pd.date_range(aqi.index[-1] + pd.Timedelta(days=1),
                              periods=forecast_days)
    fc_mean.index = fc_ci.index = fc_dates

    # Forecast plot
    fig6 = go.Figure()
    hist_window = aqi[-120:]
    fig6.add_trace(go.Scatter(
        x=hist_window.index, y=hist_window.values,
        mode='lines', name='Historical AQI',
        line=dict(color='#4e79a7', width=1.5)
    ))
    fig6.add_trace(go.Scatter(
        x=fc_dates, y=fc_mean.values,
        mode='lines+markers', name='Forecast',
        line=dict(color='#e15759', width=2.5),
        marker=dict(size=7)
    ))
    fig6.add_trace(go.Scatter(
        x=list(fc_dates) + list(fc_dates[::-1]),
        y=list(fc_ci.iloc[:,1]) + list(fc_ci.iloc[:,0][::-1]),
        fill='toself', fillcolor='rgba(225,87,89,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'
    ))
    fig6.add_vline(x=str(aqi.index[-1]), line_dash='dash',
                   line_color='gray', line_width=1)
    fig6.update_layout(
        title=f'{forecast_days}-Day AQI Forecast — ARIMA{best_order}',
        xaxis_title='Date', yaxis_title='AQI',
        hovermode='x unified', height=430,
        legend=dict(orientation='h', y=1.1)
    )
    st.plotly_chart(fig6, use_container_width=True)

    # Forecast table
    st.markdown("#### Forecast Values")
    fc_table = pd.DataFrame({
        'Date':     [d.strftime('%d %b %Y') for d in fc_dates],
        'Forecast': fc_mean.values.round(1),
        'Lower 95%': fc_ci.iloc[:,0].values.round(1),
        'Upper 95%': fc_ci.iloc[:,1].values.round(1),
        'Category': [aqi_category(v) for v in fc_mean.values]
    })
    st.dataframe(fc_table, use_container_width=True, hide_index=True)

    # Residual diagnostics
    st.markdown("#### Residual Diagnostics")
    col1, col2 = st.columns(2)
    with col1:
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(y=residuals.values, mode='lines',
                                   line=dict(color='#4e79a7', width=0.8),
                                   name='Residuals'))
        fig7.add_hline(y=0, line_dash='dash', line_color='black')
        fig7.update_layout(title='Residuals over Time',
                            yaxis_title='Residual', height=300)
        st.plotly_chart(fig7, use_container_width=True)
    with col2:
        fig8 = go.Figure()
        fig8.add_trace(go.Histogram(x=residuals.values, nbinsx=40,
                                     marker_color='#4e79a7', opacity=0.7,
                                     name='Residuals'))
        fig8.update_layout(title='Residual Distribution (should be ~normal)',
                            height=300)
        st.plotly_chart(fig8, use_container_width=True)

    lb = acorr_ljungbox(residuals.dropna(), lags=[10, 20], return_df=True)
    lb_p10 = float(lb['lb_pvalue'].iloc[0])
    lb_p20 = float(lb['lb_pvalue'].iloc[1])
    verdict = "✅ Residuals are white noise (good model fit)" \
              if lb_p10 > 0.05 else "⚠️ Residuals may have remaining structure"
    st.markdown(f'<div class="insight-box">💡 <b>Ljung-Box Test:</b> p(lag=10) = {lb_p10:.4f} | p(lag=20) = {lb_p20:.4f}<br>{verdict}</div>',
                unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# TAB 5 — MARKOV CHAIN
# ────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Markov Chain — AQI State Transitions</div>',
                unsafe_allow_html=True)

    st.markdown("""
    We model AQI as a **discrete-time Markov Chain** with 4 states.
    The key insight: tomorrow's air quality depends only on today's — not the entire history.
    The **stationary distribution π** tells us the long-run fraction of days in each state.
    """)

    P, pi, STATES, counts = run_markov(aqi_vals, aqi_dates_list)
    state_colors = ['#2e7d32','#f9a825','#e65100','#b71c1c']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Transition Matrix P")
        st.markdown("*Row = today's state, Column = tomorrow's state*")
        fig9 = go.Figure(go.Heatmap(
            z=P, x=STATES, y=STATES,
            colorscale='YlOrRd',
            zmin=0, zmax=1,
            text=[[f'{P[i,j]:.3f}' for j in range(4)] for i in range(4)],
            texttemplate='%{text}',
            textfont=dict(size=13),
            hoverongaps=False
        ))
        fig9.update_layout(
            title='P[i,j] = P(tomorrow=j | today=i)',
            xaxis_title="Tomorrow's State",
            yaxis_title="Today's State",
            height=370
        )
        st.plotly_chart(fig9, use_container_width=True)

        # Transition count table
        with st.expander("📋 Transition counts (raw)"):
            count_df = pd.DataFrame(counts, index=STATES, columns=STATES)
            st.dataframe(count_df)

    with col2:
        st.markdown("#### Stationary Distribution π")
        st.markdown("*Long-run fraction of days in each AQI state*")
        fig10 = go.Figure(go.Bar(
            x=STATES, y=(pi * 100).round(2),
            marker_color=state_colors,
            text=[f'{v*100:.1f}%' for v in pi],
            textposition='outside',
            hovertemplate='%{x}: %{y:.2f}%<extra></extra>'
        ))
        fig10.update_layout(
            title='Long-run AQI State Distribution',
            yaxis_title='Probability (%)',
            height=370,
            yaxis_range=[0, max(pi)*120]
        )
        st.plotly_chart(fig10, use_container_width=True)

    # Eigenvalue analysis
    st.markdown("#### Eigenvalue Analysis")
    from scipy.linalg import eig as sp_eig
    eigenvalues_raw, _ = sp_eig(P.T)
    eigs = np.sort(np.abs(eigenvalues_raw))[::-1]
    spectral_gap = float(1 - eigs[1])

    col1, col2, col3 = st.columns(3)
    col1.metric("λ₁ (dominant)", f"{eigs[0]:.6f}", "= 1 always ✓")
    col2.metric("λ₂", f"{eigs[1]:.6f}")
    col3.metric("Spectral Gap (1−|λ₂|)", f"{spectral_gap:.4f}")

    st.markdown(f'<div class="insight-box">💡 <b>Interpretation:</b><br>'
                f'In the long run, Mumbai air will be <b>Good</b> {pi[0]*100:.1f}% of days, '
                f'<b>Satisfactory</b> {pi[1]*100:.1f}%, '
                f'<b>Moderate</b> {pi[2]*100:.1f}%, '
                f'<b>Poor</b> {pi[3]*100:.1f}%.<br>'
                f'The spectral gap of {spectral_gap:.4f} means the chain converges to this '
                f'distribution in roughly <b>{int(1/spectral_gap)}</b> steps (days).'
                f'</div>', unsafe_allow_html=True)

    # Verification
    pi_check = pi @ P
    max_dev = float(np.max(np.abs(pi_check - pi)))
    st.markdown(f'<div class="warning-box">🔍 <b>Verification:</b> max|πP − π| = {max_dev:.2e} '
                f'{"✅ (confirms stationary distribution)" if max_dev < 1e-8 else "⚠️ check calculation"}'
                f'</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#888; font-size:0.85rem;'>"
    "Mumbai AQI Stochastic Model · ARIMA + Markov Chain · "
    "Data: CPCB via GitHub & Kaggle · Built with Streamlit"
    "</center>",
    unsafe_allow_html=True
)
