"""
A/B Testing Lab — Interactive Marketing Science Dashboard
Built with Streamlit + Plotly | Synthwave Dark Theme
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.statistics import (
    calculate_sample_size,
    check_srm,
    run_z_test,
    bayesian_ab_test,
    revenue_impact,
)
from src.data_generator import generate_ab_data

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Testing Lab | Marketing Science",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────
PURPLE  = "#8B5CF6"
CYAN    = "#06B6D4"
PINK    = "#EC4899"
GREEN   = "#10B981"
YELLOW  = "#F59E0B"
RED     = "#EF4444"
BG      = "#07080F"
SURFACE = "#0F1020"
CARD    = "#171829"
BORDER  = "#1E2048"
TEXT    = "#E8EAFF"
MUTED   = "#7B82B0"

PLOTLY = dict(
    template="plotly_dark",
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(color=TEXT, family="Segoe UI"),
    margin=dict(l=40, r=40, t=50, b=40),
)
COLOR_SEQ = [PURPLE, CYAN, PINK, GREEN, YELLOW]

st.markdown(f"""
<style>
  .stApp {{ background:{BG}; }}
  .main .block-container {{ padding:1.5rem 2rem; max-width:1400px; }}
  section[data-testid="stSidebar"] {{
      background:{SURFACE}; border-right:1px solid {BORDER};
  }}
  h1 {{ color:{PURPLE} !important; letter-spacing:-0.5px; }}
  h2 {{ color:{TEXT} !important; }}
  h3 {{ color:{TEXT} !important; }}
  [data-testid="metric-container"] {{
      background:{CARD}; border:1px solid {BORDER};
      border-radius:14px; padding:14px 18px;
  }}
  [data-testid="metric-container"] label {{ color:{MUTED} !important; font-size:12px !important; text-transform:uppercase; letter-spacing:1px; }}
  [data-testid="metric-container"] div[data-testid="metric-value"] {{ color:{TEXT} !important; font-size:26px !important; }}
  [data-testid="stMetricDelta"] {{ font-size:13px !important; }}
  .stTabs [data-baseweb="tab-list"] {{ background:{CARD}; border-radius:12px; gap:4px; padding:4px; }}
  .stTabs [data-baseweb="tab"] {{ border-radius:8px; color:{MUTED}; padding:8px 18px; }}
  .stTabs [aria-selected="true"] {{ background:{BORDER} !important; color:{PURPLE} !important; font-weight:600; }}
  .stAlert {{ border-radius:12px; }}
  .verdict-win {{
      background:linear-gradient(135deg,#064E3B,#065F46);
      border:1px solid {GREEN}; border-radius:14px;
      padding:24px; color:{GREEN};
  }}
  .verdict-lose {{
      background:linear-gradient(135deg,#450A0A,#7F1D1D);
      border:1px solid {RED}; border-radius:14px;
      padding:24px; color:{RED};
  }}
  .verdict-neutral {{
      background:linear-gradient(135deg,#1C1917,#292524);
      border:1px solid #78716C; border-radius:14px;
      padding:24px; color:#A8A29E;
  }}
  .kpi-card {{
      background:{CARD}; border:1px solid {BORDER};
      border-radius:14px; padding:20px; text-align:center;
  }}
  .kpi-card .label {{ color:{MUTED}; font-size:11px; text-transform:uppercase; letter-spacing:1px; }}
  .kpi-card .value {{ color:{TEXT}; font-size:28px; font-weight:700; margin:8px 0 4px; }}
  .kpi-card .sub   {{ color:{MUTED}; font-size:12px; }}
  .pill {{
      display:inline-block; background:{BORDER}; border-radius:20px;
      padding:3px 12px; font-size:12px; color:{PURPLE}; margin:2px;
  }}
  div[data-testid="column"] > div {{ gap:0.5rem; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🧪 A/B Testing Lab")
    st.markdown(f"<span style='color:{MUTED};font-size:12px'>Marketing Science Dashboard</span>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Dataset", "🎯 Experiment Design", "🔍 SRM Validation",
         "📈 Statistical Analysis", "📋 Business Report"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"<span style='color:{MUTED};font-size:11px'>⚙️ Test Parameters</span>", unsafe_allow_html=True)
    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01,
                      help="Probability of false positive (Type I error)")
    power_target = st.slider("Target power (1-β)", 0.70, 0.95, 0.80, 0.05,
                              help="Probability of detecting a real effect")

    st.markdown("---")
    st.markdown(f"<p style='color:{MUTED};font-size:10px;text-align:center'>Built with Streamlit · Scipy · Plotly</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helper: load dataframe
# ─────────────────────────────────────────────────────────────
def get_df():
    if st.session_state.df is not None:
        return st.session_state.df
    return None

def kpi(col, label, value, sub="", color=TEXT):
    col.markdown(
        f'<div class="kpi-card"><div class="label">{label}</div>'
        f'<div class="value" style="color:{color}">{value}</div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# PAGE 1: Dataset
# ─────────────────────────────────────────────────────────────
if page == "📊 Dataset":
    st.title("📊 Dataset Overview")
    st.markdown(f"<span style='color:{MUTED}'>Load your data or use the built-in synthetic e-commerce dataset</span>", unsafe_allow_html=True)
    st.markdown("---")

    col_load, col_info = st.columns([1, 2])

    with col_load:
        source = st.selectbox("Data source", ["🎲 Generate synthetic data", "📁 Upload CSV"])

        if source == "🎲 Generate synthetic data":
            st.markdown(f"<span style='color:{MUTED};font-size:12px'>E-commerce landing page test — 14 days, LATAM market</span>", unsafe_allow_html=True)
            n_users = st.number_input("Total users", 5_000, 200_000, 45_000, 5_000)
            ctrl_rate = st.slider("Control rate", 0.05, 0.30, 0.104, 0.001, format="%.3f")
            treat_rate = st.slider("Treatment rate", 0.05, 0.30, 0.127, 0.001, format="%.3f")
            if st.button("🚀 Generate Data", use_container_width=True):
                with st.spinner("Generating..."):
                    df = generate_ab_data(
                        n_users=n_users,
                        control_rate=ctrl_rate,
                        treatment_rate=treat_rate,
                    )
                    st.session_state.df = df
                st.success(f"✅ Generated {len(df):,} records")
                st.rerun()
        else:
            uploaded = st.file_uploader(
                "Upload CSV",
                type="csv",
                help="Required columns: group (control/treatment), converted (0/1)",
            )
            if uploaded:
                df = pd.read_csv(uploaded)
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df):,} rows")
                st.rerun()

    df = get_df()
    if df is None:
        with col_info:
            st.info("👈 Generate or upload data to get started")
            st.markdown(f"""
            **Expected CSV columns:**
            | Column | Type | Description |
            |---|---|---|
            | `group` | string | `control` or `treatment` |
            | `converted` | int | 0 or 1 |
            | `revenue` | float | Revenue per user (optional) |
            | `device` | string | mobile / desktop / tablet (optional) |
            | `timestamp` | datetime | Event timestamp (optional) |
            """)
    else:
        ctrl = df[df["group"] == "control"]
        treat = df[df["group"] == "treatment"]
        cr_c = ctrl["converted"].mean()
        cr_t = treat["converted"].mean()

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        kpi(c1, "Total Users", f"{len(df):,}", "experiment size")
        kpi(c2, "Control", f"{len(ctrl):,}", f"conv: {cr_c:.2%}", CYAN)
        kpi(c3, "Treatment", f"{len(treat):,}", f"conv: {cr_t:.2%}", PURPLE)
        kpi(c4, "Observed Lift", f"{(cr_t - cr_c) / cr_c:+.1%}", "relative uplift",
            GREEN if cr_t > cr_c else RED)
        kpi(c5, "Total Revenue", f"${df['revenue'].sum():,.0f}" if "revenue" in df.columns else "N/A",
            "combined groups" if "revenue" in df.columns else "no revenue col")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📊 Distributions", "📅 Timeline"])

        with tab1:
            st.dataframe(
                df.head(500).style.applymap(
                    lambda v: f"color:{PURPLE}" if v == "treatment" else
                              (f"color:{CYAN}" if v == "control" else ""),
                    subset=["group"] if "group" in df.columns else [],
                ),
                use_container_width=True, height=320,
            )
            st.caption(f"Showing first 500 of {len(df):,} rows")

        with tab2:
            c1, c2 = st.columns(2)

            # Conversion rate bar
            fig = go.Figure()
            for g, color, n_users_g, cr in [
                ("Control", CYAN, len(ctrl), cr_c),
                ("Treatment", PURPLE, len(treat), cr_t),
            ]:
                fig.add_trace(go.Bar(
                    name=g, x=[g],
                    y=[cr * 100],
                    marker_color=color,
                    text=f"{cr:.2%}",
                    textposition="outside",
                    width=0.4,
                ))
            fig.update_layout(
                title="Conversion Rate by Group",
                yaxis_title="Conversion Rate (%)",
                showlegend=False,
                **PLOTLY,
            )
            c1.plotly_chart(fig, use_container_width=True)

            # Device breakdown
            if "device" in df.columns:
                device_conv = df.groupby(["device", "group"])["converted"].mean().reset_index()
                fig2 = px.bar(
                    device_conv, x="device", y="converted",
                    color="group", barmode="group",
                    color_discrete_map={"control": CYAN, "treatment": PURPLE},
                    title="Conversion Rate by Device",
                    labels={"converted": "Conv. Rate"},
                    **{k: v for k, v in PLOTLY.items() if k != "template"},
                    template="plotly_dark",
                )
                fig2.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                                   font=dict(color=TEXT), margin=PLOTLY["margin"])
                c2.plotly_chart(fig2, use_container_width=True)

        with tab3:
            if "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"]).dt.date
                daily = df.groupby(["date", "group"]).agg(
                    users=("user_id", "count") if "user_id" in df.columns else ("converted", "count"),
                    conv_rate=("converted", "mean"),
                ).reset_index()
                fig3 = px.line(
                    daily, x="date", y="conv_rate", color="group",
                    color_discrete_map={"control": CYAN, "treatment": PURPLE},
                    title="Daily Conversion Rate Over Time",
                    labels={"conv_rate": "Conv. Rate", "date": "Date"},
                    markers=True, template="plotly_dark",
                )
                fig3.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                                   font=dict(color=TEXT), margin=PLOTLY["margin"])
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No `timestamp` column found — timeline not available")


# ─────────────────────────────────────────────────────────────
# PAGE 2: Experiment Design
# ─────────────────────────────────────────────────────────────
elif page == "🎯 Experiment Design":
    st.title("🎯 Experiment Design")
    st.markdown(f"<span style='color:{MUTED}'>Calculate required sample size BEFORE running your experiment</span>", unsafe_allow_html=True)
    st.markdown("---")

    col_inputs, col_result = st.columns([1, 1])

    with col_inputs:
        st.markdown("### Parameters")
        baseline = st.slider("Baseline conversion rate", 0.01, 0.40, 0.10, 0.005, format="%.3f")
        mde = st.slider("Minimum Detectable Effect (MDE)", 0.005, 0.10, 0.020, 0.005,
                        format="%.3f",
                        help="Smallest absolute difference worth detecting. E.g. 0.02 = you want to detect at least +2pp")
        st.markdown(f"<span class='pill'>Relative MDE: {mde/baseline:.1%}</span>"
                    f"<span class='pill'>Expected treatment rate: {baseline+mde:.3f}</span>",
                    unsafe_allow_html=True)

    result = calculate_sample_size(baseline, mde, alpha, power_target)

    with col_result:
        st.markdown("### Required Sample Size")
        c1, c2 = st.columns(2)
        kpi(c1, "Per Group", f"{result['n_per_group']:,}", "users needed", PURPLE)
        kpi(c2, "Total", f"{result['n_total']:,}", "both groups", CYAN)

    st.markdown("---")
    st.markdown("### 📅 How long will the experiment take?")
    daily_users = st.number_input("Daily users entering experiment", 100, 100_000, 3_000, 100)
    days_needed = result["n_total"] / daily_users
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Days Needed", f"{days_needed:.1f}", "at current traffic", YELLOW)
    kpi(c2, "Weeks", f"{days_needed/7:.1f}", "≥ 2 recommended", YELLOW)
    kpi(c3, "Min. Recommended", "14 days", "cover weekly cycles")

    if days_needed < 7:
        st.warning("⚠️ Less than 7 days — risk of **novelty effect** and missing weekly seasonality. Consider running at least 2 weeks.")
    elif days_needed > 60:
        st.warning("⚠️ Very long experiment. Consider a larger MDE or increasing traffic allocation.")
    else:
        st.success(f"✅ {days_needed:.0f} days — reasonable experiment length")

    st.markdown("---")
    st.markdown("### 📐 Sample Size Sensitivity")

    mde_range = np.arange(0.005, 0.101, 0.005)
    sizes = [calculate_sample_size(baseline, m, alpha, power_target)["n_per_group"]
             for m in mde_range]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mde_range * 100, y=sizes,
        mode="lines+markers",
        line=dict(color=PURPLE, width=3),
        marker=dict(color=PURPLE, size=7),
        fill="tozeroy",
        fillcolor=f"rgba(139,92,246,0.10)",
        name="N per group",
    ))
    fig.add_vline(x=mde * 100, line_dash="dash", line_color=CYAN,
                  annotation_text=f"Current MDE: {mde:.3f}", annotation_font_color=CYAN)
    fig.update_layout(
        title="Sample Size vs. Minimum Detectable Effect",
        xaxis_title="MDE (%)",
        yaxis_title="N per group",
        **PLOTLY,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📚 What is MDE?"):
        st.markdown(f"""
        The **Minimum Detectable Effect (MDE)** is the smallest change in conversion rate you
        care about detecting. Choosing the right MDE involves a business trade-off:

        - **Smaller MDE** → needs more users, longer experiment, more statistical precision
        - **Larger MDE** → faster experiment, but misses small real improvements

        **Rule of thumb:** Your MDE should be the smallest lift that would justify the
        engineering/design cost of shipping the feature.

        Current parameters: α = `{alpha}`, power = `{power_target}`, baseline = `{baseline:.1%}`
        """)


# ─────────────────────────────────────────────────────────────
# PAGE 3: SRM Validation
# ─────────────────────────────────────────────────────────────
elif page == "🔍 SRM Validation":
    st.title("🔍 SRM Check — Sample Ratio Mismatch")
    st.markdown(f"<span style='color:{MUTED}'>The #1 silent killer of A/B tests. Always validate before looking at results.</span>", unsafe_allow_html=True)
    st.markdown("---")

    df = get_df()
    if df is None:
        st.warning("👈 Go to **Dataset** tab first and load your data")
        st.stop()

    ctrl = df[df["group"] == "control"]
    treat = df[df["group"] == "treatment"]
    srm = check_srm(len(ctrl), len(treat), expected_split=0.5, alpha=alpha)

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Control Users", f"{srm['n_control']:,}", f"expected {srm['expected_control']:,}")
    kpi(c2, "Treatment Users", f"{srm['n_treatment']:,}", f"expected {srm['expected_treatment']:,}")
    kpi(c3, "Actual Split", f"{srm['actual_split_control']:.1%} / {1-srm['actual_split_control']:.1%}", "ctrl / treat")
    kpi(c4, "SRM p-value", f"{srm['p_value']:.4f}", "χ² test",
        RED if srm["srm_detected"] else GREEN)

    st.markdown("---")

    if srm["srm_detected"]:
        st.markdown(f"""
        <div class="verdict-lose">
            <h3>🚨 SRM Detected — Experiment is INVALID</h3>
            <p>The split between control ({srm['actual_split_control']:.1%}) and treatment
            ({1 - srm['actual_split_control']:.1%}) differs significantly from the expected 50/50
            split (p = {srm['p_value']:.4f} &lt; α = {alpha}).</p>
            <p><strong>Do NOT interpret results.</strong> Investigate the assignment mechanism.</p>
            <ul>
                <li>Check your assignment logic for bugs (especially client-side randomization)</li>
                <li>Look for bot traffic filtering differences between groups</li>
                <li>Verify your logging pipeline isn't dropping events for one group</li>
                <li>Check for redirect issues if using URL-based variants</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-win">
            <h3>✅ No SRM Detected — Experiment assignment looks healthy</h3>
            <p>The observed split ({srm['actual_split_control']:.1%} / {1-srm['actual_split_control']:.1%})
            is consistent with the expected 50/50 split (p = {srm['p_value']:.4f} &gt; α = {alpha}).</p>
            <p>Proceed to statistical analysis with confidence.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    # Visual comparison
    c1, c2 = st.columns(2)

    fig = go.Figure(go.Bar(
        x=["Control (actual)", "Treatment (actual)", "Control (expected)", "Treatment (expected)"],
        y=[srm["n_control"], srm["n_treatment"], srm["expected_control"], srm["expected_treatment"]],
        marker_color=[CYAN, PURPLE, f"rgba(6,182,212,0.3)", f"rgba(139,92,246,0.3)"],
        text=[f"{srm['n_control']:,}", f"{srm['n_treatment']:,}",
              f"{srm['expected_control']:,}", f"{srm['expected_treatment']:,}"],
        textposition="outside",
    ))
    fig.update_layout(title="Actual vs Expected Group Sizes", **PLOTLY)
    c1.plotly_chart(fig, use_container_width=True)

    # Pie chart actual
    fig2 = go.Figure(go.Pie(
        labels=["Control", "Treatment"],
        values=[srm["n_control"], srm["n_treatment"]],
        marker_colors=[CYAN, PURPLE],
        hole=0.5,
    ))
    fig2.update_layout(title="Actual Split", **PLOTLY)
    c2.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 4: Statistical Analysis
# ─────────────────────────────────────────────────────────────
elif page == "📈 Statistical Analysis":
    st.title("📈 Statistical Analysis")
    st.markdown(f"<span style='color:{MUTED}'>Frequentist Z-test + Bayesian Beta-Binomial — two lenses, one truth</span>", unsafe_allow_html=True)
    st.markdown("---")

    df = get_df()
    if df is None:
        st.warning("👈 Go to **Dataset** tab first and load your data")
        st.stop()

    ctrl = df[df["group"] == "control"]
    treat = df[df["group"] == "treatment"]

    n_c, conv_c = len(ctrl), ctrl["converted"].sum()
    n_t, conv_t = len(treat), treat["converted"].sum()

    zt = run_z_test(n_c, n_t, conv_c, conv_t, alpha)
    bay = bayesian_ab_test(conv_c, n_c, conv_t, n_t)

    tab_freq, tab_bayes, tab_seg = st.tabs(
        ["📐 Frequentist (Z-Test)", "🎲 Bayesian", "🏷️ Segment Analysis"]
    )

    # ── Frequentist ───────────────────────────────
    with tab_freq:
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Control CVR", f"{zt['rate_control']:.2%}", f"{conv_c:,} / {n_c:,}", CYAN)
        kpi(c2, "Treatment CVR", f"{zt['rate_treatment']:.2%}", f"{conv_t:,} / {n_t:,}", PURPLE)
        kpi(c3, "Absolute Lift", f"{zt['diff']:+.2%}",
            f"95% CI [{zt['ci_lower']:+.2%}, {zt['ci_upper']:+.2%}]",
            GREEN if zt["diff"] > 0 else RED)
        kpi(c4, "Relative Uplift", f"{zt['relative_uplift']:+.1f}%",
            f"p = {zt['p_value']:.4f}",
            GREEN if zt["significant"] else YELLOW)

        st.markdown("---")

        # Confidence interval waterfall
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[zt["ci_lower"] * 100, zt["ci_upper"] * 100],
            y=[0, 0],
            mode="lines",
            line=dict(color=PURPLE, width=6),
            name="95% CI",
        ))
        fig.add_trace(go.Scatter(
            x=[zt["diff"] * 100],
            y=[0],
            mode="markers",
            marker=dict(color=CYAN, size=16, symbol="diamond"),
            name="Observed difference",
        ))
        fig.add_vline(x=0, line_dash="dot", line_color=MUTED)
        fig.update_layout(
            title="Confidence Interval for Difference in Conversion Rates",
            xaxis_title="Difference (treatment - control) in percentage points",
            yaxis=dict(visible=False),
            showlegend=True,
            height=280,
            **PLOTLY,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Result box
        if zt["significant"]:
            st.markdown(f"""
            <div class="verdict-win">
                <h3>✅ Statistically Significant (p = {zt['p_value']:.4f})</h3>
                <p>Treatment conversion rate <strong>{zt['rate_treatment']:.2%}</strong>
                is significantly different from control <strong>{zt['rate_control']:.2%}</strong>
                (α = {alpha}).<br>
                Achieved power: <strong>{zt['achieved_power']:.1%}</strong></p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-neutral">
                <h3>⚪ Not Statistically Significant (p = {zt['p_value']:.4f})</h3>
                <p>We cannot reject H₀. The observed difference ({zt['diff']:+.2%}) may be due to
                random chance at α = {alpha}.<br>
                Achieved power: <strong>{zt['achieved_power']:.1%}</strong></p>
            </div>""", unsafe_allow_html=True)

        with st.expander("📚 Common mistakes — Peeking & multiple testing"):
            st.markdown("""
            **🔍 Peeking:** Stopping the test early when p < α inflates false positive rate.
            Always decide the sample size BEFORE running the experiment.

            **🎯 Multiple testing:** Testing 10 metrics and reporting only significant ones
            is p-hacking. Apply Bonferroni correction: α_adjusted = α / n_tests.

            **📏 Practical vs statistical significance:** A 0.01% lift significant with 10M users
            might not be worth shipping. Always consider the business impact.
            """)

    # ── Bayesian ──────────────────────────────────
    with tab_bayes:
        c1, c2, c3 = st.columns(3)
        kpi(c1, "P(Treatment > Control)", f"{bay['prob_treatment_better']:.1%}",
            "posterior probability", PURPLE)
        kpi(c2, "Expected Loss if Control", f"{bay['expected_loss_control']:.4f}",
            "risk of not shipping", CYAN)
        kpi(c3, "Expected Loss if Treatment", f"{bay['expected_loss_treatment']:.4f}",
            "risk of shipping", PINK)

        st.markdown("---")

        # Posterior distributions
        x_range = np.linspace(
            min(bay["samples_control"].min(), bay["samples_treatment"].min()) * 0.98,
            max(bay["samples_control"].max(), bay["samples_treatment"].max()) * 1.02,
            400,
        )
        from scipy.stats import beta as beta_dist
        a_c = int(conv_c + 1); b_c = int(n_c - conv_c + 1)
        a_t = int(conv_t + 1); b_t = int(n_t - conv_t + 1)

        pdf_c = beta_dist.pdf(x_range, a_c, b_c)
        pdf_t = beta_dist.pdf(x_range, a_t, b_t)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_range * 100, y=pdf_c, name="Control posterior",
            line=dict(color=CYAN, width=2.5),
            fill="tozeroy", fillcolor=f"rgba(6,182,212,0.15)",
        ))
        fig.add_trace(go.Scatter(
            x=x_range * 100, y=pdf_t, name="Treatment posterior",
            line=dict(color=PURPLE, width=2.5),
            fill="tozeroy", fillcolor=f"rgba(139,92,246,0.15)",
        ))
        fig.add_vline(x=bay["posterior_mean_control"] * 100, line_dash="dash",
                      line_color=CYAN, annotation_text="Control mean", annotation_font_color=CYAN)
        fig.add_vline(x=bay["posterior_mean_treatment"] * 100, line_dash="dash",
                      line_color=PURPLE, annotation_text="Treatment mean", annotation_font_color=PURPLE)
        fig.update_layout(
            title=f"Posterior Distributions — P(Treatment > Control) = {bay['prob_treatment_better']:.1%}",
            xaxis_title="Conversion Rate (%)",
            yaxis_title="Probability Density",
            **PLOTLY,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Distribution of difference
        c1, c2 = st.columns(2)
        diff_vals = bay["diff_samples"] * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=diff_vals, nbinsx=100,
            marker_color=PURPLE, opacity=0.8, name="Diff samples",
        ))
        fig2.add_vline(x=0, line_dash="dot", line_color=RED,
                       annotation_text="No effect (0)", annotation_font_color=RED)
        fig2.add_vline(x=float(np.mean(diff_vals)), line_dash="dash", line_color=CYAN,
                       annotation_text=f"Mean: {np.mean(diff_vals):.2f}pp", annotation_font_color=CYAN)
        fig2.update_layout(
            title="Posterior of Difference (treatment − control)",
            xaxis_title="Difference (pp)",
            yaxis_title="Frequency",
            **PLOTLY,
        )
        c1.plotly_chart(fig2, use_container_width=True)

        # 95% credible interval
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=[bay["ci_95_low"] * 100, bay["ci_95_high"] * 100],
            y=[0, 0], mode="lines",
            line=dict(color=PURPLE, width=8), name="95% Credible Interval",
        ))
        fig3.add_trace(go.Scatter(
            x=[float(np.mean(diff_vals))], y=[0], mode="markers",
            marker=dict(color=CYAN, size=16, symbol="diamond"),
            name="Posterior mean",
        ))
        fig3.add_vline(x=0, line_dash="dot", line_color=MUTED)
        fig3.update_layout(
            title=f"95% Credible Interval: [{bay['ci_95_low']:+.3%}, {bay['ci_95_high']:+.3%}]",
            xaxis_title="Difference in conversion rate (pp)",
            yaxis=dict(visible=False), height=260, **PLOTLY,
        )
        c2.plotly_chart(fig3, use_container_width=True)

    # ── Segment Analysis ──────────────────────────
    with tab_seg:
        if "device" not in df.columns and "country" not in df.columns:
            st.info("No segment columns (device, country) found in your data")
            st.stop()

        seg_col = st.selectbox("Segment by", [c for c in ["device", "country"] if c in df.columns])

        rows = []
        for seg_val in df[seg_col].unique():
            sub = df[df[seg_col] == seg_val]
            sc = sub[sub["group"] == "control"]
            st_ = sub[sub["group"] == "treatment"]
            if len(sc) > 50 and len(st_) > 50:
                z = run_z_test(len(sc), len(st_), sc["converted"].sum(), st_["converted"].sum(), alpha)
                rows.append({
                    "Segment": seg_val,
                    "N control": len(sc),
                    "N treatment": len(st_),
                    "CVR control": f"{z['rate_control']:.2%}",
                    "CVR treatment": f"{z['rate_treatment']:.2%}",
                    "Relative lift": f"{z['relative_uplift']:+.1f}%",
                    "p-value": f"{z['p_value']:.4f}",
                    "Significant": "✅" if z["significant"] else "❌",
                })

        seg_df = pd.DataFrame(rows)
        st.dataframe(seg_df, use_container_width=True, hide_index=True)

        # Forest plot
        fig = go.Figure()
        for i, row in enumerate(rows):
            sub = df[df[seg_col] == row["Segment"]]
            sc = sub[sub["group"] == "control"]
            st_ = sub[sub["group"] == "treatment"]
            z = run_z_test(len(sc), len(st_), sc["converted"].sum(), st_["converted"].sum(), alpha)
            color = GREEN if z["significant"] and z["diff"] > 0 else (RED if z["significant"] else MUTED)
            fig.add_trace(go.Scatter(
                x=[z["ci_lower"] * 100, z["ci_upper"] * 100],
                y=[row["Segment"], row["Segment"]],
                mode="lines", line=dict(color=color, width=5),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[z["diff"] * 100], y=[row["Segment"]],
                mode="markers", marker=dict(color=color, size=12, symbol="circle"),
                name=row["Segment"], showlegend=False,
            ))
        fig.add_vline(x=0, line_dash="dot", line_color=MUTED)
        fig.update_layout(
            title=f"Forest Plot — Lift by {seg_col.title()}",
            xaxis_title="Difference in conversion rate (pp)",
            **PLOTLY,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 5: Business Report
# ─────────────────────────────────────────────────────────────
elif page == "📋 Business Report":
    st.title("📋 Business Report")
    st.markdown(f"<span style='color:{MUTED}'>Executive summary and deployment recommendation</span>", unsafe_allow_html=True)
    st.markdown("---")

    df = get_df()
    if df is None:
        st.warning("👈 Go to **Dataset** tab first and load your data")
        st.stop()

    ctrl = df[df["group"] == "control"]
    treat = df[df["group"] == "treatment"]
    n_c, conv_c = len(ctrl), ctrl["converted"].sum()
    n_t, conv_t = len(treat), treat["converted"].sum()

    zt = run_z_test(n_c, n_t, conv_c, conv_t, alpha)
    bay = bayesian_ab_test(conv_c, n_c, conv_t, n_t)

    # Revenue projections
    avg_order = st.sidebar.number_input("Avg. order value ($)", 10, 1000, 88, 5)
    monthly_visitors = st.sidebar.number_input("Monthly visitors", 1_000, 10_000_000, 150_000, 10_000)

    impact = revenue_impact(zt["rate_control"], zt["rate_treatment"], avg_order, monthly_visitors)

    # Header verdict
    st.markdown("### 🏛️ Executive Summary")
    if zt["significant"] and zt["diff"] > 0:
        st.markdown(f"""
        <div class="verdict-win">
            <h3>🚀 SHIP IT — Treatment outperforms control</h3>
            <p>
            The new experience shows a statistically significant improvement in conversion rate
            (<strong>{zt['rate_control']:.2%} → {zt['rate_treatment']:.2%}</strong>)
            representing a <strong>{zt['relative_uplift']:+.1f}% relative uplift</strong>.<br><br>
            Bayesian confidence: <strong>{bay['prob_treatment_better']:.1%}</strong> probability
            that treatment is better.<br>
            Projected annual revenue impact: <strong>${impact['annual_uplift']:,.0f}</strong>
            </p>
        </div>""", unsafe_allow_html=True)
    elif zt["significant"] and zt["diff"] < 0:
        st.markdown(f"""
        <div class="verdict-lose">
            <h3>🛑 DO NOT SHIP — Treatment hurts conversion</h3>
            <p>
            The new experience shows a statistically significant <strong>decline</strong>
            ({zt['rate_control']:.2%} → {zt['rate_treatment']:.2%}, {zt['relative_uplift']:+.1f}%).<br>
            Keep the control and investigate what caused the regression.
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-neutral">
            <h3>⏳ INCONCLUSIVE — Keep running or redesign</h3>
            <p>
            No statistically significant difference detected (p = {zt['p_value']:.4f}).<br>
            Options: run longer to accumulate power, increase MDE, or redesign the variant.
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💰 Revenue Impact Projection")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Baseline Monthly Revenue", f"${impact['baseline_monthly']:,.0f}", "control rate", CYAN)
    kpi(c2, "Projected Monthly Revenue", f"${impact['treatment_monthly']:,.0f}", "treatment rate", PURPLE)
    kpi(c3, "Monthly Uplift", f"${impact['monthly_uplift']:+,.0f}", "incremental revenue",
        GREEN if impact["monthly_uplift"] > 0 else RED)
    kpi(c4, "Annual Uplift", f"${impact['annual_uplift']:+,.0f}", "12-month projection",
        GREEN if impact["annual_uplift"] > 0 else RED)

    # Monthly projection chart
    months = list(range(1, 13))
    baseline = [impact["baseline_monthly"] * m for m in months]
    projected = [impact["treatment_monthly"] * m for m in months]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=baseline, name="Control (baseline)",
        line=dict(color=CYAN, width=2.5, dash="dash"),
        fill="tozeroy", fillcolor="rgba(6,182,212,0.06)",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=projected, name="Treatment (projected)",
        line=dict(color=PURPLE, width=2.5),
        fill="tozeroy", fillcolor="rgba(139,92,246,0.10)",
    ))
    fig.update_layout(
        title="Cumulative Revenue: Baseline vs Treatment",
        xaxis_title="Month",
        yaxis_title="Cumulative Revenue ($)",
        **PLOTLY,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Full Results Summary")

    summary = {
        "Metric": ["Control CVR", "Treatment CVR", "Absolute Lift", "Relative Uplift",
                   "95% CI", "Z-statistic", "p-value", "Significant",
                   "P(Treatment > Control)", "Monthly Revenue Uplift", "Annual Revenue Uplift"],
        "Value": [
            f"{zt['rate_control']:.4%}",
            f"{zt['rate_treatment']:.4%}",
            f"{zt['diff']:+.4%}",
            f"{zt['relative_uplift']:+.2f}%",
            f"[{zt['ci_lower']:+.4%}, {zt['ci_upper']:+.4%}]",
            f"{zt['z_stat']:.4f}",
            f"{zt['p_value']:.4f}",
            "Yes ✅" if zt["significant"] else "No ❌",
            f"{bay['prob_treatment_better']:.2%}",
            f"${impact['monthly_uplift']:+,.0f}",
            f"${impact['annual_uplift']:+,.0f}",
        ],
    }
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Report as CSV",
        data=pd.DataFrame(summary).to_csv(index=False),
        file_name="ab_test_report.csv",
        mime="text/csv",
    )
