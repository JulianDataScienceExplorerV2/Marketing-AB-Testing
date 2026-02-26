"""
A/B Testing Lab — Marketing Science Dashboard
Clean, minimal dark design. No emoji clutter.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.statistics import (
    calculate_sample_size, check_srm,
    run_z_test, bayesian_ab_test, revenue_impact,
)
from src.data_generator import generate_ab_data

# ── Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Testing Lab",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────
P = "#7C6EF5"        # primary purple
P2 = "#A78BFA"       # light purple
CY = "#22D3EE"       # cyan
GR = "#34D399"       # green
RD = "#F87171"       # red
YL = "#FBBF24"       # yellow

BG    = "#080A12"
SB    = "#0C0E1A"    # sidebar
SURF  = "#0E1020"    # main area
CARD  = "#141628"    # card bg
CARD2 = "#191C30"    # card hover
BORD  = "#1E2145"
BORD2 = "#2A2D55"
TX    = "#E4E7FF"    # text
TX2   = "#8B90BE"    # muted text
TX3   = "#5A5F8A"    # dimmer text

# ── Global CSS ─────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* {{ box-sizing: border-box !important; }}

html, body, .stApp {{
    background: {BG} !important;
    font-family: 'Inter', sans-serif !important;
    color: {TX} !important;
}}

.main .block-container {{
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1360px !important;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {SB} !important;
    border-right: 1px solid {BORD} !important;
}}
section[data-testid="stSidebar"] * {{
    font-family: 'Inter', sans-serif !important;
}}

/* ── Radio nav items ── */
div[role="radiogroup"] label {{
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-size: 13px !important;
    color: {TX2} !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    font-weight: 400 !important;
}}
div[role="radiogroup"] label:hover {{
    background: {BORD} !important;
    color: {TX} !important;
}}
div[role="radiogroup"] label[data-testid="stMarkdownContainer"],
div[role="radiogroup"] label:has(input:checked) {{
    background: {BORD} !important;
    color: {P2} !important;
    font-weight: 500 !important;
}}
div[role="radiogroup"] input {{ display: none !important; }}

/* ── Headings ── */
h1 {{ color: {TX} !important; font-size: 22px !important; font-weight: 600 !important; letter-spacing: -0.3px !important; margin-bottom: 4px !important; }}
h2 {{ color: {TX} !important; font-size: 16px !important; font-weight: 600 !important; }}
h3 {{ color: {TX2} !important; font-size: 13px !important; font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; }}

/* ── Metrics ── */
[data-testid="metric-container"] {{
    background: {CARD} !important;
    border: 1px solid {BORD} !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
    transition: border-color 0.2s !important;
}}
[data-testid="metric-container"]:hover {{ border-color: {BORD2} !important; }}
[data-testid="metric-container"] label {{
    color: {TX3} !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    font-weight: 500 !important;
}}
div[data-testid="metric-value"] {{ color: {TX} !important; font-size: 24px !important; font-weight: 600 !important; font-family: 'Inter', sans-serif !important; }}
[data-testid="stMetricDelta"] {{ font-size: 12px !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {CARD} !important; border-radius: 10px !important;
    border: 1px solid {BORD} !important; gap: 2px !important; padding: 4px !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 7px !important; color: {TX2} !important;
    font-size: 13px !important; font-weight: 400 !important; padding: 7px 16px !important;
    background: transparent !important; border: none !important;
}}
.stTabs [aria-selected="true"] {{
    background: {BORD2} !important; color: {TX} !important;
    font-weight: 600 !important;
}}

/* ── Inputs / Selects ── */
.stSlider [data-testid="stSlider"] {{}}
.stSelectbox [data-baseweb="select"] > div {{
    background: {CARD} !important; border: 1px solid {BORD} !important;
    border-radius: 8px !important; color: {TX} !important;
}}
.stNumberInput input, .stTextInput input {{
    background: {CARD} !important; border: 1px solid {BORD} !important;
    border-radius: 8px !important; color: {TX} !important;
    font-family: 'Inter', sans-serif !important;
}}
.stNumberInput input:focus, .stTextInput input:focus {{
    border-color: {P} !important; box-shadow: 0 0 0 2px rgba(124,110,245,0.15) !important;
}}

/* ── Buttons ── */
.stButton button {{
    background: {P} !important; color: #fff !important; border: none !important;
    border-radius: 8px !important; font-family: 'Inter', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 8px 18px !important; transition: all 0.15s !important;
    cursor: pointer !important;
}}
.stButton button:hover {{ background: {P2} !important; transform: translateY(-1px) !important; }}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {{
    background: {CARD} !important; border: 1px dashed {BORD2} !important;
    border-radius: 10px !important;
}}

/* ── Dataframe ── */
.stDataFrame {{
    border-radius: 10px !important; overflow: hidden !important;
    border: 1px solid {BORD} !important;
}}
.stDataFrame thead th {{
    background: {CARD2} !important; color: {TX2} !important;
    font-size: 11px !important; text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}}
.stDataFrame tbody td {{ background: {CARD} !important; color: {TX} !important; }}

/* ── Alerts ── */
.stAlert {{ border-radius: 10px !important; border-left-width: 3px !important; }}

/* ── Sidebar slider ── */
.stSlider [data-baseweb="slider"] [data-testid="stSliderThumb"] {{
    background: {P} !important;
}}
.stSlider [role="slider"] {{ background: {P} !important; }}

/* ── Download button ── */
.stDownloadButton button {{
    background: transparent !important;
    border: 1px solid {BORD2} !important; color: {TX2} !important;
}}
.stDownloadButton button:hover {{
    background: {BORD} !important; color: {TX} !important; transform: none !important;
}}

/* ── Custom classes ── */
.page-header {{
    display: flex; align-items: flex-start; flex-direction: column;
    margin-bottom: 1.5rem;
}}
.page-title {{
    font-size: 20px; font-weight: 600; color: {TX}; margin: 0; letter-spacing: -0.2px;
}}
.page-subtitle {{
    font-size: 13px; color: {TX2}; margin: 4px 0 0; font-weight: 400;
}}
.divider {{ height: 1px; background: {BORD}; margin: 1.5rem 0; border: none; }}

.kcard {{
    background: {CARD}; border: 1px solid {BORD}; border-radius: 12px;
    padding: 20px 22px; transition: border-color 0.2s;
}}
.kcard:hover {{ border-color: {BORD2}; }}
.kcard .lbl {{
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.8px; color: {TX3}; margin-bottom: 8px;
}}
.kcard .val {{ font-size: 26px; font-weight: 600; color: {TX}; line-height: 1.1; }}
.kcard .sub {{ font-size: 11px; color: {TX2}; margin-top: 4px; }}
.kcard .badge {{
    display: inline-block; font-size: 10px; font-weight: 600;
    padding: 2px 8px; border-radius: 20px; margin-top: 6px;
}}

.verdict {{
    border-radius: 12px; padding: 20px 24px; margin: 1rem 0;
    border-left: 3px solid;
}}
.verdict.win  {{ background: rgba(52,211,153,0.07); border-color: {GR}; }}
.verdict.loss {{ background: rgba(248,113,113,0.07); border-color: {RD}; }}
.verdict.meh  {{ background: rgba(251,191,36,0.06); border-color: {YL}; }}

.verdict .vtitle {{
    font-size: 14px; font-weight: 600; margin-bottom: 8px;
}}
.verdict.win  .vtitle {{ color: {GR}; }}
.verdict.loss .vtitle {{ color: {RD}; }}
.verdict.meh  .vtitle {{ color: {YL}; }}
.verdict p {{ font-size: 13px; color: {TX2}; margin: 0; line-height: 1.6; }}

.tag {{
    display: inline-block; background: {BORD}; border-radius: 6px;
    padding: 3px 10px; font-size: 11px; color: {P2}; margin: 2px;
    font-family: 'JetBrains Mono', monospace;
}}
.section-label {{
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.8px; color: {TX3}; margin-bottom: 12px; display: block;
}}

/* Sidebar logo */
.sb-logo {{
    font-size: 15px; font-weight: 700; color: {TX};
    letter-spacing: -0.3px; padding: 4px 0 2px;
}}
.sb-sub {{
    font-size: 11px; color: {TX3}; margin-bottom: 16px;
}}
</style>
""", unsafe_allow_html=True)


# ── Plotly base theme ──────────────────────────────────────────
PL = dict(
    template="plotly_dark",
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(color=TX2, family="Inter, sans-serif", size=12),
    margin=dict(l=48, r=24, t=48, b=40),
    colorway=[P, CY, GR, YL, RD, "#F472B6"],
)

# ── Session state ──────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo">A/B Testing Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Marketing Science · v1.0</div>', unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Dataset", "Experiment Design", "SRM Validation",
         "Statistical Analysis", "Business Report"],
        label_visibility="collapsed",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Test Parameters</span>', unsafe_allow_html=True)

    alpha = st.slider("Significance level α", 0.01, 0.10, 0.05, 0.01)
    power_target = st.slider("Target power", 0.70, 0.95, 0.80, 0.05)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<span style="font-size:11px;color:{TX3}">Scipy · Statsmodels · Plotly</span>',
                unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────
def get_df():
    return st.session_state.df


def kcard(col, label, value, sub="", badge=None, badge_color=TX2):
    badge_html = (
        f'<div class="badge" style="background:rgba(124,110,245,0.12);color:{badge_color}">'
        f'{badge}</div>' if badge else ""
    )
    col.markdown(
        f'<div class="kcard">'
        f'<div class="lbl">{label}</div>'
        f'<div class="val">{value}</div>'
        f'<div class="sub">{sub}</div>'
        f'{badge_html}</div>',
        unsafe_allow_html=True,
    )


def header(title, subtitle=""):
    st.markdown(
        f'<div class="page-header">'
        f'<div class="page-title">{title}</div>'
        f'{"<div class=page-subtitle>" + subtitle + "</div>" if subtitle else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


def divider():
    st.markdown('<hr class="divider">', unsafe_allow_html=True)


def verdict(kind, title, body):
    st.markdown(
        f'<div class="verdict {kind}">'
        f'<div class="vtitle">{title}</div>'
        f'<p>{body}</p></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# PAGE: Dataset
# ══════════════════════════════════════════════════════════════
if page == "Dataset":
    header("Dataset", "Load your data or generate a synthetic e-commerce experiment")

    c_left, c_right = st.columns([1, 2], gap="large")

    with c_left:
        source = st.selectbox("Source", ["Synthetic data", "Upload CSV"],
                              label_visibility="collapsed")

        if source == "Synthetic data":
            st.markdown('<span class="section-label">Simulation parameters</span>',
                        unsafe_allow_html=True)
            n_u = st.number_input("Total users", 5_000, 200_000, 45_000, 5_000)
            c_r = st.slider("Control rate", 0.05, 0.30, 0.104, 0.001, format="%.3f")
            t_r = st.slider("Treatment rate", 0.05, 0.30, 0.127, 0.001, format="%.3f")
            if st.button("Generate dataset", use_container_width=True):
                with st.spinner("Generating..."):
                    st.session_state.df = generate_ab_data(n_users=n_u,
                                                            control_rate=c_r,
                                                            treatment_rate=t_r)
                st.rerun()
        else:
            up = st.file_uploader("CSV file", type="csv", label_visibility="collapsed",
                                   help="Required: group (control/treatment), converted (0/1)")
            if up:
                st.session_state.df = pd.read_csv(up)
                st.rerun()

    df = get_df()
    with c_right:
        if df is None:
            st.markdown(f"""
            <div class="kcard" style="padding:32px;text-align:center;">
                <div style="font-size:32px;color:{BORD2};margin-bottom:12px">◫</div>
                <div style="color:{TX2};font-size:14px;font-weight:500">No data loaded</div>
                <div style="color:{TX3};font-size:12px;margin-top:6px">
                    Generate synthetic data or upload a CSV to begin
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            ctrl  = df[df["group"] == "control"]
            treat = df[df["group"] == "treatment"]
            cr_c  = ctrl["converted"].mean()
            cr_t  = treat["converted"].mean()

            c1, c2, c3, c4 = st.columns(4, gap="small")
            kcard(c1, "Experiment size", f"{len(df):,}", "total users")
            kcard(c2, "Control CVR",  f"{cr_c:.2%}", f"{len(ctrl):,} users",
                  badge="control", badge_color=CY)
            kcard(c3, "Treatment CVR", f"{cr_t:.2%}", f"{len(treat):,} users",
                  badge="treatment", badge_color=P2)
            kcard(c4, "Relative lift",
                  f"{(cr_t - cr_c) / cr_c:+.1%}", "observed",
                  badge="significant" if abs(cr_t - cr_c) / cr_c > 0.05 else "marginal",
                  badge_color=GR if cr_t > cr_c else RD)

    if df is not None:
        divider()
        t1, t2, t3 = st.tabs(["Data preview", "Distributions", "Timeline"])

        with t1:
            st.dataframe(df.head(500), use_container_width=True, height=300)

        with t2:
            c1, c2 = st.columns(2, gap="medium")
            ctrl  = df[df["group"] == "control"]
            treat = df[df["group"] == "treatment"]
            cr_c  = ctrl["converted"].mean()
            cr_t  = treat["converted"].mean()

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Control", x=["Control"],   y=[cr_c * 100],
                                  marker_color=CY, width=0.4, text=f"{cr_c:.2%}",
                                  textposition="outside", textfont_color=TX2))
            fig.add_trace(go.Bar(name="Treatment", x=["Treatment"], y=[cr_t * 100],
                                  marker_color=P, width=0.4, text=f"{cr_t:.2%}",
                                  textposition="outside", textfont_color=TX2))
            fig.update_layout(title="Conversion Rate by Group", yaxis_title="CVR (%)",
                               showlegend=False, **PL)
            c1.plotly_chart(fig, use_container_width=True)

            if "device" in df.columns:
                dev = df.groupby(["device", "group"])["converted"].mean().reset_index()
                fig2 = px.bar(dev, x="device", y="converted", color="group", barmode="group",
                               color_discrete_map={"control": CY, "treatment": P},
                               template="plotly_dark",
                               labels={"converted": "CVR", "device": "Device"})
                fig2.update_layout(paper_bgcolor=CARD, plot_bgcolor=CARD,
                                    font=dict(color=TX2, family="Inter"),
                                    margin=PL["margin"], title="CVR by Device")
                c2.plotly_chart(fig2, use_container_width=True)

        with t3:
            if "timestamp" in df.columns:
                df["_date"] = pd.to_datetime(df["timestamp"]).dt.date
                daily = (df.groupby(["_date", "group"])["converted"]
                           .mean().reset_index()
                           .rename(columns={"_date": "date", "converted": "cvr"}))
                fig3 = px.line(daily, x="date", y="cvr", color="group", markers=True,
                                color_discrete_map={"control": CY, "treatment": P},
                                template="plotly_dark",
                                labels={"cvr": "CVR", "date": "Date"})
                fig3.update_layout(paper_bgcolor=CARD, plot_bgcolor=CARD,
                                    font=dict(color=TX2, family="Inter"),
                                    margin=PL["margin"], title="Daily CVR Over Time")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No timestamp column available for timeline view")


# ══════════════════════════════════════════════════════════════
# PAGE: Experiment Design
# ══════════════════════════════════════════════════════════════
elif page == "Experiment Design":
    header("Experiment Design",
           "Calculate the required sample size before running the experiment")

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown('<span class="section-label">Parameters</span>', unsafe_allow_html=True)
        baseline = st.slider("Baseline CVR", 0.01, 0.40, 0.10, 0.005, format="%.3f")
        mde = st.slider("Minimum Detectable Effect (MDE)", 0.005, 0.10, 0.020, 0.005,
                        format="%.3f",
                        help="Smallest absolute difference worth detecting")
        st.markdown(
            f'<span class="tag">Relative MDE: {mde/baseline:.1%}</span>'
            f'<span class="tag">Expected treatment: {baseline+mde:.3f}</span>',
            unsafe_allow_html=True,
        )

    res = calculate_sample_size(baseline, mde, alpha, power_target)

    with c2:
        st.markdown('<span class="section-label">Required sample</span>', unsafe_allow_html=True)
        ca, cb = st.columns(2, gap="small")
        kcard(ca, "Per group",   f"{res['n_per_group']:,}", "users", badge="each variant")
        kcard(cb, "Total users", f"{res['n_total']:,}",    "both groups")

    divider()
    st.markdown('<span class="section-label">Timeline estimator</span>', unsafe_allow_html=True)

    daily = st.number_input("Daily users entering experiment", 100, 100_000, 3_000, 100)
    days  = res["n_total"] / daily

    ca, cb, cc = st.columns(3, gap="small")
    kcard(ca, "Days needed",   f"{days:.1f}", "at current traffic",
          badge="fast" if days < 14 else ("ok" if days < 42 else "slow"),
          badge_color=GR if days < 14 else (YL if days < 42 else RD))
    kcard(cb, "Weeks",         f"{days/7:.1f}", "minimum 2 recommended")
    kcard(cc, "Recommendation", "14 + days", "cover weekly cycles")

    if days < 7:
        st.warning("Running less than 7 days risks novelty effects and misses weekly seasonality patterns. Consider at least 2 weeks.")
    elif days > 60:
        st.warning("Very long runtime. Consider increasing the MDE or allocating more traffic to the experiment.")
    else:
        st.success(f"Good experiment length — {days:.0f} days covers natural weekly cycles.")

    divider()
    st.markdown('<span class="section-label">Sample size sensitivity</span>', unsafe_allow_html=True)

    mde_r  = np.arange(0.005, 0.101, 0.005)
    sizes  = [calculate_sample_size(baseline, m, alpha, power_target)["n_per_group"] for m in mde_r]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mde_r * 100, y=sizes, mode="lines+markers",
        line=dict(color=P, width=2.5),
        marker=dict(color=P, size=6),
        fill="tozeroy", fillcolor=f"rgba(124,110,245,0.08)",
        name="N per group",
    ))
    fig.add_vline(x=mde * 100, line_dash="dash", line_color=CY, line_width=1.5,
                  annotation_text=f"  {mde:.3f}", annotation_font_color=CY)
    fig.update_layout(
        title="N per group vs Minimum Detectable Effect",
        xaxis_title="MDE (percentage points)",
        yaxis_title="Users per group",
        **PL,
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: SRM Validation
# ══════════════════════════════════════════════════════════════
elif page == "SRM Validation":
    header("SRM Validation",
           "Sample Ratio Mismatch — validate experiment integrity before reading results")

    df = get_df()
    if df is None:
        st.warning("Load data first in the Dataset tab")
        st.stop()

    ctrl  = df[df["group"] == "control"]
    treat = df[df["group"] == "treatment"]
    srm   = check_srm(len(ctrl), len(treat), alpha=alpha)

    c1, c2, c3, c4 = st.columns(4, gap="small")
    kcard(c1, "Control",   f"{srm['n_control']:,}",  f"expected {srm['expected_control']:,}")
    kcard(c2, "Treatment", f"{srm['n_treatment']:,}", f"expected {srm['expected_treatment']:,}")
    kcard(c3, "Actual split",
          f"{srm['actual_split_control']:.1%} / {1-srm['actual_split_control']:.1%}",
          "ctrl / treat")
    kcard(c4, "p-value (chi-square)", f"{srm['p_value']:.4f}",
          "SRM check",
          badge="SRM detected" if srm["srm_detected"] else "No SRM",
          badge_color=RD if srm["srm_detected"] else GR)

    divider()

    if srm["srm_detected"]:
        verdict("loss", "SRM Detected — Do not interpret results",
                f"The split ({srm['actual_split_control']:.1%} / {1-srm['actual_split_control']:.1%}) "
                f"deviates significantly from 50/50 (p = {srm['p_value']:.4f}). "
                "Investigate your assignment logic: client-side randomization bugs, "
                "bot filtering differences, logging gaps, or redirect issues.")
    else:
        verdict("win", "No SRM — Experiment assignment is healthy",
                f"Observed split ({srm['actual_split_control']:.1%} / "
                f"{1-srm['actual_split_control']:.1%}) is consistent with "
                f"50/50 (p = {srm['p_value']:.4f}). Safe to proceed with analysis.")

    divider()
    c1, c2 = st.columns(2, gap="medium")

    fig = go.Figure(go.Bar(
        x=["Control · actual", "Treatment · actual",
           "Control · expected", "Treatment · expected"],
        y=[srm["n_control"], srm["n_treatment"],
           srm["expected_control"], srm["expected_treatment"]],
        marker_color=[CY, P, f"rgba(34,211,238,0.25)", f"rgba(124,110,245,0.25)"],
        text=[f"{srm['n_control']:,}", f"{srm['n_treatment']:,}",
              f"{srm['expected_control']:,}", f"{srm['expected_treatment']:,}"],
        textposition="outside", textfont_color=TX2,
    ))
    fig.update_layout(title="Actual vs Expected Group Sizes", **PL)
    c1.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure(go.Pie(
        labels=["Control", "Treatment"],
        values=[srm["n_control"], srm["n_treatment"]],
        marker_colors=[CY, P], hole=0.55,
        textfont_color=TX2,
    ))
    fig2.update_layout(title="Actual Split", **PL)
    c2.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: Statistical Analysis
# ══════════════════════════════════════════════════════════════
elif page == "Statistical Analysis":
    header("Statistical Analysis",
           "Frequentist Z-test and Bayesian Beta-Binomial")

    df = get_df()
    if df is None:
        st.warning("Load data first in the Dataset tab")
        st.stop()

    ctrl  = df[df["group"] == "control"]
    treat = df[df["group"] == "treatment"]
    n_c, conv_c = len(ctrl), ctrl["converted"].sum()
    n_t, conv_t = len(treat), treat["converted"].sum()

    zt  = run_z_test(n_c, n_t, conv_c, conv_t, alpha)
    bay = bayesian_ab_test(conv_c, n_c, conv_t, n_t)

    t1, t2, t3 = st.tabs(["Frequentist", "Bayesian", "Segment Analysis"])

    # ── Frequentist ────────────────────────────────────────────
    with t1:
        c1, c2, c3, c4 = st.columns(4, gap="small")
        kcard(c1, "Control CVR",   f"{zt['rate_control']:.2%}",  f"{conv_c:,} / {n_c:,}")
        kcard(c2, "Treatment CVR", f"{zt['rate_treatment']:.2%}", f"{conv_t:,} / {n_t:,}")
        kcard(c3, "Absolute lift",
              f"{zt['diff']:+.3%}",
              f"95% CI  [{zt['ci_lower']:+.3%}, {zt['ci_upper']:+.3%}]",
              badge="+uplift" if zt["diff"] > 0 else "regression",
              badge_color=GR if zt["diff"] > 0 else RD)
        kcard(c4, "p-value",
              f"{zt['p_value']:.4f}",
              f"relative lift {zt['relative_uplift']:+.1f}%",
              badge="significant" if zt["significant"] else "not significant",
              badge_color=GR if zt["significant"] else YL)

        divider()

        # CI chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[zt["ci_lower"] * 100, zt["ci_upper"] * 100], y=[0, 0],
            mode="lines", line=dict(color=P, width=8), name="95% Confidence Interval",
        ))
        fig.add_trace(go.Scatter(
            x=[zt["diff"] * 100], y=[0], mode="markers",
            marker=dict(color=CY, size=18, symbol="diamond"), name="Observed difference",
        ))
        fig.add_vline(x=0, line_dash="dot", line_color=TX3, line_width=1.5,
                      annotation_text="  No effect", annotation_font_color=TX3)
        fig.update_layout(
            title="95% Confidence Interval for CVR Difference (treatment − control)",
            xaxis_title="Difference in percentage points",
            yaxis=dict(visible=False), height=240,
            legend=dict(orientation="h", y=1.2, x=0),
            **PL,
        )
        st.plotly_chart(fig, use_container_width=True)

        if zt["significant"]:
            verdict("win", "Statistically significant",
                    f"Treatment CVR ({zt['rate_treatment']:.2%}) differs significantly from "
                    f"control ({zt['rate_control']:.2%}) at α = {alpha}. "
                    f"Achieved power: {zt['achieved_power']:.1%}.")
        else:
            verdict("meh", "Not statistically significant",
                    f"Cannot reject H₀ at α = {alpha} (p = {zt['p_value']:.4f}). "
                    f"The observed difference ({zt['diff']:+.3%}) may be due to chance. "
                    f"Achieved power: {zt['achieved_power']:.1%}.")

        with st.expander("Common pitfalls — peeking & multiple testing"):
            st.markdown(f"""
            **Peeking:** Stopping the test early when p < α inflates false positive rate above {alpha}.
            Always commit to the sample size before running.

            **Multiple testing:** Testing 10 metrics and reporting only significant ones is p-hacking.
            Apply Bonferroni correction: α_adj = α / n_tests.

            **Practical vs statistical significance:** A statistically significant 0.1% lift with
            10M users may not justify the engineering cost. Always compute revenue impact.
            """)

    # ── Bayesian ───────────────────────────────────────────────
    with t2:
        c1, c2, c3 = st.columns(3, gap="small")
        kcard(c1, "P(treatment > control)", f"{bay['prob_treatment_better']:.1%}",
              "posterior probability",
              badge="confident" if bay["prob_treatment_better"] > 0.95 else "uncertain",
              badge_color=GR if bay["prob_treatment_better"] > 0.95 else YL)
        kcard(c2, "Expected loss if control",
              f"{bay['expected_loss_control']:.4f}", "risk of not shipping")
        kcard(c3, "Expected loss if treatment",
              f"{bay['expected_loss_treatment']:.4f}", "risk of shipping")

        divider()

        # Posteriors
        from scipy.stats import beta as beta_dist
        x  = np.linspace(
            min(bay["samples_control"].min(), bay["samples_treatment"].min()) * 0.97,
            max(bay["samples_control"].max(), bay["samples_treatment"].max()) * 1.03,
            500,
        )
        a_c = int(conv_c + 1); b_c = int(n_c - conv_c + 1)
        a_t = int(conv_t + 1); b_t = int(n_t - conv_t + 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x * 100, y=beta_dist.pdf(x, a_c, b_c),
            name="Control", line=dict(color=CY, width=2.5),
            fill="tozeroy", fillcolor="rgba(34,211,238,0.10)",
        ))
        fig.add_trace(go.Scatter(
            x=x * 100, y=beta_dist.pdf(x, a_t, b_t),
            name="Treatment", line=dict(color=P, width=2.5),
            fill="tozeroy", fillcolor="rgba(124,110,245,0.10)",
        ))
        fig.add_vline(x=bay["posterior_mean_control"] * 100,
                      line_dash="dot", line_color=CY, line_width=1)
        fig.add_vline(x=bay["posterior_mean_treatment"] * 100,
                      line_dash="dot", line_color=P, line_width=1)
        fig.update_layout(
            title=f"Posterior distributions  ·  P(treatment > control) = {bay['prob_treatment_better']:.1%}",
            xaxis_title="Conversion Rate (%)",
            yaxis_title="Density",
            **PL,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Difference posterior
        c1, c2 = st.columns(2, gap="medium")
        diff_pp = bay["diff_samples"] * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=diff_pp, nbinsx=120, marker_color=P, opacity=0.75,
            name="difference",
        ))
        fig2.add_vline(x=0, line_dash="dot", line_color=RD, line_width=1.5)
        fig2.add_vline(x=float(np.mean(diff_pp)), line_dash="dash",
                       line_color=CY, line_width=1.5,
                       annotation_text=f"  mean {np.mean(diff_pp):.2f} pp",
                       annotation_font_color=CY)
        fig2.update_layout(
            title="Posterior of Difference (treatment − control)",
            xaxis_title="Difference (pp)", yaxis_title="Frequency", **PL,
        )
        c1.plotly_chart(fig2, use_container_width=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=[bay["ci_95_low"] * 100, bay["ci_95_high"] * 100],
            y=[0, 0], mode="lines",
            line=dict(color=P, width=10), name="95% Credible Interval",
        ))
        fig3.add_trace(go.Scatter(
            x=[float(np.mean(diff_pp))], y=[0], mode="markers",
            marker=dict(color=CY, size=18, symbol="diamond"), name="Posterior mean",
        ))
        fig3.add_vline(x=0, line_dash="dot", line_color=TX3, line_width=1)
        fig3.update_layout(
            title=f"95% Credible Interval: [{bay['ci_95_low']:+.3%}, {bay['ci_95_high']:+.3%}]",
            xaxis_title="Difference (pp)",
            yaxis=dict(visible=False), height=250, **PL,
        )
        c2.plotly_chart(fig3, use_container_width=True)

    # ── Segment Analysis ───────────────────────────────────────
    with t3:
        seg_cols = [c for c in ["device", "country"] if c in df.columns]
        if not seg_cols:
            st.info("No segment columns (device, country) found")
            st.stop()
        seg = st.selectbox("Segment by", seg_cols)

        rows = []
        for val in df[seg].dropna().unique():
            sub  = df[df[seg] == val]
            sc   = sub[sub["group"] == "control"]
            st_  = sub[sub["group"] == "treatment"]
            if len(sc) > 30 and len(st_) > 30:
                z = run_z_test(len(sc), len(st_),
                               sc["converted"].sum(), st_["converted"].sum(), alpha)
                rows.append({
                    "Segment":    val,
                    "N ctrl":     len(sc),
                    "N treat":    len(st_),
                    "CVR ctrl":   f"{z['rate_control']:.2%}",
                    "CVR treat":  f"{z['rate_treatment']:.2%}",
                    "Lift":       f"{z['relative_uplift']:+.1f}%",
                    "p-value":    f"{z['p_value']:.4f}",
                    "Sig.":       "Yes" if z["significant"] else "No",
                })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        divider()
        fig = go.Figure()
        for r in rows:
            sub = df[df[seg] == r["Segment"]]
            sc  = sub[sub["group"] == "control"]
            st_ = sub[sub["group"] == "treatment"]
            z   = run_z_test(len(sc), len(st_),
                             sc["converted"].sum(), st_["converted"].sum(), alpha)
            color = GR if z["significant"] and z["diff"] > 0 else (
                    RD if z["significant"] else TX3)
            fig.add_trace(go.Scatter(
                x=[z["ci_lower"] * 100, z["ci_upper"] * 100],
                y=[r["Segment"], r["Segment"]],
                mode="lines", line=dict(color=color, width=6), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[z["diff"] * 100], y=[r["Segment"]],
                mode="markers", marker=dict(color=color, size=12, symbol="circle"),
                name=r["Segment"], showlegend=False,
            ))
        fig.add_vline(x=0, line_dash="dot", line_color=TX3, line_width=1)
        fig.update_layout(
            title=f"Forest Plot — Lift by {seg}",
            xaxis_title="Difference (percentage points)", **PL,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: Business Report
# ══════════════════════════════════════════════════════════════
elif page == "Business Report":
    header("Business Report", "Executive summary and deployment recommendation")

    df = get_df()
    if df is None:
        st.warning("Load data first in the Dataset tab")
        st.stop()

    ctrl  = df[df["group"] == "control"]
    treat = df[df["group"] == "treatment"]
    n_c, conv_c = len(ctrl), ctrl["converted"].sum()
    n_t, conv_t = len(treat), treat["converted"].sum()

    zt  = run_z_test(n_c, n_t, conv_c, conv_t, alpha)
    bay = bayesian_ab_test(conv_c, n_c, conv_t, n_t)

    c_side, _ = st.columns([1, 3])
    with c_side:
        aov      = st.number_input("Avg. order value ($)", 10, 1_000, 88, 5)
        monthly  = st.number_input("Monthly visitors",     1_000, 10_000_000, 150_000, 10_000)

    imp = revenue_impact(zt["rate_control"], zt["rate_treatment"], aov, monthly)
    divider()

    # Verdict
    st.markdown('<span class="section-label">Recommendation</span>', unsafe_allow_html=True)
    if zt["significant"] and zt["diff"] > 0:
        verdict("win", "Ship the treatment",
                f"Treatment CVR ({zt['rate_treatment']:.2%}) significantly outperforms control "
                f"({zt['rate_control']:.2%}), a {zt['relative_uplift']:+.1f}% relative uplift. "
                f"Bayesian confidence: {bay['prob_treatment_better']:.1%}. "
                f"Projected annual revenue impact: <strong>${imp['annual_uplift']:,.0f}</strong>.")
    elif zt["significant"] and zt["diff"] < 0:
        verdict("loss", "Do not ship — treatment hurts conversion",
                f"Treatment CVR ({zt['rate_treatment']:.2%}) is significantly below control "
                f"({zt['rate_control']:.2%}), a {zt['relative_uplift']:+.1f}% regression. "
                "Keep the control and investigate the root cause.")
    else:
        verdict("meh", "Inconclusive — continue or redesign",
                f"No significant difference detected (p = {zt['p_value']:.4f}). "
                "Options: run longer to accumulate statistical power, "
                "revisit the variant design, or expand the MDE threshold.")

    divider()
    st.markdown('<span class="section-label">Revenue projections</span>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="small")
    kcard(c1, "Baseline monthly",   f"${imp['baseline_monthly']:,.0f}",  "control CVR")
    kcard(c2, "Projected monthly",  f"${imp['treatment_monthly']:,.0f}", "treatment CVR")
    kcard(c3, "Monthly uplift",
          f"${imp['monthly_uplift']:+,.0f}", "incremental",
          badge="gain" if imp["monthly_uplift"] > 0 else "loss",
          badge_color=GR if imp["monthly_uplift"] > 0 else RD)
    kcard(c4, "Annual uplift",
          f"${imp['annual_uplift']:+,.0f}", "12-month",
          badge="gain" if imp["annual_uplift"] > 0 else "loss",
          badge_color=GR if imp["annual_uplift"] > 0 else RD)

    months = list(range(1, 13))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=[imp["baseline_monthly"] * m for m in months],
        name="Baseline (control)", line=dict(color=CY, width=2, dash="dash"),
        fill="tozeroy", fillcolor="rgba(34,211,238,0.05)",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=[imp["treatment_monthly"] * m for m in months],
        name="Projected (treatment)", line=dict(color=P, width=2.5),
        fill="tozeroy", fillcolor="rgba(124,110,245,0.09)",
    ))
    fig.update_layout(
        title="Cumulative Revenue — Baseline vs Treatment",
        xaxis_title="Month", yaxis_title="Cumulative Revenue ($)",
        legend=dict(orientation="h", y=1.1), **PL,
    )
    st.plotly_chart(fig, use_container_width=True)

    divider()
    st.markdown('<span class="section-label">Full results</span>', unsafe_allow_html=True)

    summary_df = pd.DataFrame({
        "Metric": ["Control CVR", "Treatment CVR", "Absolute Lift", "Relative Uplift",
                   "95% CI", "Z-statistic", "p-value", "Significant",
                   "P(treatment > control)", "Monthly Revenue Uplift", "Annual Revenue Uplift"],
        "Value": [
            f"{zt['rate_control']:.4%}",
            f"{zt['rate_treatment']:.4%}",
            f"{zt['diff']:+.4%}",
            f"{zt['relative_uplift']:+.2f}%",
            f"[{zt['ci_lower']:+.4%}, {zt['ci_upper']:+.4%}]",
            f"{zt['z_stat']:.4f}",
            f"{zt['p_value']:.4f}",
            "Yes" if zt["significant"] else "No",
            f"{bay['prob_treatment_better']:.2%}",
            f"${imp['monthly_uplift']:+,.0f}",
            f"${imp['annual_uplift']:+,.0f}",
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download report (CSV)",
        data=summary_df.to_csv(index=False),
        file_name="ab_test_report.csv",
        mime="text/csv",
    )
