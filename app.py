"""
app.py  ── IPL Match Intelligence System
=========================================
A Streamlit web app for IPL win prediction, player analytics, and insights.

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas    as pd
import numpy     as np
import pickle, os, warnings
import plotly.express        as px
import plotly.graph_objects  as go
from plotly.subplots  import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/ipl_data.csv"
MODEL_PATH  = "models/win_probability_model.pkl"
FEATURES    = ["runs_left", "balls_left", "wickets_left",
               "current_run_rate", "required_run_rate"]

# Colour palette ─ IPL-inspired
CLR_GOLD   = "#F5A623"
CLR_RED    = "#E63946"
CLR_GREEN  = "#2DC653"
CLR_BLUE   = "#1D3557"
CLR_LIGHT  = "#A8DADC"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "IPL Match Intelligence System",
    page_icon  = "🏏",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── General ── */
  html, body, [class*="css"] { font-family: 'Segoe UI', Tahoma, sans-serif; }

  /* ── Header banner ── */
  .header-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.4rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 5px solid #F5A623;
  }
  .header-banner h1 { color: #F5A623; margin: 0; font-size: 2rem; }
  .header-banner p  { color: #a8dadc; margin: 0.3rem 0 0; font-size: 1rem; }

  /* ── Metric cards ── */
  .metric-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-card .label { color: #a8dadc; font-size: 0.82rem; text-transform: uppercase; letter-spacing: .06em; }
  .metric-card .value { color: #F5A623; font-size: 2rem; font-weight: 700; margin: .3rem 0; }
  .metric-card .delta { color: #888; font-size: 0.78rem; }

  /* ── Insight pills ── */
  .insight-pill {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    margin: 0.3rem 0.2rem;
    font-size: 0.88rem;
    font-weight: 500;
  }
  .insight-danger  { background: #3d1a1a; border: 1px solid #E63946; color: #ff8080; }
  .insight-warning { background: #3d2d0a; border: 1px solid #F5A623; color: #ffd080; }
  .insight-success { background: #0a2d1a; border: 1px solid #2DC653; color: #80ff80; }
  .insight-info    { background: #0a1a3d; border: 1px solid #A8DADC; color: #a8dadc; }

  /* ── Win probability gauge ── */
  .prob-display {
    text-align: center;
    padding: 1rem;
  }
  .prob-number { font-size: 3.5rem; font-weight: 800; }
  .prob-label  { font-size: 1rem; color: #888; margin-top: -0.5rem; }

  /* ── Section titles ── */
  .section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #F5A623;
    border-bottom: 2px solid #F5A623;
    padding-bottom: 0.3rem;
    margin-bottom: 1rem;
  }

  /* ── Sidebar tweaks ── */
  section[data-testid="stSidebar"] { background-color: #0f0f23 !important; }
  section[data-testid="stSidebar"] .css-1d391kg { background-color: #0f0f23 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADERS  (cached for performance)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load the IPL ball-by-ball CSV into a DataFrame."""
    if not os.path.exists(path):
        return pd.DataFrame()   # handled in UI
    df = pd.read_csv(path)
    # Basic type safety
    df["runs_off_bat"]   = pd.to_numeric(df["runs_off_bat"],   errors="coerce").fillna(0)
    df["is_wicket"]      = pd.to_numeric(df["is_wicket"],      errors="coerce").fillna(0)
    df["season"]         = pd.to_numeric(df["season"],         errors="coerce")
    return df


def load_model(path: str):
    if not os.path.exists("data/ipl_data.csv"):
        import generate_data
        generate_data.generate_dataset().to_csv("data/ipl_data.csv", index=False)
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(path):
        import train_model
        train_model.train()
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_win_probability(model, runs_left: float, balls_left: float,
                            wickets_left: float, crr: float, rrr: float) -> float:
    """Return batting-team win probability (0–1)."""
    X = pd.DataFrame([{
        "runs_left"         : runs_left,
        "balls_left"        : balls_left,
        "wickets_left"      : wickets_left,
        "current_run_rate"  : crr,
        "required_run_rate" : rrr,
    }])
    proba = model.predict_proba(X)[0]
    # Class 1 = batting team wins
    return float(proba[1]) if len(proba) > 1 else float(proba[0])


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-batsman stats: runs, dismissals, balls, SR, avg."""
    stats = (
        df.groupby("batsman")
          .agg(
              total_runs  = ("runs_off_bat", "sum"),
              balls_faced = ("runs_off_bat", "count"),
              dismissals  = ("is_wicket",    "sum"),
          )
          .reset_index()
    )
    stats["strike_rate"] = (stats["total_runs"] / stats["balls_faced"] * 100).round(2)
    stats["average"]     = np.where(
        stats["dismissals"] > 0,
        (stats["total_runs"] / stats["dismissals"]).round(2),
        stats["total_runs"].astype(float),   # not out throughout
    )
    # Only include players with meaningful data
    stats = stats[stats["balls_faced"] >= 20].sort_values("total_runs", ascending=False)
    return stats.reset_index(drop=True)


def plot_player_comparison(stats_df: pd.DataFrame, selected: list[str]) -> go.Figure:
    """Radar / bar chart comparing selected players."""
    sub = stats_df[stats_df["batsman"].isin(selected)].copy()
    if sub.empty:
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Total Runs", "Strike Rate", "Average"],
        horizontal_spacing=0.08,
    )
    colours = [CLR_GOLD, CLR_LIGHT, CLR_RED, CLR_GREEN, "#B388FF", "#80DEEA"]

    for i, (_, row) in enumerate(sub.iterrows()):
        c = colours[i % len(colours)]
        fig.add_trace(go.Bar(name=row["batsman"], x=[row["batsman"]],
                             y=[row["total_runs"]], marker_color=c,
                             showlegend=(i == 0)), row=1, col=1)
        fig.add_trace(go.Bar(name=row["batsman"], x=[row["batsman"]],
                             y=[row["strike_rate"]], marker_color=c,
                             showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(name=row["batsman"], x=[row["batsman"]],
                             y=[row["average"]], marker_color=c,
                             showlegend=False), row=1, col=3)

    fig.update_layout(
        paper_bgcolor="#0f0f23",
        plot_bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        title_text="Player Comparison",
        title_font_color=CLR_GOLD,
        showlegend=False,
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#2a2a4e")
    return fig


def plot_top_batsmen(stats_df: pd.DataFrame, n: int = 10) -> go.Figure:
    """Horizontal bar chart of top-N run scorers."""
    top = stats_df.head(n).sort_values("total_runs")
    fig = go.Figure(go.Bar(
        x             = top["total_runs"],
        y             = top["batsman"],
        orientation   = "h",
        marker_color  = CLR_GOLD,
        text          = top["total_runs"].astype(int),
        textposition  = "outside",
        textfont      = dict(color="white", size=11),
    ))
    fig.update_layout(
        paper_bgcolor = "#0f0f23",
        plot_bgcolor  = "#1a1a2e",
        font_color    = "#e0e0e0",
        title_text    = f"Top {n} Run Scorers",
        title_font_color = CLR_GOLD,
        height        = 380,
        margin        = dict(l=140, r=80, t=50, b=20),
        xaxis         = dict(showgrid=True, gridcolor="#2a2a4e"),
        yaxis         = dict(showgrid=False),
    )
    return fig


def plot_strike_rate_scatter(stats_df: pd.DataFrame) -> go.Figure:
    """Scatter plot: total runs vs strike rate, size = balls faced."""
    top50 = stats_df.head(50)
    fig = px.scatter(
        top50,
        x           = "total_runs",
        y           = "strike_rate",
        size        = "balls_faced",
        color       = "average",
        hover_name  = "batsman",
        hover_data  = {"balls_faced": True, "average": ":.1f"},
        color_continuous_scale = "Plasma",
        labels      = {"total_runs": "Total Runs", "strike_rate": "Strike Rate"},
        title       = "Runs vs Strike Rate (size = balls faced, colour = avg)",
    )
    fig.update_layout(
        paper_bgcolor    = "#0f0f23",
        plot_bgcolor     = "#1a1a2e",
        font_color       = "#e0e0e0",
        title_font_color = CLR_GOLD,
        height           = 420,
        coloraxis_colorbar = dict(title="Average", tickfont=dict(color="#e0e0e0"),
                                  titlefont=dict(color="#e0e0e0")),
    )
    fig.update_xaxes(gridcolor="#2a2a4e")
    fig.update_yaxes(gridcolor="#2a2a4e")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# INSIGHT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_insights(runs_left: float, balls_left: float,
                      wickets_left: float, crr: float, rrr: float,
                      win_prob: float) -> list[dict]:
    """
    Rule-based insight engine.
    Returns a list of dicts: {text, level}  where level ∈ {danger, warning, success, info}.
    """
    insights = []

    # ── Win probability insights ──────────────────────────────────────────────
    if win_prob >= 0.80:
        insights.append({"text": f"🟢 Strong favourites — win probability {win_prob*100:.0f}%", "level": "success"})
    elif win_prob >= 0.60:
        insights.append({"text": f"🔵 Comfortable position — win probability {win_prob*100:.0f}%", "level": "info"})
    elif win_prob >= 0.40:
        insights.append({"text": f"🟡 Evenly contested — win probability {win_prob*100:.0f}%", "level": "warning"})
    else:
        insights.append({"text": f"🔴 Uphill task — win probability {win_prob*100:.0f}%", "level": "danger"})

    # ── Required run rate insights ────────────────────────────────────────────
    if rrr > 15:
        insights.append({"text": "🚨 Required run rate > 15 — nearly impossible territory!", "level": "danger"})
    elif rrr > 12:
        insights.append({"text": "⚠️ Required run rate > 12 — need maximums every over", "level": "danger"})
    elif rrr > 10:
        insights.append({"text": "⚡ High pressure — required run rate above 10", "level": "warning"})
    elif rrr < 6 and balls_left > 0:
        insights.append({"text": "✅ Run rate comfortable — target well within reach", "level": "success"})

    # ── Wickets insights ──────────────────────────────────────────────────────
    if wickets_left == 1:
        insights.append({"text": "🚨 Last wicket standing — one ball can end the match!", "level": "danger"})
    elif wickets_left <= 2:
        insights.append({"text": "⚠️ Extremely high collapse risk — only tail-enders left", "level": "danger"})
    elif wickets_left <= 4:
        insights.append({"text": "⚡ High collapse risk — lower order in play", "level": "warning"})
    elif wickets_left == 10:
        insights.append({"text": "✅ All wickets intact — maximum batting resources available", "level": "success"})

    # ── Balls / overs remaining ───────────────────────────────────────────────
    overs_left = balls_left / 6
    if balls_left <= 6:
        insights.append({"text": f"🏁 Final over — {balls_left} ball(s) remaining", "level": "warning"})
    elif overs_left <= 3:
        insights.append({"text": f"💥 Death overs — {overs_left:.1f} overs left, big shots needed", "level": "info"})
    elif overs_left >= 15:
        insights.append({"text": f"🕐 Plenty of time — {overs_left:.1f} overs remaining", "level": "info"})

    # ── Run rate comparison ───────────────────────────────────────────────────
    gap = rrr - crr
    if gap > 4:
        insights.append({"text": f"📉 Required rate ({rrr:.1f}) far exceeds current rate ({crr:.1f})", "level": "danger"})
    elif gap > 2:
        insights.append({"text": f"📊 Batting team needs to up the ante — RRR is {gap:.1f} above CRR", "level": "warning"})
    elif gap < -2:
        insights.append({"text": f"📈 Current rate exceeds required rate by {-gap:.1f} — ahead of schedule", "level": "success"})

    # ── Runs needed ───────────────────────────────────────────────────────────
    if runs_left <= 0:
        insights.append({"text": "🏆 Target achieved — batting team wins!", "level": "success"})
    elif runs_left <= 6 and balls_left >= 6:
        insights.append({"text": f"🎯 Just {int(runs_left)} run(s) needed with {int(balls_left)} balls — almost there!", "level": "success"})
    elif runs_left > balls_left * 2:
        insights.append({"text": "💀 Runs required per ball > 2 — essentially asking for sixes every ball", "level": "danger"})

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# WIN PROBABILITY GAUGE CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_win_gauge(win_prob: float, team_name: str = "Batting Team") -> go.Figure:
    """Semi-circular gauge for win probability."""
    pct = win_prob * 100
    color = CLR_GREEN if pct >= 60 else (CLR_GOLD if pct >= 40 else CLR_RED)

    fig = go.Figure(go.Indicator(
        mode       = "gauge+number+delta",
        value      = pct,
        number     = {"suffix": "%", "font": {"size": 48, "color": color}},
        delta      = {"reference": 50, "relative": False,
                      "font": {"size": 16},
                      "increasing": {"color": CLR_GREEN},
                      "decreasing": {"color": CLR_RED}},
        gauge      = {
            "axis"  : {"range": [0, 100], "tickwidth": 1,
                       "tickcolor": "#888", "tickfont": {"color": "#888"}},
            "bar"   : {"color": color, "thickness": 0.28},
            "bgcolor"   : "#1a1a2e",
            "borderwidth": 0,
            "steps" : [
                {"range": [0,  40],  "color": "#2d0a0a"},
                {"range": [40, 60],  "color": "#2d2a0a"},
                {"range": [60, 100], "color": "#0a2d1a"},
            ],
            "threshold": {
                "line"  : {"color": "white", "width": 3},
                "thickness": 0.8,
                "value" : 50,
            },
        },
        title = {"text": f"{team_name}<br><span style='font-size:13px;color:#888'>Win Probability</span>",
                 "font": {"size": 18, "color": "#e0e0e0"}},
    ))
    fig.update_layout(
        paper_bgcolor = "#0f0f23",
        height        = 320,
        margin        = dict(l=30, r=30, t=60, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY HISTORY CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_prob_history(history: list[tuple]) -> go.Figure:
    """Line chart of win probability over simulated overs."""
    if len(history) < 2:
        return go.Figure()
    over_labels = [h[0] for h in history]
    probs       = [h[1] * 100 for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x           = over_labels,
        y           = probs,
        mode        = "lines+markers",
        line        = dict(color=CLR_GOLD, width=2.5),
        marker      = dict(color=CLR_GOLD, size=7),
        fill        = "tozeroy",
        fillcolor   = "rgba(245,166,35,0.12)",
        name        = "Win %",
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="#666",
                  annotation_text="50%", annotation_font_color="#888")
    fig.update_layout(
        paper_bgcolor    = "#0f0f23",
        plot_bgcolor     = "#1a1a2e",
        font_color       = "#e0e0e0",
        title_text       = "Win Probability History",
        title_font_color = CLR_GOLD,
        height           = 280,
        margin           = dict(l=20, r=20, t=50, b=20),
        xaxis = dict(title="Simulation Step", showgrid=False),
        yaxis = dict(title="Win %", range=[0, 100], gridcolor="#2a2a4e"),
        showlegend = False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TEAM PERFORMANCE CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_team_win_rates(df: pd.DataFrame) -> go.Figure:
    """Bar chart of team win rates in 2nd innings."""
    df2 = df[df["innings"] == 2].copy()
    if df2.empty:
        return go.Figure()

    # Last ball of each match for each team
    summary = (
        df2.sort_values(["match_id", "over", "ball"])
           .groupby(["match_id", "batting_team"])
           .last()
           .reset_index()[["batting_team", "result"]]
    )
    rates = (
        summary.groupby("batting_team")["result"]
               .agg(wins="sum", played="count")
               .reset_index()
    )
    rates["win_rate"] = (rates["wins"] / rates["played"] * 100).round(1)
    rates = rates.sort_values("win_rate", ascending=True)

    fig = go.Figure(go.Bar(
        x            = rates["win_rate"],
        y            = rates["batting_team"],
        orientation  = "h",
        marker_color = rates["win_rate"].apply(
            lambda v: CLR_GREEN if v >= 55 else (CLR_GOLD if v >= 45 else CLR_RED)
        ),
        text         = rates["win_rate"].astype(str) + "%",
        textposition = "outside",
        textfont     = dict(color="white", size=11),
    ))
    fig.update_layout(
        paper_bgcolor    = "#0f0f23",
        plot_bgcolor     = "#1a1a2e",
        font_color       = "#e0e0e0",
        title_text       = "Team Win Rates (2nd innings / run chases)",
        title_font_color = CLR_GOLD,
        height           = 420,
        margin           = dict(l=200, r=100, t=50, b=20),
        xaxis            = dict(title="Win Rate (%)", showgrid=True, gridcolor="#2a2a4e", range=[0, 110]),
        yaxis            = dict(showgrid=False),
    )
    fig.add_vline(x=50, line_dash="dash", line_color="#666")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="header-banner">
        <h1>🏏 IPL Match Intelligence System</h1>
        <p>Real-time win prediction · Player analytics · Insight engine · Match simulator</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load assets ───────────────────────────────────────────────────────────
    with st.spinner("Loading data & model …"):
        df    = load_data(DATA_PATH)
        model = load_model(MODEL_PATH)

    # Friendly error if setup not done
    if df.empty:
        st.error("❌ Dataset not found. Run `python generate_data.py` first.")
        st.code("python generate_data.py\npython train_model.py\nstreamlit run app.py")
        return
    if model is None:
        st.error("❌ Model not found. Run `python train_model.py` first.")
        st.code("python train_model.py")
        return

    player_stats = compute_player_stats(df)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏏 IPL Intelligence")
        st.markdown("---")

        page = st.radio(
            "Navigate to",
            ["🎯 Win Predictor", "📊 Player Stats", "💡 Insight Engine", "🧪 Match Simulator"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Quick Simulator")
        st.caption("Adjust match state:")

        sb_target  = st.number_input("Target",       min_value=50,  max_value=300, value=170, step=1)
        sb_scored  = st.number_input("Runs scored",  min_value=0,   max_value=299, value=80,  step=1)
        sb_wickets = st.slider("Wickets fallen", 0, 9, 3)
        sb_balls   = st.slider("Balls bowled",   1, 119, 60)

        sb_balls_left    = max(0, 120 - sb_balls)
        sb_runs_left     = max(0, sb_target - sb_scored)
        sb_wickets_left  = 10 - sb_wickets
        sb_crr           = round((sb_scored / sb_balls) * 6, 2) if sb_balls > 0 else 0.0
        sb_rrr           = round((sb_runs_left / sb_balls_left) * 6, 2) if sb_balls_left > 0 else 99.0
        sb_prob          = predict_win_probability(model, sb_runs_left, sb_balls_left,
                                                   sb_wickets_left, sb_crr, sb_rrr)

        prob_color = "#2DC653" if sb_prob >= 0.6 else ("#F5A623" if sb_prob >= 0.4 else "#E63946")
        st.markdown(f"""
        <div style="text-align:center; background:#1a1a2e; border-radius:10px; padding:1rem; margin-top:0.5rem;">
            <div style="font-size:0.8rem;color:#888;">WIN PROBABILITY</div>
            <div style="font-size:2.5rem;font-weight:800;color:{prob_color};">{sb_prob*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.caption("Built with ❤️ using Streamlit + scikit-learn")

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE: WIN PREDICTOR
    # ═══════════════════════════════════════════════════════════════════════════
    if page == "🎯 Win Predictor":
        st.markdown('<div class="section-title">🎯 Win Probability Predictor</div>', unsafe_allow_html=True)
        st.caption("Enter the current match state to get an instant prediction.")

        col1, col2, col3 = st.columns(3)
        with col1:
            target   = st.number_input("Target (runs to chase)",  50,  300, 170, key="wp_target")
            scored   = st.number_input("Runs scored so far",       0,  299,  80, key="wp_scored")
        with col2:
            wickets  = st.slider("Wickets fallen", 0, 9, 3, key="wp_wkt")
            balls_b  = st.slider("Balls bowled",   1, 119, 60, key="wp_balls")
        with col3:
            batting_team = st.text_input("Batting team", value="Mumbai Indians", key="wp_team")

        # Derived
        runs_left    = max(0, target - scored)
        balls_left   = max(0, 120 - balls_b)
        wickets_left = 10 - wickets
        crr          = round((scored / balls_b) * 6, 2) if balls_b > 0 else 0.0
        rrr          = round((runs_left / balls_left) * 6, 2) if balls_left > 0 else 99.0

        win_prob = predict_win_probability(model, runs_left, balls_left, wickets_left, crr, rrr)

        st.markdown("---")

        g_col, m_col = st.columns([1.2, 1])
        with g_col:
            st.plotly_chart(plot_win_gauge(win_prob, batting_team),
                            use_container_width=True, config={"displayModeBar": False})
        with m_col:
            st.markdown("#### 📋 Match State")
            mcols = st.columns(2)
            metrics = [
                ("Runs left",      f"{int(runs_left)}"),
                ("Balls left",     f"{int(balls_left)}"),
                ("Wickets left",   f"{int(wickets_left)}"),
                ("Current RR",     f"{crr:.2f}"),
                ("Required RR",    f"{rrr:.2f}"),
                ("Overs done",     f"{balls_b // 6}.{balls_b % 6}"),
            ]
            for i, (lbl, val) in enumerate(metrics):
                with mcols[i % 2]:
                    st.metric(lbl, val)

        # Insights for this state
        st.markdown("#### 💡 Live Insights")
        insights = generate_insights(runs_left, balls_left, wickets_left, crr, rrr, win_prob)
        ins_html = "".join(
            f'<span class="insight-pill insight-{i["level"]}">{i["text"]}</span>'
            for i in insights
        )
        st.markdown(ins_html, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE: PLAYER STATS
    # ═══════════════════════════════════════════════════════════════════════════
    elif page == "📊 Player Stats":
        st.markdown('<div class="section-title">📊 Player Performance Analysis</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard", "🔍 Player Lookup", "📈 Analytics"])

        with tab1:
            top_n = st.slider("Show top N players", 5, 20, 10, key="lb_n")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_top_batsmen(player_stats, top_n),
                                use_container_width=True, config={"displayModeBar": False})
            with c2:
                st.plotly_chart(plot_team_win_rates(df),
                                use_container_width=True, config={"displayModeBar": False})

            st.markdown("#### Full Leaderboard")
            display_cols = ["batsman", "total_runs", "balls_faced",
                            "dismissals", "strike_rate", "average"]
            st.dataframe(
                player_stats[display_cols].head(50)
                    .rename(columns={"batsman":"Player","total_runs":"Runs",
                                     "balls_faced":"Balls","dismissals":"Dismissals",
                                     "strike_rate":"Strike Rate","average":"Average"}),
                use_container_width=True, hide_index=True,
            )

        with tab2:
            all_players = sorted(player_stats["batsman"].tolist())
            selected_player = st.selectbox("Select player", all_players)

            if selected_player:
                row = player_stats[player_stats["batsman"] == selected_player].iloc[0]
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Total Runs",   int(row["total_runs"]))
                p2.metric("Balls Faced",  int(row["balls_faced"]))
                p3.metric("Strike Rate",  f"{row['strike_rate']:.1f}")
                p4.metric("Average",      f"{row['average']:.1f}")

                # Season-wise breakdown for this player
                pdata = df[df["batsman"] == selected_player]
                season_runs = (
                    pdata.groupby("season")["runs_off_bat"].sum().reset_index()
                         .rename(columns={"runs_off_bat": "runs"})
                )
                if not season_runs.empty:
                    fig_s = px.bar(season_runs, x="season", y="runs",
                                   color="runs", color_continuous_scale="Oranges",
                                   title=f"{selected_player} — Runs per Season")
                    fig_s.update_layout(paper_bgcolor="#0f0f23", plot_bgcolor="#1a1a2e",
                                        font_color="#e0e0e0", title_font_color=CLR_GOLD,
                                        coloraxis_showscale=False,
                                        xaxis=dict(showgrid=False, type="category"),
                                        yaxis=dict(gridcolor="#2a2a4e"),
                                        height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_s, use_container_width=True,
                                    config={"displayModeBar": False})

        with tab3:
            st.plotly_chart(plot_strike_rate_scatter(player_stats),
                            use_container_width=True, config={"displayModeBar": False})

            # Compare players
            st.markdown("#### Player Comparison")
            compare_list = st.multiselect(
                "Select 2–5 players to compare",
                options=player_stats["batsman"].head(40).tolist(),
                default=player_stats["batsman"].head(3).tolist(),
                max_selections=5,
            )
            if len(compare_list) >= 2:
                st.plotly_chart(plot_player_comparison(player_stats, compare_list),
                                use_container_width=True, config={"displayModeBar": False})

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE: INSIGHT ENGINE
    # ═══════════════════════════════════════════════════════════════════════════
    elif page == "💡 Insight Engine":
        st.markdown('<div class="section-title">💡 Insight Engine</div>', unsafe_allow_html=True)
        st.caption("Set the match state below and the engine will generate context-aware insights.")

        c1, c2, c3 = st.columns(3)
        with c1:
            ie_target  = st.number_input("Target",        50,  300, 165, key="ie_t")
            ie_scored  = st.number_input("Runs scored",    0,  299,  90, key="ie_s")
        with c2:
            ie_wkt     = st.slider("Wickets fallen", 0, 9, 4, key="ie_w")
            ie_balls   = st.slider("Balls bowled",   1, 119, 72, key="ie_b")
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("Insights update automatically as you adjust the sliders.")

        ie_runs_left   = max(0, ie_target - ie_scored)
        ie_balls_left  = max(0, 120 - ie_balls)
        ie_wkt_left    = 10 - ie_wkt
        ie_crr         = round((ie_scored / ie_balls) * 6, 2) if ie_balls > 0 else 0.0
        ie_rrr         = round((ie_runs_left / ie_balls_left) * 6, 2) if ie_balls_left > 0 else 99.0
        ie_prob        = predict_win_probability(model, ie_runs_left, ie_balls_left,
                                                 ie_wkt_left, ie_crr, ie_rrr)

        insights = generate_insights(ie_runs_left, ie_balls_left, ie_wkt_left,
                                     ie_crr, ie_rrr, ie_prob)

        st.markdown("---")
        st.markdown("### Generated Insights")

        for ins in insights:
            lvl_map = {
                "danger":  ("🔴", "#3d1a1a", "#ff8080", "#E63946"),
                "warning": ("🟡", "#3d2d0a", "#ffd080", "#F5A623"),
                "success": ("🟢", "#0a2d1a", "#80ff80", "#2DC653"),
                "info":    ("🔵", "#0a1a3d", "#a8dadc", "#A8DADC"),
            }
            _, bg, text_c, border_c = lvl_map.get(ins["level"], lvl_map["info"])
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border_c};border-radius:10px;
                        padding:0.75rem 1.2rem;margin:0.4rem 0;color:{text_c};font-size:0.95rem;">
                {ins["text"]}
            </div>""", unsafe_allow_html=True)

        # Rule reference
        with st.expander("📖 Insight Rule Reference"):
            rules = [
                ("Win Probability ≥ 80%",     "Strong favourites",               "success"),
                ("Win Probability 60–79%",     "Comfortable position",            "info"),
                ("Win Probability 40–59%",     "Evenly contested",                "warning"),
                ("Win Probability < 40%",      "Uphill task",                     "danger"),
                ("Required RR > 15",           "Nearly impossible territory",     "danger"),
                ("Required RR > 12",           "Need maximums every over",        "danger"),
                ("Required RR > 10",           "High pressure situation",         "warning"),
                ("Required RR < 6",            "Target well within reach",        "success"),
                ("Wickets left ≤ 2",           "High collapse risk",              "danger"),
                ("Wickets left ≤ 4",           "Lower order in play",             "warning"),
                ("Balls left ≤ 6",             "Final over",                      "warning"),
                ("RRR – CRR > 4",              "Run rate gap dangerously wide",   "danger"),
                ("CRR – RRR > 2",              "Ahead of required pace",          "success"),
            ]
            rdf = pd.DataFrame(rules, columns=["Condition", "Insight", "Level"])
            st.dataframe(rdf, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE: MATCH SIMULATOR
    # ═══════════════════════════════════════════════════════════════════════════
    elif page == "🧪 Match Simulator":
        st.markdown('<div class="section-title">🧪 Interactive Match Simulator</div>', unsafe_allow_html=True)
        st.caption("Simulate different match states and watch how win probability evolves.")

        # Initialise history in session state
        if "sim_history" not in st.session_state:
            st.session_state["sim_history"] = []

        c1, c2 = st.columns([1, 1.6])

        with c1:
            st.markdown("#### ⚙️ Set Match State")
            sim_target  = st.number_input("Target (runs to chase)", 50, 300, 175, key="sim_t")
            sim_scored  = st.number_input("Runs scored",             0, 299,  60, key="sim_s")
            sim_wickets = st.slider("Wickets fallen",        0, 9, 2, key="sim_w")
            sim_balls   = st.slider("Balls bowled (1–119)", 1, 119, 36, key="sim_b")
            sim_team    = st.text_input("Batting team", "Chennai Super Kings", key="sim_team")

            sim_runs_left   = max(0, sim_target - sim_scored)
            sim_balls_left  = max(0, 120 - sim_balls)
            sim_wkt_left    = 10 - sim_wickets
            sim_crr         = round((sim_scored / sim_balls) * 6, 2) if sim_balls > 0 else 0.0
            sim_rrr         = round((sim_runs_left / sim_balls_left) * 6, 2) if sim_balls_left > 0 else 99.0
            sim_prob        = predict_win_probability(model, sim_runs_left, sim_balls_left,
                                                      sim_wkt_left, sim_crr, sim_rrr)

            if st.button("➕ Add to Simulation History", use_container_width=True):
                over_label = f"{sim_balls // 6}.{sim_balls % 6}"
                st.session_state["sim_history"].append((over_label, sim_prob))
                st.success(f"Added: Over {over_label} → {sim_prob*100:.1f}%")

            if st.button("🔄 Clear History", use_container_width=True):
                st.session_state["sim_history"] = []
                st.rerun()

        with c2:
            st.plotly_chart(plot_win_gauge(sim_prob, sim_team),
                            use_container_width=True, config={"displayModeBar": False})

            # Live state summary
            s1, s2, s3 = st.columns(3)
            s1.metric("Runs left",   int(sim_runs_left))
            s2.metric("Balls left",  int(sim_balls_left))
            s3.metric("Wickets left",int(sim_wkt_left))
            s4, s5, _ = st.columns(3)
            s4.metric("Current RR",  f"{sim_crr:.2f}")
            s5.metric("Required RR", f"{sim_rrr:.2f}")

        # Probability history chart
        if st.session_state["sim_history"]:
            st.markdown("---")
            st.markdown("#### 📈 Win Probability History")
            st.plotly_chart(
                plot_prob_history(st.session_state["sim_history"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            hist_df = pd.DataFrame(
                st.session_state["sim_history"],
                columns=["Over", "Win Probability"],
            )
            hist_df["Win Probability"] = (hist_df["Win Probability"] * 100).round(1).astype(str) + "%"
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
        else:
            st.info("💡 Adjust the match state above and click **Add to Simulation History** to track how win probability changes over the course of the match.")

        # Insights for current simulator state
        st.markdown("---")
        st.markdown("#### 💡 Current Situation Insights")
        sim_insights = generate_insights(sim_runs_left, sim_balls_left, sim_wkt_left,
                                          sim_crr, sim_rrr, sim_prob)
        for ins in sim_insights[:4]:   # top 4 most relevant
            lvl_map = {"danger": "#E63946", "warning": "#F5A623",
                       "success": "#2DC653", "info": "#A8DADC"}
            border_c = lvl_map.get(ins["level"], "#A8DADC")
            bg_map   = {"danger": "#3d1a1a", "warning": "#3d2d0a",
                        "success": "#0a2d1a", "info": "#0a1a3d"}
            bg_c    = bg_map.get(ins["level"], "#0a1a3d")
            tc_map  = {"danger": "#ff8080", "warning": "#ffd080",
                       "success": "#80ff80", "info": "#a8dadc"}
            tc      = tc_map.get(ins["level"], "#a8dadc")
            st.markdown(f"""
            <div style="background:{bg_c};border:1px solid {border_c};border-radius:8px;
                        padding:0.6rem 1rem;margin:0.3rem 0;color:{tc};font-size:0.9rem;">
                {ins["text"]}
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
