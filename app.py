# ======================================================================
#  FINANCIAL CRISIS EARLY WARNING SYSTEM – STREAMLIT DASHBOARD
#  Interactive Plotly Version | Dissertation-Ready
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score,
    average_precision_score
)

import warnings
warnings.filterwarnings("ignore")

# ======================================================================
# STREAMLIT CONFIG
# ======================================================================
st.set_page_config(
    page_title="Financial Crisis Early Warning System",
    layout="wide"
)

st.title("📉 Financial Crisis Early Warning System")
st.caption("USA · UK · Canada | 1870–2020 | Interactive Dissertation Dashboard")

# ======================================================================
# SIDEBAR CONTROLS
# ======================================================================
st.sidebar.header("Dashboard Controls")

selected_countries = st.sidebar.multiselect(
    "Select Countries",
    ["USA", "UK", "Canada"],
    default=["USA", "UK", "Canada"]
)

risk_threshold = st.sidebar.slider(
    "Risk Threshold",
    0.05, 0.9, 0.20, 0.01
)

show_crises = st.sidebar.checkbox(
    "Show Crisis Periods",
    value=True
)

# ======================================================================
# DATA PIPELINE
# ======================================================================

@st.cache_data
def load_data(file="JSTdatasetR6.xlsx"):
    df = pd.read_excel(file)
    df = df[df["country"].isin(["USA", "UK", "Canada"])].copy()
    df = df.sort_values(["country", "year"])
    return df


def engineer_features(df):
    df = df.copy()

    # --- Banking fragility ---
    df["leverage_risk"] = 1 / (df["lev"] + 0.01)

    def expanding_z(df_inner, col):
        def z(s):
            m = s.expanding().mean()
            sd = s.expanding().std().replace(0, np.nan)
            return (s - m) / (sd + 1e-9)
        return df_inner.groupby("country")[col].transform(z)

    df["noncore_z"] = expanding_z(df, "noncore")
    df["ltd_z"] = expanding_z(df, "ltd")
    df["leverage_z"] = expanding_z(df, "leverage_risk")

    df["banking_fragility"] = (
        0.4 * df["noncore_z"] +
        0.3 * df["ltd_z"] +
        0.3 * df["leverage_z"]
    )

    # --- Housing bubble ---
    df["hp_real"] = df["hpnom"] / df["cpi"]
    df["hp_trend"] = df.groupby("country")["hp_real"].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )
    df["housing_bubble"] = (df["hp_real"] - df["hp_trend"]) / (df["hp_trend"] + 1e-9)

    # --- Credit ---
    df["real_credit"] = df["tloans"] / df["cpi"]
    df["credit_growth"] = df.groupby("country")["real_credit"].pct_change()

    # --- Yield curve ---
    df["yield_curve"] = df["ltrate"] - df["stir"]

    # --- Sovereign spread vs USA ---
    us_ltrate = (
        df[df["country"] == "USA"]
        .drop_duplicates("year")
        .set_index("year")["ltrate"]
        .to_dict()
    )
    df["us_ltrate"] = df["year"].map(us_ltrate)
    df["sovereign_spread"] = df["ltrate"] - df["us_ltrate"]

    # --- Monetary & external ---
    df["money_gdp"] = df["money"] / df["gdp"]
    df["money_expansion"] = df.groupby("country")["money_gdp"].pct_change()
    df["ca_gdp"] = df["ca"] / df["gdp"]

    features = [
        "housing_bubble", "credit_growth", "banking_fragility",
        "sovereign_spread", "yield_curve",
        "money_expansion", "ca_gdp"
    ]

    df = df[["country", "year", "crisisJST"] + features]
    df = df.replace([np.inf, -np.inf], np.nan)

    return df, features


def clean_data(df):
    df = df.copy()

    # Remove wartime distortions
    df = df[~df["year"].between(1914, 1918)]
    df = df[~df["year"].between(1939, 1945)]

    features = [c for c in df.columns if c not in ["country", "year", "crisisJST"]]

    df[features] = df.groupby("country")[features].transform(
        lambda x: x.ffill(limit=3).bfill(limit=3)
    )
    df[features] = df.groupby("country")[features].transform(
        lambda x: x.fillna(x.median())
    )

    return df


def create_target(df):
    df = df.copy()

    # Crisis in t+1 to t+2 window
    df["target"] = (
        df.groupby("country")["crisisJST"]
        .shift(-1)
        .rolling(2, min_periods=1)
        .max()
    )

    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    return df

# ======================================================================
# MODEL TRAINING (CACHED)
# ======================================================================

@st.cache_data
def train_model():
    df = load_data()
    df, features = engineer_features(df)
    df = clean_data(df)
    df = create_target(df)

    train = df[df["year"] < 1970]
    test  = df[df["year"] >= 1990]

    X_train = train.drop(columns=["target", "crisisJST", "country", "year"])
    y_train = train["target"]

    X_test = test.drop(columns=["target", "crisisJST", "country", "year"])
    y_test = test["target"]

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.01,
        max_depth=3,
        random_state=42
    )

    model.fit(Xs_train, y_train)

    return model, scaler, df, Xs_test, y_test, X_test.columns


model, scaler, df_full, Xs_test, y_test, feature_cols = train_model()

# ======================================================================
# TABS
# ======================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Results",
    "📈 Risk Timeline",
    "📥 Upload & Predict",
    "📂 Data Explorer"
])

# ======================================================================
# TAB 1 – MODEL RESULTS
# ======================================================================
with tab1:
    st.subheader("Out-of-Sample Performance (Post-1990)")

    probs = model.predict_proba(Xs_test)[:, 1]
    preds = (probs >= risk_threshold).astype(int)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("ROC-AUC", f"{roc_auc_score(y_test, probs):.3f}")
    c2.metric("PR-AUC", f"{average_precision_score(y_test, probs):.3f}")
    c3.metric("Recall", f"{recall_score(y_test, preds):.3f}")
    c4.metric("F1 Score", f"{f1_score(y_test, preds):.3f}")

# ======================================================================
# TAB 2 – INTERACTIVE RISK TIMELINE
# ======================================================================
with tab2:
    st.subheader("Interactive Crisis Risk Timeline")

    for c in selected_countries:
        df_c = df_full[df_full["country"] == c].copy()

        X = df_c.drop(columns=["target", "crisisJST", "country", "year"])
        Xs = scaler.transform(X)
        df_c["crisis_prob"] = model.predict_proba(Xs)[:, 1]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_c["year"],
                y=df_c["crisis_prob"],
                mode="lines",
                name="Predicted Crisis Probability",
                hovertemplate="Year: %{x}<br>Risk: %{y:.2f}<extra></extra>"
            )
        )

        fig.add_hline(
            y=risk_threshold,
            line_dash="dash",
            line_color="black",
            annotation_text="Risk Threshold"
        )

        if show_crises:
            for y in df_c[df_c["crisisJST"] == 1]["year"]:
                fig.add_vrect(
                    x0=y - 0.5,
                    x1=y + 0.5,
                    fillcolor="red",
                    opacity=0.25,
                    line_width=0
                )

        fig.update_layout(
            title=f"{c} – Crisis Risk Timeline",
            xaxis_title="Year",
            yaxis_title="Crisis Probability",
            yaxis=dict(range=[0, 1]),
            hovermode="x unified",
            height=350,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# TAB 3 – UPLOAD & PREDICT
# ======================================================================
with tab3:
    st.subheader("Upload Data for Prediction")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        user_df = pd.read_csv(uploaded)

        required_cols = set(feature_cols) | {
            "country","year","lev","noncore","ltd","hpnom","cpi",
            "tloans","ltrate","stir","money","gdp","ca"
        }

        if not required_cols.issubset(user_df.columns):
            st.error("Uploaded file is missing required columns.")
        else:
            df_u, _ = engineer_features(user_df)
            df_u = clean_data(df_u)

            X_u = df_u.drop(columns=["country","year","crisisJST"], errors="ignore")
            Xs_u = scaler.transform(X_u)

            df_u["predicted_prob"] = model.predict_proba(Xs_u)[:, 1]
            df_u["predicted_class"] = (df_u["predicted_prob"] >= risk_threshold).astype(int)

            st.success(
                f"{df_u['predicted_class'].sum()} high-risk observations detected "
                f"({100 * df_u['predicted_class'].mean():.1f}%)"
            )

            st.dataframe(
                df_u[["country","year","predicted_prob","predicted_class"]],
                use_container_width=True
            )

# ======================================================================
# TAB 4 – DATA EXPLORER
# ======================================================================
with tab4:
    st.subheader("Processed Dataset Preview")
    st.dataframe(df_full.tail(50), use_container_width=True)
