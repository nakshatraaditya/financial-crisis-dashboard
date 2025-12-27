import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Page config
st.set_page_config(
    page_title="Financial Crisis Dashboard",
    page_icon="📉",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Title
'''
# 📉 Financial Crisis Early Warning Dashboard

Interactive exploration of **crisis risk probabilities** for  
**USA · UK · Canada (1870–2020)** based on macro-financial indicators.

This dashboard follows a **policy-style early-warning framework** and is
designed for academic and dissertation use.
'''

''
''

# -----------------------------------------------------------------------------
# Data loading

@st.cache_data
def load_data():
    df = pd.read_excel("JSTdatasetR6.xlsx")
    df = df[df["country"].isin(["USA", "UK", "Canada"])].copy()
    df = df.sort_values(["country", "year"])
    return df


@st.cache_data
def prepare_data():
    df = load_data()

    # --- Feature engineering ---
    df["leverage_risk"] = 1 / (df["lev"] + 0.01)

    def expanding_z(col):
        return df.groupby("country")[col].transform(
            lambda s: (s - s.expanding().mean()) /
                      (s.expanding().std().replace(0, np.nan) + 1e-9)
        )

    df["banking_fragility"] = (
        0.4 * expanding_z("noncore") +
        0.3 * expanding_z("ltd") +
        0.3 * expanding_z("leverage_risk")
    )

    df["hp_real"] = df["hpnom"] / df["cpi"]
    df["hp_trend"] = df.groupby("country")["hp_real"].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )
    df["housing_bubble"] = (df["hp_real"] - df["hp_trend"]) / (df["hp_trend"] + 1e-9)

    df["real_credit"] = df["tloans"] / df["cpi"]
    df["credit_growth"] = df.groupby("country")["real_credit"].pct_change()

    df["yield_curve"] = df["ltrate"] - df["stir"]
    df["money_gdp"] = df["money"] / df["gdp"]
    df["money_expansion"] = df.groupby("country")["money_gdp"].pct_change()
    df["ca_gdp"] = df["ca"] / df["gdp"]

    features = [
        "housing_bubble",
        "credit_growth",
        "banking_fragility",
        "yield_curve",
        "money_expansion",
        "ca_gdp"
    ]

    df = df[["country", "year", "crisisJST"] + features]
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove wars
    df = df[~df["year"].between(1914, 1918)]
    df = df[~df["year"].between(1939, 1945)]

    df[features] = df.groupby("country")[features].transform(
        lambda x: x.ffill(limit=3).bfill(limit=3).fillna(x.median())
    )

    # Target: crisis in t+1 or t+2
    df["target"] = (
        df.groupby("country")["crisisJST"]
        .shift(-1)
        .rolling(2, min_periods=1)
        .max()
    )
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    return df, features


@st.cache_data
def train_model():
    df, features = prepare_data()

    train = df[df["year"] < 1970]
    test = df[df["year"] >= 1990]

    X_train = train[features]
    y_train = train["target"]
    X_test = test[features]
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

    test_probs = model.predict_proba(Xs_test)[:, 1]

    return model, scaler, df, features, y_test, test_probs


model, scaler, df_full, FEATURES, y_test, test_probs = train_model()

# -----------------------------------------------------------------------------
# Sidebar controls

min_year = int(df_full["year"].min())
max_year = int(df_full["year"].max())

from_year, to_year = st.slider(
    "Which years are you interested in?",
    min_value=min_year,
    max_value=max_year,
    value=(1900, max_year)
)

countries = sorted(df_full["country"].unique())

selected_countries = st.multiselect(
    "Which countries would you like to view?",
    countries,
    default=countries
)

threshold = st.slider(
    "Risk threshold",
    0.05, 0.9, 0.20, 0.01
)

''
''

# -----------------------------------------------------------------------------
# Filter data
filtered_df = df_full[
    (df_full["country"].isin(selected_countries)) &
    (df_full["year"] >= from_year) &
    (df_full["year"] <= to_year)
].copy()

Xs = scaler.transform(filtered_df[FEATURES])
filtered_df["crisis_prob"] = model.predict_proba(Xs)[:, 1]

# -----------------------------------------------------------------------------
# Line chart (Streamlit-native)

st.header("Crisis risk over time", divider="gray")

st.line_chart(
    filtered_df,
    x="year",
    y="crisis_prob",
    color="country",
)

''
''

# -----------------------------------------------------------------------------
# Metrics section (GDP-style)

st.header(f"Crisis risk summary ({to_year})", divider="gray")

cols = st.columns(3)

latest = filtered_df[filtered_df["year"] == to_year]

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]

    with col:
        val = latest[latest["country"] == country]["crisis_prob"].mean()

        if math.isnan(val):
            st.metric(country, "n/a")
        else:
            st.metric(
                label=f"{country} risk",
                value=f"{val:.2f}",
                delta="HIGH" if val >= threshold else "LOW",
                delta_color="inverse"
            )

''
''

# -----------------------------------------------------------------------------
# Model performance (simple, examiner-friendly)

st.header("Model performance (post-1990)", divider="gray")

c1, c2, c3 = st.columns(3)

c1.metric("ROC-AUC", f"{roc_auc_score(y_test, test_probs):.3f}")
c2.metric("Recall", f"{recall_score(y_test, test_probs >= threshold):.3f}")
c3.metric("F1 score", f"{f1_score(y_test, test_probs >= threshold):.3f}")

''
''

st.caption(
    "⚠️ Crisis periods shaded implicitly through probability spikes. "
    "This tool is designed for **early-warning interpretation**, not point forecasting."
)
