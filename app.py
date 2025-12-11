# ======================================================================
#  FINANCIAL CRISIS EARLY WARNING SYSTEM – STREAMLIT DASHBOARD (TABS + SHAP)
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve,
    average_precision_score, f1_score
)
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Financial Crisis Early Warning System",
                   layout="wide")

st.title("📉 Financial Crisis Early Warning Dashboard")
st.markdown("An interactive early-warning system for USA, UK, and Canada using machine learning.")

# ======================================================================
# 1. LOAD DATA
# ======================================================================
@st.cache_data
def load_data(file="JSTdatasetR6.xlsx"):
    df = pd.read_excel(file)
    df = df[df["country"].isin(["USA", "UK", "Canada"])].copy()
    df = df.sort_values(["country", "year"])
    return df

# ======================================================================
# 2. FEATURE ENGINEERING
# ======================================================================
def engineer_features(df):
    df = df.copy()
    df["leverage_risk"] = 1 / (df["lev"] + 0.01)

    def expanding_z(df_inner, col):
        def _z(series):
            m = series.expanding().mean()
            sd = series.expanding().std().replace(0, np.nan)
            return (series - m) / (sd + 1e-9)
        return df_inner.groupby("country")[col].transform(_z)

    df["noncore_z"] = expanding_z(df, "noncore")
    df["ltd_z"] = expanding_z(df, "ltd")
    df["leverage_z"] = expanding_z(df, "leverage_risk")
    df["banking_fragility"] = (
        0.4 * df["noncore_z"] +
        0.3 * df["ltd_z"] +
        0.3 * df["leverage_z"]
    )

    df["hp_real"] = df["hpnom"] / df["cpi"]
    df["hp_trend"] = df.groupby("country")["hp_real"].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )
    df["housing_bubble"] = (df["hp_real"] - df["hp_trend"]) / (df["hp_trend"] + 1e-9)

    df["real_credit"] = df["tloans"] / df["cpi"]
    df["credit_growth"] = df.groupby("country")["real_credit"].pct_change()

    df["yield_curve"] = df["ltrate"] - df["stir"]

    us_ltrate_map = (
        df[df["country"] == "USA"]
        .drop_duplicates("year")
        .set_index("year")["ltrate"].to_dict()
    )
    df["us_ltrate"] = df["year"].map(us_ltrate_map)
    df["sovereign_spread"] = df["ltrate"] - df["us_ltrate"]

    df["money_gdp"] = df["money"] / df["gdp"]
    df["money_expansion"] = df.groupby("country")["money_gdp"].pct_change()

    df["ca_gdp"] = df["ca"] / df["gdp"]

    features = [
        "housing_bubble", "credit_growth", "banking_fragility",
        "sovereign_spread", "yield_curve",
        "money_expansion", "ca_gdp",
    ]

    df = df[["country", "year", "crisisJST"] + features]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df, features

# ======================================================================
# 3. CLEAN DATA (Causal Imputation)
# ======================================================================
def clean_data(df):
    df = df.copy()
    df = df[~df["year"].between(1914, 1918)]
    df = df[~df["year"].between(1939, 1945)]
    features = [c for c in df.columns if c not in ["country", "year", "crisisJST"]]

    for col in features:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    df[features] = df.groupby("country")[features].transform(
        lambda s: s.ffill(limit=3).bfill(limit=3)
    )
    df[features] = df.groupby("country")[features].transform(
        lambda s: s.fillna(s.median())
    )
    return df

# ======================================================================
# 4. TARGET VARIABLE
# ======================================================================
def create_target(df):
    df = df.copy()
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
# 5. MODEL TRAINING
# ======================================================================
def train_model(train_df, features):
    X = train_df[features]
    y = train_df["target"]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    weights = np.where(y == 1, 12, 1)

    clf = LogisticRegression(max_iter=5000, penalty="l2")
    clf.fit(Xs, y, sample_weight=weights)

    return clf, scaler

# ======================================================================
# PROCESS DATA
# ======================================================================
df_raw = load_data()
df_feat, features = engineer_features(df_raw)
df_clean = clean_data(df_feat)
df_target = create_target(df_clean)

train = df_target[df_target["year"] < 1970]
val   = df_target[(df_target["year"] >= 1970) & (df_target["year"] < 1990)]
test  = df_target[df_target["year"] >= 1990]

model, scaler = train_model(train, features)

# Threshold tuning
Xs_val = scaler.transform(val[features])
val_probs = model.predict_proba(Xs_val)[:, 1]

best_f1 = 0
best_t = 0.5
for t in np.arange(0.1, 0.9, 0.01):
    preds = (val_probs >= t).astype(int)
    f1 = f1_score(val["target"], preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

threshold = best_t

# ======================================================================
# CREATE TABS
# ======================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Results",
    "🔍 SHAP Explainability",
    "📈 Risk Timeline",
    "📂 Data Explorer"
])

# ======================================================================
# TAB 1 — MODEL RESULTS
# ======================================================================
with tab1:
    st.header("📊 Model Performance on Test Set (1990–2020)")

    test_X = scaler.transform(test[features])
    test_probs = model.predict_proba(test_X)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    roc = roc_auc_score(test["target"], test_probs)
    pr = average_precision_score(test["target"], test_probs)
    precision = precision_score(test["target"], test_preds)
    recall = recall_score(test["target"], test_preds)
    f1 = f1_score(test["target"], test_preds)

    st.write(pd.DataFrame({
        "ROC-AUC": [roc],
        "PR-AUC": [pr],
        "Precision": [precision],
        "Recall": [recall],
        "Threshold": [threshold]
    }))

    # PR Curve
    st.subheader("Precision–Recall Curve")
    prec_curve, rec_curve, _ = precision_recall_curve(test["target"], test_probs)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rec_curve, prec_curve)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ======================================================================
# TAB 2 — SHAP Explainability
# ======================================================================
with tab2:
    st.header("🔍 SHAP Feature Importance")

    shap.initjs()
    X_train_scaled = scaler.transform(train[features])
    X_test_scaled = scaler.transform(test[features])

    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

    st.subheader("Global Feature Impact (Summary Plot)")
    shap_fig = shap.summary_plot(shap_values, test[features], show=False)
    st.pyplot(bbox_inches="tight")

# ======================================================================
# TAB 3 — RISK TIMELINE
# ======================================================================
with tab3:
    st.header("📈 Crisis Risk Timeline")

    country = st.selectbox("Select country", ["USA", "UK", "Canada"])
    df_c = test[test["country"] == country].copy()
    df_c["risk"] = test_probs[test["country"] == country]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_c["year"], df_c["risk"], color="purple")
    ax.axhline(threshold, color="orange", linestyle="--")

    for y in df_c[df_c["crisisJST"] == 1]["year"]:
        ax.axvspan(y - 0.5, y + 0.5, color="red", alpha=0.3)

    ax.set_title(f"{country} — Crisis Risk Timeline")
    st.pyplot(fig)

    st.subheader("🇺🇸🇬🇧🇨🇦 Country Comparison")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for c in ["USA", "UK", "Canada"]:
        df_temp = test[test["country"] == c]
        ax2.plot(df_temp["year"], test_probs[df_temp.index], label=c)

    ax2.legend()
    ax2.set_title("Risk Comparison Across Countries")
    st.pyplot(fig2)

# ======================================================================
# TAB 4 — DATA EXPLORER
# ======================================================================
with tab4:
    st.header("📂 Data Explorer")

    selected_country = st.selectbox("Filter by Country", ["USA", "UK", "Canada", "All"])

    df_exp = test.copy()
    df_exp["predicted_prob"] = test_probs
    df_exp["predicted_class"] = test_preds

    if selected_country != "All":
        df_exp = df_exp[df_exp["country"] == selected_country]

    st.dataframe(df_exp)

    st.download_button(
        "Download Data as CSV",
        df_exp.to_csv(index=False),
        "crisis_predictions.csv",
        "text/csv"
    )
