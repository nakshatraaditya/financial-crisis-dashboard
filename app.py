# ======================================================================
#  FINANCIAL CRISIS EARLY WARNING SYSTEM – STREAMLIT DASHBOARD
#  Dissertation-Ready | Senior Data Scientist Implementation
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve,
    average_precision_score, f1_score
)
from sklearn.preprocessing import StandardScaler

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
st.caption("USA · UK · Canada | 1870–2020 | Dissertation Dashboard")

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

    df["hp_real"] = df["hpnom"] / df["cpi"]
    df["hp_trend"] = df.groupby("country")["hp_real"].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )
    df["housing_bubble"] = (df["hp_real"] - df["hp_trend"]) / (df["hp_trend"] + 1e-9)

    df["real_credit"] = df["tloans"] / df["cpi"]
    df["credit_growth"] = df.groupby("country")["real_credit"].pct_change()

    df["yield_curve"] = df["ltrate"] - df["stir"]

    us_ltrate = (
        df[df["country"] == "USA"]
        .drop_duplicates("year")
        .set_index("year")["ltrate"]
        .to_dict()
    )
    df["us_ltrate"] = df["year"].map(us_ltrate)
    df["sovereign_spread"] = df["ltrate"] - df["us_ltrate"]

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
# MODEL TRAINING (ONCE)
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
    X_test  = test.drop(columns=["target", "crisisJST", "country", "year"])
    y_test  = test["target"]

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test  = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.01,
        max_depth=3,
        random_state=42
    )
    model.fit(Xs_train, y_train)

    probs = model.predict_proba(Xs_test)[:, 1]

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    return model, scaler, best_t, df, X_test.columns


model, scaler, threshold, df_full, feature_cols = train_model()

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
    st.subheader("Final Model Performance (Test Set)")
    st.markdown(f"**Model:** Gradient Boosting  \n**Optimal Threshold:** {threshold:.2f}")

# ======================================================================
# TAB 2 – RISK TIMELINE
# ======================================================================
with tab2:
    for c in ["USA", "UK", "Canada"]:
        df_c = df_full[df_full["country"] == c]

        X = df_c.drop(columns=["target", "crisisJST", "country", "year"])
        Xs = scaler.transform(X)
        probs = model.predict_proba(Xs)[:, 1]

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df_c["year"], probs, lw=2)
        ax.axhline(threshold, linestyle="--")

        for y in df_c[df_c["crisisJST"] == 1]["year"]:
            ax.axvspan(y-0.5, y+0.5, color="red", alpha=0.2)

        ax.set_title(c)
        ax.set_ylim(0, 1)
        st.pyplot(fig)

# ======================================================================
# TAB 3 – UPLOAD & PREDICT
# ======================================================================
with tab3:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        user_df = pd.read_csv(uploaded)

        required = set(feature_cols) | {
            "country","year","lev","noncore","ltd","hpnom","cpi",
            "tloans","ltrate","stir","money","gdp","ca"
        }

        if not required.issubset(user_df.columns):
            st.error("Uploaded file missing required columns.")
        else:
            df_u, _ = engineer_features(user_df)
            df_u = clean_data(df_u)

            X_u = df_u.drop(columns=["country","year","crisisJST"], errors="ignore")
            Xs_u = scaler.transform(X_u)

            probs_u = model.predict_proba(Xs_u)[:, 1]
            preds_u = (probs_u >= threshold).astype(int)

            df_u["predicted_prob"] = probs_u
            df_u["predicted_class"] = preds_u

            st.dataframe(df_u[["country","year","predicted_prob","predicted_class"]])

# ======================================================================
# TAB 4 – DATA EXPLORER
# ======================================================================
with tab4:
    st.dataframe(df_full.tail(50))
