import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, precision_score, recall_score, confusion_matrix

st.set_page_config(page_title="Financial Crisis Dashboard", layout="wide")
st.title("📉 Financial Crisis Early Warning System — USA, UK, Canada")


# =============================================================================
# 1. LOAD DATA
# =============================================================================
@st.cache_data
def load_data(path="JSTdatasetR6.xlsx - Sheet1.csv"):
    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_excel("JSTdatasetR6.xlsx")

    df = df[df["country"].isin(["USA", "UK", "Canada"])].copy()
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    return df


# =============================================================================
# 2. FEATURE ENGINEERING (Leakage-Free)
# =============================================================================
def add_base_features(df):

    df = df.copy()

    def z(x): 
        return (x - x.mean()) / (x.std() + 1e-8)

    if "lev" in df.columns:
        df["leverage_risk"] = 1 / (df["lev"] + 0.01)

    if all(c in df.columns for c in ["noncore", "ltd", "lev"]):
        df["banking_fragility_index"] = (
            0.4 * z(df["noncore"].fillna(0)) +
            0.3 * z(df["ltd"].fillna(0)) +
            0.3 * z(df["leverage_risk"].fillna(0))
        )

    if "hpnom" in df.columns and "cpi" in df.columns:
        df["hp_real"] = df["hpnom"] / df["cpi"]

    if "tloans" in df.columns and "cpi" in df.columns:
        df["real_credit"] = df["tloans"] / df["cpi"]

    if "money" in df.columns and "gdp" in df.columns:
        df["money_gdp_ratio"] = df["money"] / df["gdp"]

    if "ca" in df.columns and "gdp" in df.columns:
        df["ca_gdp_ratio"] = df["ca"] / df["gdp"]

    if "ltrate" in df.columns:
        us_curve = df[df["country"] == "USA"][["year", "ltrate"]].drop_duplicates("year")
        us_curve = us_curve.set_index("year")["ltrate"].to_dict()
        df["us_ltrate"] = df["year"].map(us_curve)
        df["sovereign_spread"] = df["ltrate"] - df["us_ltrate"]

    if "stir" in df.columns and "ltrate" in df.columns:
        df["yield_curve_slope"] = df["ltrate"] - df["stir"]

    return df


def add_rolling_features(df):

    df = df.copy()
    df["credit_growth"] = df.groupby("country")["real_credit"].pct_change()

    df["hp_trend"] = df.groupby("country")["hp_real"].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )
    df["housing_bubble"] = (df["hp_real"] - df["hp_trend"]) / df["hp_trend"]

    df["money_expansion"] = df.groupby("country")["money_gdp_ratio"].pct_change()

    return df


def make_target(df, horizon=2):
    df = df.copy()
    df["target"] = df.groupby("country")["crisisJST"].shift(-horizon)
    return df.dropna(subset=["target"])


# =============================================================================
# 3. MODELING UTILITIES
# =============================================================================
def prepare_xy(train, test, feature_cols):

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()

    X_train = imp.fit_transform(train[feature_cols])
    X_train = scl.fit_transform(X_train)
    y_train = train["target"].astype(int)

    X_test = imp.transform(test[feature_cols])
    X_test = scl.transform(X_test)
    y_test = test["target"].astype(int)

    return X_train, y_train, X_test, y_test, imp, scl


def train_model(X, y):

    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000
    )
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]

    P, R, T = precision_recall_curve(y, probs)
    f2 = (5 * P * R) / (4 * P + R + 1e-9)

    best_t = T[np.argmax(f2)]

    return model, best_t


def predict(model, X, threshold):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds, probs


# =============================================================================
# 4. SHAP PLOT WRAPPER FOR STREAMLIT (NO initjs)
# =============================================================================
def st_shap(plot, height=400):
    """Render SHAP force plots in Streamlit."""
    import streamlit.components.v1 as components
    html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(html, height=height)


# =============================================================================
# 5. PIPELINE + UI
# =============================================================================
df = load_data()

train_raw = df[df["year"] < 1990].copy()
test_raw  = df[df["year"] >= 1990].copy()

train_base = add_base_features(train_raw)
test_base  = add_base_features(test_raw)

train_feat = add_rolling_features(train_base)
test_feat  = add_rolling_features(test_base)

train = make_target(train_feat)
test  = make_target(test_feat)

drop_cols = ["country", "year", "crisisJST", "target", "hp_trend"]
feature_cols = [
    c for c in train.columns
    if c not in drop_cols and pd.api.types.is_numeric_dtype(train[c])
]

X_train, y_train, X_test, y_test, imp, scl = prepare_xy(train, test, feature_cols)

model, threshold = train_model(X_train, y_train)
preds, probs = predict(model, X_test, threshold)


# =============================================================================
# DISPLAY METRICS
# =============================================================================
st.subheader("📊 Model Performance (Test Set)")

col1, col2, col3 = st.columns(3)
col1.metric("ROC-AUC", f"{roc_auc_score(y_test, probs):.3f}")
col2.metric("Precision", f"{precision_score(y_test, preds):.3f}")
col3.metric("Recall", f"{recall_score(y_test, preds):.3f}")

st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, preds))


# =============================================================================
# COUNTRY RISK PLOTS
# =============================================================================
st.subheader("📉 Country Crisis Risk Over Time")

for country in ["USA", "UK", "Canada"]:

    cdf = test[test["country"] == country].copy()
    cdf["risk"] = probs[test["country"] == country]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(cdf["year"], cdf["risk"], label="Predicted Risk", linewidth=2)

    for yr in cdf[cdf["target"] == 1]["year"]:
        ax.axvspan(yr, yr+1, color="red", alpha=0.2)

    ax.set_title(f"{country} Crisis Risk")
    ax.set_ylabel("Predicted Probability")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)


# =============================================================================
# SHAP EXPLAINABILITY
# =============================================================================
st.subheader("🧠 SHAP Explainability")

explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

st.write("### SHAP Summary Plot")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
st.pyplot(fig)

st.write("### Individual Prediction Explanation")

i = st.slider("Select Test Observation", 0, len(X_test)-1, 0)
force_plot = shap.ForcePlot(
    explainer.expected_value,
    shap_values[i],
    feature_names=feature_cols
)
st_shap(force_plot, height=300)
