# ======================================================================
#  FINANCIAL CRISIS EARLY WARNING SYSTEM – STREAMLIT DASHBOARD
#  (FULL UPDATED VERSION)
#
#  What’s new (your request):
#   ✅ If you upload a country (e.g., Spain), Tab 1 will switch to show the
#      uploaded country predictions instead of USA/UK/Canada.
#   ✅ You can toggle Tab 1 data source in the sidebar:
#      - Training (USA/UK/Canada)
#      - Uploaded (your CSV)
#   ✅ Tab 1 charts + summary automatically adapt to uploaded country + years
#   ✅ Uploaded macro charts (GDP + real house prices) use your uploaded CSV
#      (gdp, hpnom, cpi), so they show for Spain too.
#
#  Spain upload crash fix kept:
#   ✅ No NaN/Inf passed into model in Upload & Predict
#   ✅ Sovereign spread uses USA long-rate mapping from JST training data
#   ✅ Remaining NaNs filled using training medians (fallback 0)
#
#  Other features preserved:
#   ✅ Multi-model training + best model selection (validation)
#   ✅ Threshold slider
#   ✅ RGB-consistent colors for country charts (Altair) + deterministic colors for new countries
#   ✅ Risk summary as PERCENTAGE
#   ✅ SHAP interactive pie (Plotly if installed) + explanation under it
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt

# Optional Plotly (interactive pie). If not installed, app still runs.
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Altair for RGB-controlled charts
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from sklearn.preprocessing import StandardScaler

import shap
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# STREAMLIT CONFIG
st.set_page_config(
    page_title="Financial Crisis EWS Dashboard",
    page_icon="📉",
    layout="wide"
)

APP_DIR = Path(__file__).parent
JST_XLSX = APP_DIR / "JSTdatasetR6.xlsx"

# -----------------------------------------------------------------------------
# Base RGB map (you can edit these)
BASE_COUNTRY_ORDER = ["USA", "UK", "Canada"]
BASE_COUNTRY_RGB = {
    "USA": "rgb(255, 99, 71)",      # Tomato
    "UK": "rgb(100, 149, 237)",     # Cornflower Blue
    "Canada": "rgb(60, 179, 113)",  # Medium Sea Green
}

# -----------------------------------------------------------------------------
# HEADER
"""
# 📉 Financial Crisis Early Warning System (EWS)

This dashboard implements your **multi-model dissertation pipeline**:
- Missing-value flags + causal imputation  
- Proper scaling (continuous only)  
- Validation-based model selection + threshold optimization  
- SHAP explainability  
- Crisis risk timeline + macro context (GDP + Real House Prices)  
- Upload & Predict in the **sidebar**
"""
st.write("")

# -----------------------------------------------------------------------------
# Cache reset
with st.sidebar:
    if st.button("Clear cache & rerun"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# ======================================================================
# Helpers
# ======================================================================

def stable_rgb(name: str) -> str:
    """Deterministic RGB for any unseen country label (so Spain isn’t grey)."""
    h = abs(hash(name))
    r = 60 + (h % 140)
    g = 60 + ((h // 140) % 140)
    b = 60 + ((h // (140 * 140)) % 140)
    return f"rgb({r},{g},{b})"

def build_color_scale(categories):
    """Return (domain, range_colors) using base RGB + deterministic for new countries."""
    cats = [str(c) for c in categories]
    domain = []
    # keep base order first if present
    for c in BASE_COUNTRY_ORDER:
        if c in cats and c not in domain:
            domain.append(c)
    # then others sorted
    for c in sorted(set(cats)):
        if c not in domain:
            domain.append(c)

    range_colors = [BASE_COUNTRY_RGB.get(c, stable_rgb(c)) for c in domain]
    return domain, range_colors

def altair_line_chart(df: pd.DataFrame, x_col: str, y_col: str, color_col: str, title: str = ""):
    """Interactive line with fixed RGB colors (Altair). Falls back to st.line_chart if Altair missing."""
    if df is None or df.empty:
        st.info("No data to plot.")
        return

    if not ALTAIR_OK:
        st.line_chart(df, x=x_col, y=y_col, color=color_col)
        return

    domain, range_colors = build_color_scale(df[color_col].unique())

    chart = (
        alt.Chart(df)
        .mark_line(point=False)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_col),
            y=alt.Y(f"{y_col}:Q", title=y_col),
            color=alt.Color(
                f"{color_col}:N",
                scale=alt.Scale(domain=domain, range=range_colors),
                legend=alt.Legend(title=color_col),
            ),
            tooltip=[
                alt.Tooltip(f"{color_col}:N"),
                alt.Tooltip(f"{x_col}:Q"),
                alt.Tooltip(f"{y_col}:Q", format=".6f"),
            ],
        )
        .properties(height=320, title=title)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# ======================================================================
# 1) JST MACRO DATA (GDP + House prices) FOR TRAINING COUNTRIES
# ======================================================================

@st.cache_data
def get_jst_macro_series(jst_path: str) -> pd.DataFrame:
    df = pd.read_excel(jst_path)
    df = df[df["country"].isin(["USA", "UK", "Canada"])].copy()
    df = df.sort_values(["country", "year"])

    keep_cols = ["country", "year"]
    for c in ["gdp", "hpnom", "cpi"]:
        if c in df.columns:
            keep_cols.append(c)

    out = df[keep_cols].copy()

    if "hpnom" in out.columns and "cpi" in out.columns:
        out["house_price_real"] = out["hpnom"] / (out["cpi"] + 1e-9)

    return out

# ======================================================================
# 2) MODEL PIPELINE FUNCTIONS
# ======================================================================

def load_data(file="JSTdatasetR6.xlsx"):
    df = pd.read_excel(file)
    df = df[df["country"].isin(["USA", "UK", "Canada"])].copy()
    df = df.sort_values(["country", "year"])
    return df

def engineer_features(df, us_ltrate_map: dict | None = None):
    """
    If us_ltrate_map is provided, sovereign_spread uses USA long rate from that mapping.
    This prevents single-country uploads from producing all-NaN sovereign_spread.
    """
    df = df.copy()

    if "crisisJST" not in df.columns:
        df["crisisJST"] = 0

    df["country"] = df["country"].astype(str).str.strip()

    df["leverage_risk"] = 1 / (df["lev"] + 0.01)

    def expanding_z(df_inner, col):
        def z(s):
            mu = s.expanding().mean()
            sd = s.expanding().std().replace(0, np.nan)
            return (s - mu) / (sd + 1e-9)
        return df_inner.groupby("country")[col].transform(z)

    df["noncore_z"]  = expanding_z(df, "noncore")
    df["ltd_z"]      = expanding_z(df, "ltd")
    df["leverage_z"] = expanding_z(df, "leverage_risk")

    df["banking_fragility"] = (
        0.4 * df["noncore_z"] +
        0.3 * df["ltd_z"] +
        0.3 * df["leverage_z"]
    )

    df["hp_real"]  = df["hpnom"] / (df["cpi"] + 1e-9)
    df["hp_trend"] = df.groupby("country")["hp_real"].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )
    df["housing_bubble"] = (df["hp_real"] - df["hp_trend"]) / (df["hp_trend"] + 1e-9)

    df["real_credit"]   = df["tloans"] / (df["cpi"] + 1e-9)
    df["credit_growth"] = df.groupby("country")["real_credit"].pct_change()

    df["yield_curve"] = df["ltrate"] - df["stir"]

    # Sovereign spread
    if us_ltrate_map is None:
        if "USA" in set(df["country"].unique()):
            us_ltrate_map = (
                df[df["country"] == "USA"]
                .drop_duplicates("year")
                .set_index("year")["ltrate"]
                .to_dict()
            )
        else:
            us_ltrate_map = {}

    df["us_ltrate"] = df["year"].map(us_ltrate_map) if isinstance(us_ltrate_map, dict) else np.nan
    df["sovereign_spread"] = df["ltrate"] - df["us_ltrate"]

    df["money_gdp"] = df["money"] / (df["gdp"] + 1e-9)
    df["money_expansion"] = df.groupby("country")["money_gdp"].pct_change()

    df["ca_gdp"] = df["ca"] / (df["gdp"] + 1e-9)

    base_features = [
        "housing_bubble", "credit_growth", "banking_fragility",
        "sovereign_spread", "yield_curve",
        "money_expansion", "ca_gdp"
    ]

    df = df[["country", "year", "crisisJST"] + base_features].copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    return df, base_features

def clean_data(df, base_features):
    df = df.copy()

    df = df[~df["year"].between(1914, 1918)]
    df = df[~df["year"].between(1939, 1945)]

    for col in base_features:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    df[base_features] = df.groupby("country")[base_features].transform(
        lambda x: x.ffill(limit=3).bfill(limit=3)
    )

    df[base_features] = df.groupby("country")[base_features].transform(
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

def build_model_set():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Random Forest": RandomForestClassifier(
            n_estimators=600, max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.01,
            max_depth=3,
            random_state=42
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True,
            C=2.0, class_weight="balanced"
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=2000,
            random_state=42
        )
    }

def evaluate_model(name, model, X_tr, y_tr, X_val, y_val):
    weights = np.where(y_tr == 1, 10, 1)

    if name == "Neural Network":
        model.fit(X_tr, y_tr)
    else:
        try:
            model.fit(X_tr, y_tr, sample_weight=weights)
        except TypeError:
            model.fit(X_tr, y_tr)

    probs = model.predict_proba(X_val)[:, 1]

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    return {
        "model": name,
        "ROC-AUC": float(roc_auc_score(y_val, probs)),
        "PR-AUC": float(average_precision_score(y_val, probs)),
        "F1": float(best_f1),
        "threshold": float(best_t),
        "clf": model
    }

# ======================================================================
# 3) TRAINING + CACHING
# ======================================================================

@st.cache_resource
def train_pipeline(jst_path: str):
    df_raw = load_data(jst_path)

    us_ltrate_map = (
        df_raw[df_raw["country"] == "USA"]
        .drop_duplicates("year")
        .set_index("year")["ltrate"]
        .to_dict()
    )

    df_feat, base_features = engineer_features(df_raw, us_ltrate_map=us_ltrate_map)
    df_clean = clean_data(df_feat, base_features)

    df_target = create_target(df_clean).reset_index(drop=True)

    missing_features = [f"{f}_missing" for f in base_features]
    all_features = base_features + missing_features

    train = df_target[df_target["year"] < 1970]
    val   = df_target[(df_target["year"] >= 1970) & (df_target["year"] < 1990)]
    test  = df_target[df_target["year"] >= 1990]

    X_train = train[all_features]
    X_val   = val[all_features]
    X_test  = test[all_features]

    y_train = train["target"]
    y_val   = val["target"]
    y_test  = test["target"]

    base_feature_medians = df_target[base_features].median(numeric_only=True).to_dict()

    scaler = StandardScaler()
    Xs_train_cont = scaler.fit_transform(X_train[base_features])
    Xs_val_cont   = scaler.transform(X_val[base_features])
    Xs_test_cont  = scaler.transform(X_test[base_features])

    Xs_train = np.hstack([Xs_train_cont, X_train[missing_features].values])
    Xs_val   = np.hstack([Xs_val_cont,   X_val[missing_features].values])
    Xs_test  = np.hstack([Xs_test_cont,  X_test[missing_features].values])

    results = []
    fitted_models = {}

    for name, clf in build_model_set().items():
        res = evaluate_model(name, clf, Xs_train, y_train, Xs_val, y_val)
        fitted_models[name] = res["clf"]
        results.append({k: v for k, v in res.items() if k != "clf"})

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)

    best_name = results_df.loc[0, "model"]
    best_thresh = float(results_df.loc[0, "threshold"])

    test_probs_by_model = {}
    for name, clf in fitted_models.items():
        test_probs_by_model[name] = clf.predict_proba(Xs_test)[:, 1]

    X_full = df_target[all_features]
    X_full_cont = scaler.transform(X_full[base_features])
    X_full_scaled = np.hstack([X_full_cont, X_full[missing_features].values])

    return {
        "df_target": df_target,
        "base_features": base_features,
        "missing_features": missing_features,
        "all_features": all_features,
        "scaler": scaler,
        "models": fitted_models,
        "results_df": results_df,
        "best_name": best_name,
        "best_thresh": best_thresh,
        "Xs_test": Xs_test,
        "y_test": y_test.to_numpy(),
        "X_full_scaled": X_full_scaled,
        "test_probs_by_model": test_probs_by_model,
        "us_ltrate_map": us_ltrate_map,
        "base_feature_medians": base_feature_medians,
    }

# ======================================================================
# 4) SIDEBAR UPLOAD & PREDICT (returns predictions + macro series for Tab 1)
# ======================================================================

def run_upload_predict_sidebar(
    selected_model,
    scaler,
    threshold,
    base_features,
    missing_features,
    all_features,
    us_ltrate_map,
    base_feature_medians,
):
    st.sidebar.header("📥 Upload & Predict")

    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="upload_csv")

    if not uploaded:
        st.sidebar.info("Upload a CSV to generate predictions.")
        return None

    try:
        user_df = pd.read_csv(uploaded)
    except Exception:
        st.sidebar.error("Could not read CSV. Make sure it is a valid CSV file.")
        return None

    required_raw = {
        "country","year","lev","noncore","ltd","hpnom","cpi",
        "tloans","ltrate","stir","money","gdp","ca"
    }
    missing_cols = sorted(list(required_raw - set(user_df.columns)))
    if missing_cols:
        st.sidebar.error(
            "Missing columns (showing up to 12): " + ", ".join(missing_cols[:12]) +
            ("..." if len(missing_cols) > 12 else "")
        )
        return None

    # Build uploaded macro series for Tab 1
    macro = user_df[["country", "year", "gdp", "hpnom", "cpi"]].copy()
    macro["country"] = macro["country"].astype(str).str.strip()
    macro["house_price_real"] = macro["hpnom"] / (macro["cpi"] + 1e-9)
    macro = macro.sort_values(["country", "year"])

    # Feature engineering using JST USA ltrate map
    df_feat, base_feats_upload = engineer_features(user_df, us_ltrate_map=us_ltrate_map)
    df_clean = clean_data(df_feat, base_feats_upload)

    # Hardening: fill remaining NaNs using training medians (then 0)
    for col in base_feats_upload:
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        fill_val = base_feature_medians.get(col, np.nan)
        if pd.isna(fill_val):
            fill_val = 0.0
        df_clean[col] = df_clean[col].fillna(fill_val)

    for col in base_feats_upload:
        miss_col = f"{col}_missing"
        if miss_col not in df_clean.columns:
            df_clean[miss_col] = 0

    # Build matrix in exact training order
    for f in all_features:
        if f not in df_clean.columns:
            df_clean[f] = 0

    X_u = df_clean[all_features].copy()

    X_u_cont = scaler.transform(X_u[base_features])
    X_u_scaled = np.hstack([X_u_cont, X_u[missing_features].values])

    if not np.isfinite(X_u_scaled).all():
        st.sidebar.error("Upload data still contains NaN/Inf after cleaning (cannot predict).")
        return None

    try:
        probs = selected_model.predict_proba(X_u_scaled)[:, 1]
    except Exception:
        st.sidebar.error("Prediction failed. Try selecting a different model.")
        return None

    preds = (probs >= threshold).astype(int)

    out = df_clean[["country", "year"]].copy()
    out["predicted_prob"] = probs
    out["predicted_class"] = preds
    out = out.sort_values(["country", "year"])

    st.sidebar.success(f"{int(preds.sum())} high-risk rows ({100 * preds.mean():.1f}%)")

    return {
        "preds": out,
        "macro": macro
    }

# ======================================================================
# 5) SHAP PIE (cached)
# ======================================================================

def get_shap_pie_data(model_key: str, selected_model, Xs_test: np.ndarray, feature_names: list, sample_n: int):
    if "shap_pie_cache" not in st.session_state:
        st.session_state["shap_pie_cache"] = {}

    cache_key = (model_key, int(sample_n))
    if cache_key in st.session_state["shap_pie_cache"]:
        return st.session_state["shap_pie_cache"][cache_key]

    X_test_shap = pd.DataFrame(Xs_test, columns=feature_names)
    Xsamp = X_test_shap.sample(n=min(sample_n, len(X_test_shap)), random_state=42)

    explainer = shap.Explainer(selected_model, X_test_shap)
    shap_vals = explainer(Xsamp)

    values = shap_vals.values
    if values.ndim == 3:
        values = values[:, :, 1]

    mean_abs = np.mean(np.abs(values), axis=0)
    total = float(mean_abs.sum()) + 1e-12
    shares = mean_abs / total

    df_pie = pd.DataFrame({"Feature": feature_names, "Share": shares})
    df_pie = df_pie.sort_values("Share", ascending=False).reset_index(drop=True)

    st.session_state["shap_pie_cache"][cache_key] = df_pie
    return df_pie

# ======================================================================
# 6) LOAD EVERYTHING
# ======================================================================

if not JST_XLSX.exists():
    st.error(f"Missing JST dataset file: {JST_XLSX.name}. Put it next to app.py.")
    st.stop()

bundle = train_pipeline(str(JST_XLSX))

df_target = bundle["df_target"]
base_features = bundle["base_features"]
missing_features = bundle["missing_features"]
all_features = bundle["all_features"]
scaler = bundle["scaler"]
models = bundle["models"]
results_df = bundle["results_df"]
best_name = bundle["best_name"]
best_thresh = bundle["best_thresh"]
Xs_test = bundle["Xs_test"]
y_test = bundle["y_test"]
X_full_scaled = bundle["X_full_scaled"]
test_probs_by_model = bundle["test_probs_by_model"]
us_ltrate_map = bundle["us_ltrate_map"]
base_feature_medians = bundle["base_feature_medians"]

macro_train_df = get_jst_macro_series(str(JST_XLSX))

# ======================================================================
# 7) SIDEBAR CONTROLS (model first, upload, then Tab 1 data source)
# ======================================================================

st.sidebar.header("🎛️ Model & Filters")

model_names = list(models.keys())
default_index = model_names.index(best_name) if best_name in model_names else 0

model_choice = st.sidebar.selectbox(
    "Select model",
    options=model_names,
    index=default_index,
    key="model_choice"
)

selected_model = models[model_choice]

threshold = st.sidebar.slider(
    "Risk threshold",
    min_value=0.05,
    max_value=0.90,
    value=float(best_thresh),
    step=0.01,
    key="risk_threshold"
)

st.sidebar.subheader("🧠 SHAP Pie (first page)")
shap_pie_sample_n = st.sidebar.slider("SHAP sample size", 50, 400, 150, 25)

# Upload & predict bundle
uploaded_bundle = run_upload_predict_sidebar(
    selected_model=selected_model,
    scaler=scaler,
    threshold=threshold,
    base_features=base_features,
    missing_features=missing_features,
    all_features=all_features,
    us_ltrate_map=us_ltrate_map,
    base_feature_medians=base_feature_medians,
)

# Data source for Tab 1
st.sidebar.subheader("📌 Tab 1 view")
if uploaded_bundle is not None:
    tab1_source = st.sidebar.radio(
        "Show on Tab 1",
        ["Uploaded (your CSV)", "Training (USA/UK/Canada)"],
        index=0,
        key="tab1_source"
    )
else:
    tab1_source = st.sidebar.radio(
        "Show on Tab 1",
        ["Training (USA/UK/Canada)", "Uploaded (your CSV)"],
        index=0,
        key="tab1_source"
    )

use_uploaded_on_tab1 = (tab1_source.startswith("Uploaded") and uploaded_bundle is not None)

# Filters for Tab 1 (change depending on source)
if use_uploaded_on_tab1:
    preds_u = uploaded_bundle["preds"]
    countries_u = sorted(preds_u["country"].unique())
    min_y = int(preds_u["year"].min())
    max_y = int(preds_u["year"].max())

    view_countries = st.sidebar.multiselect(
        "Countries (uploaded)",
        options=countries_u,
        default=countries_u,
        key="uploaded_countries"
    )

    from_year, to_year = st.sidebar.slider(
        "Year range (uploaded)",
        min_value=min_y,
        max_value=max_y,
        value=(min_y, max_y),
        key="uploaded_year_slider"
    )

    show_crisis_years = False  # crisisJST not present/reliable in uploads
else:
    view_countries = st.sidebar.multiselect(
        "Countries (training timeline)",
        ["USA", "UK", "Canada"],
        default=["USA", "UK", "Canada"],
        key="train_countries"
    )

    min_y = int(df_target["year"].min())
    max_y = int(df_target["year"].max())

    from_year, to_year = st.sidebar.slider(
        "Year range (training)",
        min_value=min_y,
        max_value=max_y,
        value=(1900, max_y),
        key="train_year_slider"
    )

    show_crisis_years = st.sidebar.checkbox("Show crisis years list (training)", value=True)

# ======================================================================
# 8) BUILD TAB 1 DISPLAY DF
# ======================================================================

if use_uploaded_on_tab1:
    preds_u = uploaded_bundle["preds"].copy()
    display_df = preds_u[
        (preds_u["country"].isin(view_countries)) &
        (preds_u["year"] >= from_year) &
        (preds_u["year"] <= to_year)
    ].copy()

    # unify column name for charting
    display_df["crisis_prob"] = display_df["predicted_prob"]

    macro_df = uploaded_bundle["macro"].copy()
    macro_df = macro_df[
        (macro_df["country"].isin(view_countries)) &
        (macro_df["year"] >= from_year) &
        (macro_df["year"] <= to_year)
    ].copy()

else:
    display_df = df_target[
        (df_target["country"].isin(view_countries)) &
        (df_target["year"] >= from_year) &
        (df_target["year"] <= to_year)
    ].copy()

    # compute probs for training timeline
    scaled = X_full_scaled[display_df.index.to_numpy()]
    display_df["crisis_prob"] = selected_model.predict_proba(scaled)[:, 1]

    macro_df = macro_train_df[
        (macro_train_df["country"].isin(view_countries)) &
        (macro_train_df["year"] >= from_year) &
        (macro_train_df["year"] <= to_year)
    ].copy()

# ======================================================================
# 9) TABS
# ======================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Crisis Risk",
    "📊 Model Results",
    "📥 Uploaded Predictions",
    "🧠 SHAP",
    "📂 Data Explorer",
])

# -----------------------------------------------------------------------------
# TAB 1: NOW ADAPTS TO UPLOAD OR TRAINING
with tab1:
    st.header("Crisis risk over time", divider="gray")

    if display_df.empty:
        st.warning("No data for the selected filters.")
    else:
        altair_line_chart(
            display_df.sort_values(["country", "year"]),
            x_col="year",
            y_col="crisis_prob",
            color_col="country",
            title="Predicted crisis probability over time"
        )

        st.write("")
        st.subheader(f"Crisis summary ({to_year})", divider="gray")

        latest = display_df[display_df["year"] == to_year]
        cols = st.columns(min(4, max(1, len(view_countries))))

        for i, c in enumerate(view_countries):
            with cols[i % len(cols)]:
                val = latest[latest["country"] == c]["crisis_prob"].mean()
                if pd.isna(val):
                    st.metric(label=f"{c} crisis risk", value="n/a")
                else:
                    val_pct = val * 100
                    thr_pct = threshold * 100
                    delta_pp = val_pct - thr_pct
                    st.metric(
                        label=f"{c} crisis risk",
                        value=f"{val_pct:.1f}%",
                        delta=f"{delta_pp:+.1f} pp vs threshold",
                        delta_color="inverse" if val >= threshold else "normal",
                    )

        if (not use_uploaded_on_tab1) and show_crisis_years:
            with st.expander("Crisis years in selected window (training only)"):
                for c in view_countries:
                    years = display_df[(display_df["country"] == c) & (display_df["crisisJST"] == 1)]["year"].tolist()
                    st.write(f"**{c}:** " + (", ".join(map(str, years)) if years else "none"))

        st.write("")
        st.subheader("Macro context + explainability", divider="gray")

        left, right = st.columns(2)

        # LEFT: GDP + house prices for whichever dataset is active
        with left:
            st.markdown("### 🌍 GDP")
            if "gdp" in macro_df.columns and not macro_df["gdp"].isna().all():
                altair_line_chart(
                    macro_df.sort_values(["country", "year"]),
                    x_col="year",
                    y_col="gdp",
                    color_col="country",
                    title="GDP"
                )
            else:
                st.info("GDP not available for the selected data / years.")

            st.write("")
            st.markdown("### 🏠 Real house prices (hpnom / cpi)")
            if "house_price_real" in macro_df.columns and not macro_df["house_price_real"].isna().all():
                altair_line_chart(
                    macro_df.sort_values(["country", "year"]),
                    x_col="year",
                    y_col="house_price_real",
                    color_col="country",
                    title="Real house prices"
                )
            else:
                st.info("House price data not available for the selected data / years.")

        # RIGHT: SHAP pie + explanation
        with right:
            st.markdown("### 🧠 SHAP Explainability")
            st.caption("Pie uses **mean(|SHAP|)** across sampled test rows (share of total influence).")

            top_k = 8
            try:
                shap_pie_df = get_shap_pie_data(
                    model_key=model_choice,
                    selected_model=selected_model,
                    Xs_test=Xs_test,
                    feature_names=all_features,
                    sample_n=shap_pie_sample_n
                )

                top = shap_pie_df.head(top_k).copy()
                other_share = float(shap_pie_df["Share"].iloc[top_k:].sum()) if len(shap_pie_df) > top_k else 0.0
                if other_share > 0:
                    top = pd.concat(
                        [top, pd.DataFrame([{"Feature": "Other", "Share": other_share}])],
                        ignore_index=True
                    )

                top["Share_%"] = (top["Share"] * 100).round(2)

                if PLOTLY_OK:
                    fig = px.pie(
                        top,
                        names="Feature",
                        values="Share",
                        hover_data={"Share_%": True, "Share": True},
                        title="SHAP Feature Impact Share (mean |SHAP|)"
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        height=520,
                        margin=dict(l=10, r=10, t=60, b=20),
                        showlegend=False,
                    )
                    fig.update_traces(
                        textposition="outside",
                        textinfo="percent+label",
                        textfont_size=14,
                        pull=[0.02] * len(top),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    labels = top["Feature"].tolist()
                    sizes = top["Share"].tolist()
                    plt.figure(figsize=(7, 6))
                    plt.pie(
                        sizes,
                        labels=labels,
                        autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",
                        startangle=90
                    )
                    plt.title("SHAP Feature Impact Share (mean |SHAP|)")
                    st.pyplot(plt.gcf(), clear_figure=True)
                    st.info("Install `plotly` to make the pie chart interactive (add `plotly` to requirements.txt).")

                st.markdown(
                    """
**What SHAP values represent**

- SHAP decomposes a model prediction into **feature contributions** around a baseline.  
- **Positive SHAP** pushes predicted crisis risk **up**; **negative SHAP** pushes it **down**.  
- Larger **|SHAP|** means **stronger influence**.

**What this pie chart shows**

This pie uses **mean(|SHAP|)** over many observations, so it is **global importance** (share of total influence).  
It shows *how much* a feature matters overall, not the direction of its effect.
"""
                )

                with st.expander("Show SHAP importance table (top 20)"):
                    st.dataframe(
                        shap_pie_df.assign(Share_pct=shap_pie_df["Share"] * 100).head(20),
                        use_container_width=True
                    )

            except Exception:
                st.warning(
                    "SHAP could not be computed for this model. "
                    "Try Gradient Boosting / Random Forest or reduce SHAP sample size."
                )

# -----------------------------------------------------------------------------
# TAB 2: Model Results
with tab2:
    st.header("Model comparison (validation)", divider="gray")
    st.dataframe(results_df, use_container_width=True)

    st.write("")
    st.info(
        "The table above shows **VALIDATION (1970–1989)** metrics. "
        "The metrics below show **TEST (1990–2020)** performance for the selected model."
    )

    st.subheader("Out-of-sample performance (post-1990)", divider="gray")
    st.markdown(f"**Current model:** `{model_choice}`  \n**Threshold:** `{threshold:.2f}`")

    test_probs = test_probs_by_model[model_choice]
    test_preds = (test_probs >= threshold).astype(int)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{roc_auc_score(y_test, test_probs):.3f}")
    c2.metric("PR-AUC", f"{average_precision_score(y_test, test_probs):.3f}")
    c3.metric("Precision", f"{precision_score(y_test, test_preds, zero_division=0):.3f}")
    c4.metric("Recall", f"{recall_score(y_test, test_preds, zero_division=0):.3f}")
    c5.metric("F1", f"{f1_score(y_test, test_preds, zero_division=0):.3f}")

    st.write("")
    st.subheader("Confusion matrix (test)", divider="gray")
    cm = confusion_matrix(y_test, test_preds)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm_df, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Uploaded Predictions (still available)
with tab3:
    st.header("Predictions from uploaded CSV", divider="gray")

    if uploaded_bundle is None:
        st.info("Upload a CSV from the sidebar to view predictions here.")
    else:
        uploaded_preds = uploaded_bundle["preds"]
        st.dataframe(uploaded_preds, use_container_width=True)

        st.write("")
        st.subheader("Uploaded risk over time", divider="gray")
        altair_line_chart(
            uploaded_preds.sort_values(["country", "year"]),
            x_col="year",
            y_col="predicted_prob",
            color_col="country",
            title="Uploaded predicted crisis probability over time"
        )

        csv_bytes = uploaded_preds.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions as CSV",
            data=csv_bytes,
            file_name="uploaded_predictions.csv",
            mime="text/csv",
        )

# -----------------------------------------------------------------------------
# TAB 4: SHAP (detailed)
with tab4:
    st.header("SHAP explainability", divider="gray")

    st.markdown(
        """
**What are SHAP values?**  
SHAP (SHapley Additive exPlanations) explains a model prediction by attributing it to individual feature contributions.

For a single observation:

**prediction = base value + sum(feature contributions)**

**Interpretation**
- **Positive SHAP** → pushes prediction **towards crisis (higher risk)**
- **Negative SHAP** → pushes prediction **away from crisis (lower risk)**
- Larger **|SHAP|** → stronger effect

**Note:** The pie chart uses **mean absolute SHAP** (importance share), which shows *how much* a feature matters overall, not direction.
"""
    )

    sample_n = st.slider("Sample size (rows)", 50, 500, 200, 25, key="shap_sample_n_tab")
    plot_type = st.selectbox("Plot type", ["bar", "dot"], index=0)

    if st.button("Compute SHAP", type="primary", key="compute_shap_tab"):
        with st.spinner("Computing SHAP..."):
            feature_names = all_features
            X_test_shap = pd.DataFrame(Xs_test, columns=feature_names)
            Xsamp = X_test_shap.sample(n=min(sample_n, len(X_test_shap)), random_state=42)

            explainer = shap.Explainer(selected_model, X_test_shap)
            shap_vals = explainer(Xsamp)

            plt.figure()
            if plot_type == "bar":
                shap.summary_plot(shap_vals, Xsamp, plot_type="bar", show=False)
            else:
                shap.summary_plot(shap_vals, Xsamp, show=False)

            st.pyplot(plt.gcf(), clear_figure=True)

# -----------------------------------------------------------------------------
# TAB 5: Data Explorer
with tab5:
    st.header("Data explorer (processed)", divider="gray")
    st.caption("Cleaned + engineered dataset used for modelling (includes missing flags + target).")

    show_cols = ["country", "year", "crisisJST", "target"] + all_features
    st.dataframe(
        df_target[show_cols].sort_values(["country", "year"]).tail(120),
        use_container_width=True
    )

    with st.expander("Show feature lists"):
        st.write("**Base (continuous) features**:", base_features)
        st.write("**Missing flags**:", missing_features)
        st.write("**All features (model input order)**:", all_features)
