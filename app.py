# ======================================================================
#  FINANCIAL CRISIS EARLY WARNING SYSTEM – STREAMLIT DASHBOARD (FULL)
#
#  INCLUDED (your requests):
#   ✅ Tab 1 switches to uploaded country when you upload (optional toggle)
#   ✅ Crisis risk chart:
#        - Training (USA/UK/Canada): shaded crisis bands (crisisJST)
#        - Uploaded: WHITE vertical crisis lines (requires crisisJST in upload)
#   ✅ Tab 1 ALSO shows (for BOTH default + uploaded):
#        - GDP line chart (from JST for default; from uploaded CSV for uploaded)
#        - Real house price chart (hpnom/cpi)
#        - SHAP interactive pie + explanation directly under it
#   ✅ RGB-consistent country colors (Altair) across charts
#   ✅ Crisis summary displayed as percentage + delta vs threshold
#   ✅ Upload crash fix (Spain etc.): sovereign spread uses USA ltrate from JST + robust NaN handling
#   ✅ Arrow-safe results table (no sklearn objects in dataframe)
#
#  Notes:
#   - Plotly is OPTIONAL. If not installed, pie falls back to matplotlib (non-interactive).
#   - Altair is OPTIONAL. If not installed, charts fall back to st.line_chart (no custom colors/lines).
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt

# Optional Plotly for interactive pie
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Optional Altair for interactive RGB charts + crisis overlays
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
st.set_page_config(page_title="Financial Crisis EWS Dashboard", page_icon="📉", layout="wide")

APP_DIR = Path(__file__).parent
JST_XLSX = APP_DIR / "JSTdatasetR6.xlsx"

# -----------------------------------------------------------------------------
# RGB color settings
BASE_COUNTRY_ORDER = ["USA", "UK", "Canada"]
BASE_COUNTRY_RGB = {
    "USA": "rgb(255, 99, 71)",      # tomato
    "UK": "rgb(100, 149, 237)",     # cornflower blue
    "Canada": "rgb(60, 179, 113)",  # medium sea green
}

# -----------------------------------------------------------------------------
# HEADER
st.title("📉 Financial Crisis Early Warning System (EWS)")
st.caption(
    "Multi-model pipeline (missing flags + proper scaling + validation selection) "
    "with risk timeline, GDP/house-price context, and SHAP explainability."
)
st.write("")

# -----------------------------------------------------------------------------
# Clear cache button
with st.sidebar:
    if st.button("Clear cache & rerun"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# ======================================================================
# Helpers (Colors + Charts)
# ======================================================================

def stable_rgb(name: str) -> str:
    """Deterministic RGB for any unseen country label (so Spain etc. get a stable color)."""
    h = abs(hash(name))
    r = 60 + (h % 140)
    g = 60 + ((h // 140) % 140)
    b = 60 + ((h // (140 * 140)) % 140)
    return f"rgb({r},{g},{b})"

def build_color_scale(categories):
    cats = [str(c) for c in categories]
    domain = []
    for c in BASE_COUNTRY_ORDER:
        if c in cats and c not in domain:
            domain.append(c)
    for c in sorted(set(cats)):
        if c not in domain:
            domain.append(c)
    range_colors = [BASE_COUNTRY_RGB.get(c, stable_rgb(c)) for c in domain]
    return domain, range_colors

def _altair_base_line(df, x_col, y_col, color_col):
    domain, range_colors = build_color_scale(df[color_col].unique())
    base = alt.Chart(df).encode(
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
    line = base.mark_line(point=False)
    return line, domain, range_colors

def line_chart_training_with_bands(df, x_col, y_col, color_col, title, crisis_df):
    """Training view: shaded crisis bands by country."""
    if df.empty:
        st.info("No data to plot.")
        return
    if not ALTAIR_OK:
        st.line_chart(df, x=x_col, y=y_col, color=color_col)
        return

    line, domain, range_colors = _altair_base_line(df, x_col, y_col, color_col)
    layers = []

    if crisis_df is not None and not crisis_df.empty:
        cdf = crisis_df.copy()
        cdf[color_col] = cdf[color_col].astype(str)
        cdf["x_start"] = cdf["year"] - 0.5
        cdf["x_end"] = cdf["year"] + 0.5

        rect = (
            alt.Chart(cdf)
            .mark_rect(opacity=0.15)
            .encode(
                x=alt.X("x_start:Q", title=None),
                x2="x_end:Q",
                y=alt.value(0),
                y2=alt.value(1),
                color=alt.Color(
                    f"{color_col}:N",
                    scale=alt.Scale(domain=domain, range=range_colors),
                    legend=None,
                ),
            )
        )
        layers.append(rect)

    layers.append(line)
    chart = alt.layer(*layers).properties(height=320, title=title).interactive()
    st.altair_chart(chart, use_container_width=True)

def line_chart_uploaded_with_white_lines(df, x_col, y_col, color_col, title, crisis_years):
    """Uploaded view: WHITE vertical lines at crisis years (single set across all uploaded countries)."""
    if df.empty:
        st.info("No data to plot.")
        return
    if not ALTAIR_OK:
        st.line_chart(df, x=x_col, y=y_col, color=color_col)
        return

    line, _, _ = _altair_base_line(df, x_col, y_col, color_col)
    layers = [line]

    crisis_years = sorted(set([int(y) for y in (crisis_years or [])]))
    if crisis_years:
        rule_df = pd.DataFrame({"year": crisis_years})
        rules = alt.Chart(rule_df).mark_rule(color="white", strokeWidth=2, opacity=0.95).encode(x="year:Q")
        layers.append(rules)

    chart = alt.layer(*layers).properties(height=320, title=title).interactive()
    st.altair_chart(chart, use_container_width=True)

def line_chart_simple(df, x_col, y_col, color_col, title):
    """Fallback generic interactive chart (no crisis overlays)."""
    if df.empty:
        st.info("No data to plot.")
        return
    if not ALTAIR_OK:
        st.line_chart(df, x=x_col, y=y_col, color=color_col)
        return
    line, _, _ = _altair_base_line(df, x_col, y_col, color_col)
    st.altair_chart(line.properties(height=320, title=title).interactive(), use_container_width=True)

# ======================================================================
# JST Macro series (GDP + House prices) for training countries
# ======================================================================

@st.cache_data
def get_jst_macro_series(jst_path: str) -> pd.DataFrame:
    df = pd.read_excel(jst_path)
    df = df[df["country"].isin(["USA", "UK", "Canada"])].copy()
    df = df.sort_values(["country", "year"])

    cols = ["country", "year", "gdp", "hpnom", "cpi"]
    cols = [c for c in cols if c in df.columns]

    out = df[cols].copy()
    if "hpnom" in out.columns and "cpi" in out.columns:
        out["house_price_real"] = out["hpnom"] / (out["cpi"] + 1e-9)
    return out

# ======================================================================
# Your pipeline functions
# ======================================================================

def load_data(file="JSTdatasetR6.xlsx"):
    df = pd.read_excel(file)
    df = df[df["country"].isin(["USA", "UK", "Canada"])].copy()
    df = df.sort_values(["country", "year"])
    return df

def engineer_features(df, us_ltrate_map: dict | None = None):
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

    df["banking_fragility"] = (0.4 * df["noncore_z"] + 0.3 * df["ltd_z"] + 0.3 * df["leverage_z"])

    df["hp_real"]  = df["hpnom"] / (df["cpi"] + 1e-9)
    df["hp_trend"] = df.groupby("country")["hp_real"].transform(lambda x: x.rolling(10, min_periods=5).mean())
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
        "sovereign_spread", "yield_curve", "money_expansion", "ca_gdp"
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

    df[base_features] = df.groupby("country")[base_features].transform(lambda x: x.ffill(limit=3).bfill(limit=3))
    df[base_features] = df.groupby("country")[base_features].transform(lambda x: x.fillna(x.median()))
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
            n_estimators=600, max_depth=4, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.01, max_depth=3, random_state=42
        ),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, C=2.0, class_weight="balanced"),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42),
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

    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.90, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)

    return {
        "model": name,
        "ROC-AUC": float(roc_auc_score(y_val, probs)),
        "PR-AUC": float(average_precision_score(y_val, probs)),
        "F1": float(best_f1),
        "threshold": float(best_t),
        "clf": model,
    }

# ======================================================================
# Train pipeline (cached)
# ======================================================================

@st.cache_resource
def train_pipeline(jst_path: str):
    df_raw = load_data(jst_path)

    # USA long-rate map from JST (needed for uploads like Spain)
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

    # training medians for robust upload fills
    base_feature_medians = df_target[base_features].median(numeric_only=True).to_dict()

    # scale continuous only
    scaler = StandardScaler()
    Xs_train_cont = scaler.fit_transform(X_train[base_features])
    Xs_val_cont   = scaler.transform(X_val[base_features])
    Xs_test_cont  = scaler.transform(X_test[base_features])

    Xs_train = np.hstack([Xs_train_cont, X_train[missing_features].values])
    Xs_val   = np.hstack([Xs_val_cont,   X_val[missing_features].values])
    Xs_test  = np.hstack([Xs_test_cont,  X_test[missing_features].values])

    # train models
    results = []
    fitted_models = {}

    for name, clf in build_model_set().items():
        res = evaluate_model(name, clf, Xs_train, y_train, Xs_val, y_val)
        fitted_models[name] = res["clf"]
        # Arrow-safe results (exclude estimator)
        results.append({k: v for k, v in res.items() if k != "clf"})

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    best_name = results_df.loc[0, "model"]
    best_thresh = float(results_df.loc[0, "threshold"])

    # test probs for all models
    test_probs_by_model = {name: m.predict_proba(Xs_test)[:, 1] for name, m in fitted_models.items()}

    # full scaled matrix aligned to df_target indices
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
# Upload & Predict (sidebar)
# ======================================================================

def run_upload_predict_sidebar(selected_model, scaler, threshold,
                              base_features, missing_features, all_features,
                              us_ltrate_map, base_feature_medians):
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

    required_raw = {"country","year","lev","noncore","ltd","hpnom","cpi","tloans","ltrate","stir","money","gdp","ca"}
    missing_cols = sorted(list(required_raw - set(user_df.columns)))
    if missing_cols:
        st.sidebar.error("Missing columns: " + ", ".join(missing_cols[:12]) + ("..." if len(missing_cols) > 12 else ""))
        return None

    # Macro series (for GDP + house price charts)
    macro = user_df[["country","year","gdp","hpnom","cpi"]].copy()
    macro["country"] = macro["country"].astype(str).str.strip()
    macro["house_price_real"] = macro["hpnom"] / (macro["cpi"] + 1e-9)
    macro = macro.sort_values(["country","year"])

    # Features engineered using JST USA ltrate map (prevents sovereign_spread all-NaN)
    df_feat, base_feats_upload = engineer_features(user_df, us_ltrate_map=us_ltrate_map)
    df_clean = clean_data(df_feat, base_feats_upload)

    # Harden remaining NaNs using training medians (then 0)
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

    for f in all_features:
        if f not in df_clean.columns:
            df_clean[f] = 0

    X_u = df_clean[all_features].copy()
    X_u_cont = scaler.transform(X_u[base_features])
    X_u_scaled = np.hstack([X_u_cont, X_u[missing_features].values])

    if not np.isfinite(X_u_scaled).all():
        st.sidebar.error("Upload still contains NaN/Inf after cleaning (cannot predict).")
        return None

    try:
        probs = selected_model.predict_proba(X_u_scaled)[:, 1]
    except Exception:
        st.sidebar.error("Prediction failed for this model. Try a different model.")
        return None

    preds = (probs >= threshold).astype(int)

    out = df_clean[["country","year","crisisJST"]].copy()  # crisisJST included if uploaded had it, else 0
    out["predicted_prob"] = probs
    out["predicted_class"] = preds
    out = out.sort_values(["country","year"])

    st.sidebar.success(f"{int(preds.sum())} high-risk rows ({100 * preds.mean():.1f}%)")

    return {"preds": out, "macro": macro}

# ======================================================================
# SHAP pie (cached per model + sample)
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
    # If binary classifier returns (n, p, 2), take class 1
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
# Load / Train
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
# Sidebar controls
# ======================================================================

st.sidebar.header("🎛️ Model & View")

model_names = list(models.keys())
default_index = model_names.index(best_name) if best_name in model_names else 0
model_choice = st.sidebar.selectbox("Select model", options=model_names, index=default_index)
selected_model = models[model_choice]

threshold = st.sidebar.slider("Risk threshold", 0.05, 0.90, float(best_thresh), 0.01)

st.sidebar.subheader("🧠 SHAP (Tab 1)")
shap_pie_sample_n = st.sidebar.slider("SHAP sample size", 50, 400, 150, 25)

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

st.sidebar.subheader("📌 Tab 1 content")
if uploaded_bundle is not None:
    tab1_source = st.sidebar.radio("Show on Tab 1", ["Uploaded (your CSV)", "Training (USA/UK/Canada)"], index=0)
else:
    tab1_source = st.sidebar.radio("Show on Tab 1", ["Training (USA/UK/Canada)", "Uploaded (your CSV)"], index=0)

use_uploaded_on_tab1 = tab1_source.startswith("Uploaded") and uploaded_bundle is not None

# Filters (switch by source)
if use_uploaded_on_tab1:
    preds_u = uploaded_bundle["preds"]
    countries_u = sorted(preds_u["country"].unique())
    min_y = int(preds_u["year"].min())
    max_y = int(preds_u["year"].max())

    view_countries = st.sidebar.multiselect("Countries (uploaded)", countries_u, default=countries_u)
    from_year, to_year = st.sidebar.slider("Year range (uploaded)", min_y, max_y, (min_y, max_y))
else:
    view_countries = st.sidebar.multiselect("Countries (training)", ["USA", "UK", "Canada"], default=["USA", "UK", "Canada"])
    min_y = int(df_target["year"].min())
    max_y = int(df_target["year"].max())
    from_year, to_year = st.sidebar.slider("Year range (training)", min_y, max_y, (1900, max_y))

# ======================================================================
# Build Tab 1 datasets
# ======================================================================

if use_uploaded_on_tab1:
    preds_u = uploaded_bundle["preds"].copy()
    display_df = preds_u[
        (preds_u["country"].isin(view_countries)) &
        (preds_u["year"] >= from_year) &
        (preds_u["year"] <= to_year)
    ].copy()
    display_df["crisis_prob"] = display_df["predicted_prob"]

    # Uploaded crisis years for white vertical lines (requires crisisJST in upload)
    uploaded_crisis_years = display_df.loc[display_df["crisisJST"] == 1, "year"].dropna().astype(int).tolist()

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
    scaled = X_full_scaled[display_df.index.to_numpy()]
    display_df["crisis_prob"] = selected_model.predict_proba(scaled)[:, 1]

    crisis_years_df_train = (
        display_df[display_df["crisisJST"] == 1][["country","year"]]
        .drop_duplicates()
        .copy()
    )

    macro_df = macro_train_df[
        (macro_train_df["country"].isin(view_countries)) &
        (macro_train_df["year"] >= from_year) &
        (macro_train_df["year"] <= to_year)
    ].copy()

# ======================================================================
# Tabs
# ======================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Crisis Risk",
    "📊 Model Results",
    "📥 Uploaded Predictions",
    "🧠 SHAP",
    "📂 Data Explorer",
])

# -----------------------------------------------------------------------------
# TAB 1: Risk + GDP + House Prices + SHAP pie (FOR BOTH default and upload)
with tab1:
    st.header("Crisis risk over time", divider="gray")

    if display_df.empty:
        st.warning("No data for the selected filters.")
        st.stop()

    # Risk chart with crisis overlay
    if use_uploaded_on_tab1:
        line_chart_uploaded_with_white_lines(
            display_df.sort_values(["country","year"]),
            x_col="year",
            y_col="crisis_prob",
            color_col="country",
            title="Predicted crisis probability (white lines = crisis years from uploaded crisisJST)",
            crisis_years=uploaded_crisis_years
        )
    else:
        line_chart_training_with_bands(
            display_df.sort_values(["country","year"]),
            x_col="year",
            y_col="crisis_prob",
            color_col="country",
            title="Predicted crisis probability (shaded bands = crisis years in JST)",
            crisis_df=crisis_years_df_train
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
                val_pct = 100 * float(val)
                thr_pct = 100 * float(threshold)
                delta_pp = val_pct - thr_pct
                st.metric(
                    label=f"{c} crisis risk",
                    value=f"{val_pct:.1f}%",
                    delta=f"{delta_pp:+.1f} pp vs threshold",
                    delta_color="inverse" if val >= threshold else "normal",
                )

    st.write("")
    st.subheader("Macro context + explainability", divider="gray")
    left, right = st.columns(2)

    # ---- LEFT: GDP + House Prices (FOR BOTH)
    with left:
        st.markdown("### 🌍 GDP")
        if "gdp" in macro_df.columns and not macro_df["gdp"].isna().all():
            if use_uploaded_on_tab1:
                # (optional) add the same white crisis lines on macro charts if crisis years exist
                line_chart_uploaded_with_white_lines(
                    macro_df.sort_values(["country","year"]),
                    x_col="year",
                    y_col="gdp",
                    color_col="country",
                    title="GDP (uploaded/training source depending on Tab 1)",
                    crisis_years=uploaded_crisis_years if use_uploaded_on_tab1 else []
                )
            else:
                # training: no need to band macro chart; you can band if you want, but keeping clean
                line_chart_simple(
                    macro_df.sort_values(["country","year"]),
                    x_col="year",
                    y_col="gdp",
                    color_col="country",
                    title="GDP (JST)"
                )
        else:
            st.info("GDP not available for the selected data/years.")

        st.write("")
        st.markdown("### 🏠 Real house prices (hpnom / cpi)")
        if "house_price_real" in macro_df.columns and not macro_df["house_price_real"].isna().all():
            if use_uploaded_on_tab1:
                line_chart_uploaded_with_white_lines(
                    macro_df.sort_values(["country","year"]),
                    x_col="year",
                    y_col="house_price_real",
                    color_col="country",
                    title="Real house prices",
                    crisis_years=uploaded_crisis_years if use_uploaded_on_tab1 else []
                )
            else:
                line_chart_simple(
                    macro_df.sort_values(["country","year"]),
                    x_col="year",
                    y_col="house_price_real",
                    color_col="country",
                    title="Real house prices (JST)"
                )
        else:
            st.info("House price data not available for the selected data/years.")

    # ---- RIGHT: SHAP pie + explanation (FOR BOTH)
    with right:
        st.markdown("### 🧠 SHAP Explainability")
        st.caption("Pie = **mean(|SHAP|)** across sampled test rows → share of total feature influence (global importance).")

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
                top = pd.concat([top, pd.DataFrame([{"Feature": "Other", "Share": other_share}])], ignore_index=True)

            top["Share_%"] = (top["Share"] * 100).round(2)

            if PLOTLY_OK:
                fig = px.pie(
                    top, names="Feature", values="Share",
                    hover_data={"Share_%": True, "Share": True},
                    title="SHAP Feature Impact Share (mean |SHAP|)"
                )
                fig.update_layout(
                    height=520, margin=dict(l=10, r=10, t=60, b=20),
                    showlegend=False
                )
                fig.update_traces(textposition="outside", textinfo="percent+label", textfont_size=14, pull=[0.02]*len(top))
                st.plotly_chart(fig, use_container_width=True)
            else:
                # fallback: matplotlib pie (not interactive)
                plt.figure(figsize=(7, 6))
                plt.pie(
                    top["Share"].tolist(),
                    labels=top["Feature"].tolist(),
                    autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",
                    startangle=90
                )
                plt.title("SHAP Feature Impact Share (mean |SHAP|)")
                st.pyplot(plt.gcf(), clear_figure=True)
                st.info("For an interactive pie chart, add `plotly` to requirements.txt and redeploy.")

            # Explanation under the pie (as requested)
            st.markdown(
                """
**What SHAP values are**
- SHAP (SHapley Additive exPlanations) assigns each feature a contribution to a prediction relative to a baseline.
- **Positive SHAP** → pushes predicted crisis risk **up**.
- **Negative SHAP** → pushes predicted crisis risk **down**.
- Larger **|SHAP|** → stronger influence.

**What this pie represents**
- This is **global importance**: mean absolute SHAP across many rows.
- It shows *how much* each feature matters overall (share of total influence), not the direction.
"""
            )

            with st.expander("Show SHAP importance table (top 20)"):
                st.dataframe(
                    shap_pie_df.assign(Share_pct=shap_pie_df["Share"] * 100).head(20),
                    use_container_width=True
                )

        except Exception:
            st.warning("SHAP could not be computed for this model. Try Gradient Boosting / Random Forest or reduce SHAP sample size.")

# -----------------------------------------------------------------------------
# TAB 2: Model results + test metrics
with tab2:
    st.header("Model comparison (validation)", divider="gray")
    st.dataframe(results_df, use_container_width=True)

    st.write("")
    st.subheader("Out-of-sample performance (test: 1990–2020)", divider="gray")
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
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Uploaded predictions + chart + download
with tab3:
    st.header("Predictions from uploaded CSV", divider="gray")

    if uploaded_bundle is None:
        st.info("Upload a CSV from the sidebar to view predictions here.")
    else:
        uploaded_preds = uploaded_bundle["preds"].copy()
        st.dataframe(uploaded_preds, use_container_width=True)

        st.write("")
        st.subheader("Uploaded risk over time", divider="gray")
        tmp = uploaded_preds.copy()
        tmp["crisis_prob"] = tmp["predicted_prob"]
        crisis_years = tmp.loc[tmp["crisisJST"] == 1, "year"].dropna().astype(int).tolist()

        line_chart_uploaded_with_white_lines(
            tmp.sort_values(["country","year"]),
            x_col="year",
            y_col="crisis_prob",
            color_col="country",
            title="Uploaded predicted crisis probability (white lines = uploaded crisisJST years)",
            crisis_years=crisis_years
        )

        csv_bytes = uploaded_preds.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions as CSV", data=csv_bytes, file_name="uploaded_predictions.csv", mime="text/csv")

# -----------------------------------------------------------------------------
# TAB 4: Detailed SHAP plots (optional)
with tab4:
    st.header("SHAP explainability (detailed)", divider="gray")
    st.markdown(
        """
SHAP explains individual predictions by distributing the difference between the prediction and a baseline
across the features.

- **Positive SHAP** increases predicted crisis risk.
- **Negative SHAP** decreases predicted crisis risk.
- **Mean absolute SHAP** is used for global importance (what your pie chart shows).
"""
    )

    sample_n = st.slider("Sample size (rows)", 50, 500, 200, 25, key="shap_sample_n_tab4")
    plot_type = st.selectbox("Plot type", ["bar", "dot"], index=0)

    if st.button("Compute SHAP summary", type="primary"):
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
    st.header("Data explorer (processed dataset used for modelling)", divider="gray")
    show_cols = ["country", "year", "crisisJST", "target"] + all_features
    st.dataframe(df_target[show_cols].sort_values(["country","year"]).tail(150), use_container_width=True)

    with st.expander("Show feature lists"):
        st.write("**Base (continuous) features**:", base_features)
        st.write("**Missing flags**:", missing_features)
        st.write("**All features (model input order)**:", all_features)
