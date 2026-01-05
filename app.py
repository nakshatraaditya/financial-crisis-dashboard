# ======================================================================
#  FINANCIAL CRISIS EARLY WARNING SYSTEM – STREAMLIT DASHBOARD (FULL)
#
#  VERSION: Upload REMOVED (no upload tab, no sidebar uploader)
#
#  INCLUDED:
#   ✅ Robust multi-model pipeline (missing flags + proper scaling)
#   ✅ Works on JST advanced economies subset (you can change country list)
#   ✅ Interactive RGB country selection across charts (Altair if available)
#   ✅ Crisis Risk tab with crisis overlays (bands)
#   ✅ GDP + Real House Price charts from JST (gdp, hpnom/cpi)
#   ✅ SHAP interactive pie + explanation directly under it (Plotly if available)
#   ✅ Model Results tab with validation comparison + test metrics + confusion matrix
#   ✅ Data Explorer tab
#
#  NOTES:
#   - If Plotly isn't installed on Streamlit Cloud, pie will fallback to Matplotlib.
#   - If Altair isn't installed, charts fallback to st.line_chart (no overlays/colors).
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt

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

# -----------------------------------------------------------------------------
# STREAMLIT CONFIG
st.set_page_config(page_title="Financial Crisis EWS Dashboard", page_icon="📉", layout="wide")

APP_DIR = Path(__file__).parent
JST_XLSX = APP_DIR / "JSTdatasetR6.xlsx"

# -----------------------------------------------------------------------------
# RGB color settings (fixed for key countries, deterministic for others)
BASE_COUNTRY_ORDER = ["USA", "UK", "Canada", "Australia", "Ireland"]
BASE_COUNTRY_RGB = {
    "USA": "rgb(255, 99, 71)",      # tomato
    "UK": "rgb(100, 149, 237)",     # cornflower blue
    "Canada": "rgb(60, 179, 113)",  # medium sea green
    "Australia": "rgb(255, 165, 0)",# orange
    "Ireland": "rgb(138, 43, 226)", # blueviolet
}

# -----------------------------------------------------------------------------
# HEADER
st.title("📉 Financial Crisis Early Warning System (EWS)")
st.caption(
    "Multi-model pipeline (missing flags + proper scaling + validation selection) "
    "with risk timeline, macro context (GDP, house prices), and SHAP explainability."
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
    """Deterministic RGB for any unseen country label."""
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

def line_chart_simple(df, x_col, y_col, color_col, title):
    if df.empty:
        st.info("No data to plot.")
        return
    if not ALTAIR_OK:
        st.line_chart(df, x=x_col, y=y_col, color=color_col)
        return
    line, _, _ = _altair_base_line(df, x_col, y_col, color_col)
    st.altair_chart(line.properties(height=320, title=title).interactive(), use_container_width=True)

# ======================================================================
# JST Macro series (GDP + House prices)
# ======================================================================

@st.cache_data
def get_jst_macro_series(jst_path: str, countries: list[str]) -> pd.DataFrame:
    df = pd.read_excel(jst_path)
    df["country"] = df["country"].astype(str).str.strip()
    df = df[df["country"].isin(countries)].copy()
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

def load_data(file: str, countries: list[str]):
    df = pd.read_excel(file)
    df["country"] = df["country"].astype(str).str.strip()
    df = df[df["country"].isin(countries)].copy()
    df = df.sort_values(["country", "year"])
    return df

def engineer_features(df):
    df = df.copy()

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

    # Universal term spread
    df["yield_curve"] = df["ltrate"] - df["stir"]

    df["money_gdp"] = df["money"] / (df["gdp"] + 1e-9)
    df["money_expansion"] = df.groupby("country")["money_gdp"].pct_change()

    df["ca_gdp"] = df["ca"] / (df["gdp"] + 1e-9)

    base_features = [
        "housing_bubble", "credit_growth", "banking_fragility",
        "yield_curve", "money_expansion", "ca_gdp"
    ]

    df = df[["country", "year", "crisisJST"] + base_features].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df, base_features

def clean_data(df, base_features):
    df = df.copy()
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
        "SVM (RBF)": SVC(kernel="rbf", probability=True, C=2.0, class_weight="balanced"),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
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
def train_pipeline(jst_path: str, countries: list[str]):
    df_raw = load_data(jst_path, countries)

    df_feat, base_features = engineer_features(df_raw)
    df_clean = clean_data(df_feat, base_features)
    df_target = create_target(df_clean).reset_index(drop=True)

    missing_features = [f"{f}_missing" for f in base_features]
    all_features = base_features + missing_features

    # Time split across panel
    train = df_target[df_target["year"] < 1970]
    val   = df_target[(df_target["year"] >= 1970) & (df_target["year"] < 1990)]
    test  = df_target[df_target["year"] >= 1990]

    X_train = train[all_features]
    X_val   = val[all_features]
    X_test  = test[all_features]

    y_train = train["target"]
    y_val   = val["target"]
    y_test  = test["target"]

    # Scale continuous only
    scaler = StandardScaler()
    Xs_train_cont = scaler.fit_transform(X_train[base_features])
    Xs_val_cont   = scaler.transform(X_val[base_features])
    Xs_test_cont  = scaler.transform(X_test[base_features])

    Xs_train = np.hstack([Xs_train_cont, X_train[missing_features].values])
    Xs_val   = np.hstack([Xs_val_cont,   X_val[missing_features].values])
    Xs_test  = np.hstack([Xs_test_cont,  X_test[missing_features].values])

    # Train & evaluate
    results = []
    fitted_models = {}

    for name, clf in build_model_set().items():
        res = evaluate_model(name, clf, Xs_train, y_train, Xs_val, y_val)
        fitted_models[name] = res["clf"]
        results.append({k: v for k, v in res.items() if k != "clf"})  # Arrow-safe

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)

    best_name = results_df.loc[0, "model"]
    best_thresh = float(results_df.loc[0, "threshold"])

    # Test probs for all models
    test_probs_by_model = {name: m.predict_proba(Xs_test)[:, 1] for name, m in fitted_models.items()}

    # Full scaled matrix aligned to df_target
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
    }

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
# MAIN
# ======================================================================

if not JST_XLSX.exists():
    st.error(f"Missing JST dataset file: {JST_XLSX.name}. Put it next to app.py.")
    st.stop()

# Choose your default dashboard countries (change if you want)
DEFAULT_COUNTRIES = ["USA", "UK", "Canada"]

# Train on all 18 advanced economies? or subset?
# If you want "robust all advanced economies", replace this list with the 18.
TRAIN_COUNTRIES = [
    "Australia", "Belgium", "Canada", "Denmark", "Finland", "France",
    "Germany", "Ireland", "Italy", "Japan", "Netherlands", "Norway",
    "Portugal", "Spain", "Sweden", "Switzerland", "UK", "USA"
]

# Sidebar controls
st.sidebar.header("🎛️ Controls")

# Countries selection used in charts
country_options = TRAIN_COUNTRIES
selected_countries = st.sidebar.multiselect(
    "Countries (charts)",
    options=country_options,
    default=[c for c in DEFAULT_COUNTRIES if c in country_options],
)

if not selected_countries:
    st.sidebar.warning("Select at least one country.")
    st.stop()

# Train pipeline (cached)
bundle = train_pipeline(str(JST_XLSX), TRAIN_COUNTRIES)

df_target = bundle["df_target"]
base_features = bundle["base_features"]
missing_features = bundle["missing_features"]
all_features = bundle["all_features"]
models = bundle["models"]
results_df = bundle["results_df"]
best_name = bundle["best_name"]
best_thresh = bundle["best_thresh"]
Xs_test = bundle["Xs_test"]
y_test = bundle["y_test"]
X_full_scaled = bundle["X_full_scaled"]
test_probs_by_model = bundle["test_probs_by_model"]

model_names = list(models.keys())
default_index = model_names.index(best_name) if best_name in model_names else 0
model_choice = st.sidebar.selectbox("Model", options=model_names, index=default_index)
selected_model = models[model_choice]

threshold = st.sidebar.slider("Risk threshold", 0.05, 0.90, float(best_thresh), 0.01)

# Year range
min_y = int(df_target["year"].min())
max_y = int(df_target["year"].max())
from_year, to_year = st.sidebar.slider("Year range", min_y, max_y, (1900, max_y))

# SHAP sample size
shap_pie_sample_n = st.sidebar.slider("SHAP sample size", 50, 400, 150, 25)

# Macro data
macro_df = get_jst_macro_series(str(JST_XLSX), TRAIN_COUNTRIES)
macro_df = macro_df[
    (macro_df["country"].isin(selected_countries)) &
    (macro_df["year"] >= from_year) &
    (macro_df["year"] <= to_year)
].copy()

# Risk data
risk_df = df_target[
    (df_target["country"].isin(selected_countries)) &
    (df_target["year"] >= from_year) &
    (df_target["year"] <= to_year)
].copy()

if risk_df.empty:
    st.warning("No risk data for selected filters.")
    st.stop()

scaled = X_full_scaled[risk_df.index.to_numpy()]
risk_df["crisis_prob"] = selected_model.predict_proba(scaled)[:, 1]

crisis_years_df = risk_df[risk_df["crisisJST"] == 1][["country", "year"]].drop_duplicates()

# ======================================================================
# TABS
# ======================================================================

tab1, tab2, tab3 = st.tabs([
    "📈 Crisis Risk",
    "📊 Model Results",
    "📂 Data Explorer",
])

# -----------------------------------------------------------------------------
# TAB 1: Risk + GDP + House Prices + SHAP pie
with tab1:
    st.header("Crisis risk over time", divider="gray")

    line_chart_training_with_bands(
        risk_df.sort_values(["country", "year"]),
        x_col="year",
        y_col="crisis_prob",
        color_col="country",
        title="Predicted crisis probability (shaded bands = crisis years in JST)",
        crisis_df=crisis_years_df,
    )

    st.write("")
    st.subheader(f"Crisis summary ({to_year})", divider="gray")
    latest = risk_df[risk_df["year"] == to_year]
    cols = st.columns(min(4, max(1, len(selected_countries))))
    for i, c in enumerate(selected_countries):
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

    # ---- LEFT: GDP + House Prices
    with left:
        st.markdown("### 🌍 GDP (from JST)")
        if "gdp" in macro_df.columns and not macro_df["gdp"].isna().all():
            line_chart_simple(
                macro_df.sort_values(["country", "year"]),
                x_col="year",
                y_col="gdp",
                color_col="country",
                title="GDP",
            )
        else:
            st.info("GDP not available for the selected countries/years.")

        st.write("")
        st.markdown("### 🏠 Real house prices (hpnom / cpi)")
        if "house_price_real" in macro_df.columns and not macro_df["house_price_real"].isna().all():
            line_chart_simple(
                macro_df.sort_values(["country", "year"]),
                x_col="year",
                y_col="house_price_real",
                color_col="country",
                title="Real house prices",
            )
        else:
            st.info("House price data not available for the selected countries/years.")

    # ---- RIGHT: SHAP pie + explanation under it
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
                    top,
                    names="Feature",
                    values="Share",
                    hover_data={"Share_%": True, "Share": True},
                    title="SHAP Feature Impact Share (mean |SHAP|)"
                )
                fig.update_layout(
                    height=520,
                    margin=dict(l=10, r=10, t=60, b=20),
                    showlegend=False
                )
                fig.update_traces(textposition="outside", textinfo="percent+label", textfont_size=14)
                st.plotly_chart(fig, use_container_width=True)
            else:
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
# TAB 3: Data Explorer
with tab3:
    st.header("Data explorer (processed dataset used for modelling)", divider="gray")
    show_cols = ["country", "year", "crisisJST", "target"] + all_features
    st.dataframe(df_target[show_cols].sort_values(["country", "year"]).tail(180), use_container_width=True)

    with st.expander("Show feature lists"):
        st.write("**Base (continuous) features**:", base_features)
        st.write("**Missing flags**:", missing_features)
        st.write("**All features (model input order)**:", all_features)
