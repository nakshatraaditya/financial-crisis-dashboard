# ======================================================================
#  FINANCIAL CRISIS EARLY WARNING SYSTEM – STREAMLIT DASHBOARD (FULL)
#  (UPLOAD REMOVED COMPLETELY)
#
#  ✅ No sidebar upload / no upload tab
#  ✅ Interactive RGB country charts (Altair if available; st.line_chart fallback)
#  ✅ Crisis Risk tab includes:
#       - Risk timeline with crisis-year shading (JST crisisJST)
#       - Crisis summary as PERCENTAGE
#       - GDP + Real House Price charts (from JST)
#       - SHAP pie chart + explanation
#       - NEW: Top-6 SHAP bar chart + explanations (Tab 1)
#  ✅ Model Results tab (Arrow-safe results_df)
#  ✅ SHAP detailed tab (summary plot)
#  ✅ Data Explorer tab
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt

# Optional Altair (interactive + RGB + shading + pie)
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
# RGB color settings
COUNTRY_ORDER = ["USA", "UK", "Canada"]
COUNTRY_RGB = {
    "USA": "rgb(255, 99, 71)",      # tomato
    "UK": "rgb(100, 149, 237)",     # cornflower blue
    "Canada": "rgb(60, 179, 113)",  # medium sea green
}

def stable_rgb(name: str) -> str:
    h = abs(hash(name))
    r = 60 + (h % 140)
    g = 60 + ((h // 140) % 140)
    b = 60 + ((h // (140 * 140)) % 140)
    return f"rgb({r},{g},{b})"

def build_color_scale(categories):
    cats = [str(c) for c in categories]
    domain = []
    for c in COUNTRY_ORDER:
        if c in cats and c not in domain:
            domain.append(c)
    for c in sorted(set(cats)):
        if c not in domain:
            domain.append(c)
    range_colors = [COUNTRY_RGB.get(c, stable_rgb(c)) for c in domain]
    return domain, range_colors

# -----------------------------------------------------------------------------
# SHAP feature explanations (used for Top-6 explanation in Tab 1)
FEATURE_EXPLANATIONS = {
    "housing_bubble": "Deviation of real house prices from a 10-year rolling trend (proxy for housing overvaluation).",
    "credit_growth": "Growth in real bank credit (tloans/cpi). Rapid increases often signal overheating/leverage build-up.",
    "banking_fragility": "Composite fragility index from expanding z-scores of noncore funding, LTD, and leverage risk.",
    "sovereign_spread": "Long-term rate minus USA long-term rate (proxy for risk premium / differential vs US benchmark).",
    "yield_curve": "Term spread (ltrate − stir). Flattening/inversion can reflect tighter conditions and recession risk.",
    "money_expansion": "Change in money-to-GDP ratio (proxy for liquidity expansion / credit conditions).",
    "ca_gdp": "Current account balance scaled by GDP (external imbalances can amplify crisis risk).",
}

def explain_feature(name: str) -> str:
    if name.endswith("_missing"):
        base = name.replace("_missing", "")
        base_desc = FEATURE_EXPLANATIONS.get(base, "Base feature missingness indicator.")
        return f"Missing-value flag for `{base}` (1 if missing, else 0). Captures information in data gaps. Base meaning: {base_desc}"
    return FEATURE_EXPLANATIONS.get(name, "Engineered macro-financial indicator used by the model.")

# -----------------------------------------------------------------------------
# HEADER
st.title("📉 Financial Crisis Early Warning System (EWS)")
st.caption("USA · UK · Canada | JST dataset | Multi-model pipeline + SHAP explainability")
st.write("")

with st.sidebar:
    if st.button("Clear cache & rerun"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# ======================================================================
# DATA + PIPELINE FUNCTIONS (Your multi-model pipeline)
# ======================================================================

@st.cache_data
def load_data(jst_path: str):
    df = pd.read_excel(jst_path)
    df["country"] = df["country"].astype(str).str.strip()
    df = df[df["country"].isin(["USA", "UK", "Canada"])].copy()
    df = df.sort_values(["country", "year"])
    return df

def engineer_features(df):
    df = df.copy()

    # ensure crisisJST exists
    if "crisisJST" not in df.columns:
        df["crisisJST"] = 0

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

    df["hp_real"] = df["hpnom"] / (df["cpi"] + 1e-9)
    df["hp_trend"] = df.groupby("country")["hp_real"].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )
    df["housing_bubble"] = (df["hp_real"] - df["hp_trend"]) / (df["hp_trend"] + 1e-9)

    df["real_credit"] = df["tloans"] / (df["cpi"] + 1e-9)
    df["credit_growth"] = df.groupby("country")["real_credit"].pct_change()

    df["yield_curve"] = df["ltrate"] - df["stir"]

    # Sovereign spread vs USA long rate (works for USA/UK/Canada view)
    us_ltrate = (
        df[df["country"] == "USA"]
        .drop_duplicates("year")
        .set_index("year")["ltrate"]
        .to_dict()
    )
    df["us_ltrate"] = df["year"].map(us_ltrate)
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

    # War periods removed
    df = df[~df["year"].between(1914, 1918)]
    df = df[~df["year"].between(1939, 1945)]

    # Missing flags
    for col in base_features:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # Causal fill within country
    df[base_features] = df.groupby("country")[base_features].transform(
        lambda x: x.ffill(limit=3).bfill(limit=3)
    )

    # Median fallback within country
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

    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.90, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)

    return {
        "model": name,
        "ROC-AUC": float(roc_auc_score(y_val, probs)) if len(np.unique(y_val)) > 1 else np.nan,
        "PR-AUC": float(average_precision_score(y_val, probs)) if len(np.unique(y_val)) > 1 else np.nan,
        "F1": float(best_f1),
        "threshold": float(best_t),
        "clf": model
    }

# ======================================================================
# Train once (cached)
# ======================================================================

@st.cache_resource
def train_pipeline(jst_path: str):
    df_raw = load_data(jst_path)

    # macro series for GDP + real house price charts (from JST raw)
    macro = df_raw[["country", "year", "gdp", "hpnom", "cpi", "crisisJST"]].copy()
    macro["house_price_real"] = macro["hpnom"] / (macro["cpi"] + 1e-9)
    macro = macro.sort_values(["country", "year"])

    df_feat, base_features = engineer_features(df_raw)
    df_clean = clean_data(df_feat, base_features)
    df_target = create_target(df_clean).reset_index(drop=True)

    missing_features = [f"{f}_missing" for f in base_features]
    all_features = base_features + missing_features

    # Train/Val/Test time split
    train = df_target[df_target["year"] < 1970]
    val   = df_target[(df_target["year"] >= 1970) & (df_target["year"] < 1990)]
    test  = df_target[df_target["year"] >= 1990]

    X_train = train[all_features]
    X_val   = val[all_features]
    X_test  = test[all_features]

    y_train = train["target"]
    y_val   = val["target"]
    y_test  = test["target"]

    # Proper scaling: continuous only
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

    test_probs_by_model = {name: m.predict_proba(Xs_test)[:, 1] for name, m in fitted_models.items()}

    X_full = df_target[all_features]
    X_full_cont = scaler.transform(X_full[base_features])
    X_full_scaled = np.hstack([X_full_cont, X_full[missing_features].values])

    return {
        "df_target": df_target,
        "macro": macro,
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
# Chart helpers
# ======================================================================

def altair_line(df, x_col, y_col, color_col, title, height=320, y_format=None):
    domain, range_colors = build_color_scale(df[color_col].unique())
    tooltip = [
        alt.Tooltip(f"{color_col}:N"),
        alt.Tooltip(f"{x_col}:Q"),
        alt.Tooltip(f"{y_col}:Q", format=y_format if y_format else ".6f"),
    ]
    base = alt.Chart(df).encode(
        x=alt.X(f"{x_col}:Q", title=x_col),
        y=alt.Y(f"{y_col}:Q", title=y_col),
        color=alt.Color(
            f"{color_col}:N",
            scale=alt.Scale(domain=domain, range=range_colors),
            legend=alt.Legend(title=color_col),
        ),
        tooltip=tooltip,
    )
    return base.mark_line().properties(height=height, title=title).interactive()

def altair_risk_with_crisis_bands(risk_df, crisis_df, title, threshold=None):
    domain, range_colors = build_color_scale(risk_df["country"].unique())

    line = alt.Chart(risk_df).mark_line().encode(
        x=alt.X("year:Q", title="year"),
        y=alt.Y("crisis_prob:Q", title="crisis probability"),
        color=alt.Color("country:N", scale=alt.Scale(domain=domain, range=range_colors), legend=alt.Legend(title="country")),
        tooltip=["country:N", "year:Q", alt.Tooltip("crisis_prob:Q", format=".3f")],
    )

    layers = []

    if crisis_df is not None and not crisis_df.empty:
        cdf = crisis_df.copy()
        cdf["x_start"] = cdf["year"] - 0.5
        cdf["x_end"] = cdf["year"] + 0.5
        rect = alt.Chart(cdf).mark_rect(opacity=0.18).encode(
            x="x_start:Q", x2="x_end:Q",
            y=alt.value(0), y2=alt.value(1),
            color=alt.Color("country:N", scale=alt.Scale(domain=domain, range=range_colors), legend=None),
        )
        layers.append(rect)

    layers.append(line)

    if threshold is not None:
        rule = alt.Chart(pd.DataFrame({"y":[float(threshold)]})).mark_rule(strokeDash=[6,4], opacity=0.95).encode(y="y:Q")
        layers.append(rule)

    return alt.layer(*layers).properties(height=320, title=title).interactive()

def compute_shap_global(selected_model, Xs_test, feature_names, sample_n=150):
    """
    Returns a dataframe with:
      - mean_abs (mean |SHAP|)
      - mean_signed (mean SHAP)
      - share (mean_abs / sum(mean_abs))
    """
    X_test_shap = pd.DataFrame(Xs_test, columns=feature_names)
    Xsamp = X_test_shap.sample(n=min(sample_n, len(X_test_shap)), random_state=42)

    explainer = shap.Explainer(selected_model, X_test_shap)
    shap_vals = explainer(Xsamp)

    values = shap_vals.values
    if values.ndim == 3:  # (n, features, classes) -> use class 1
        values = values[:, :, 1]

    mean_abs = np.mean(np.abs(values), axis=0)
    mean_signed = np.mean(values, axis=0)
    total = float(mean_abs.sum()) + 1e-12
    share = mean_abs / total

    out = pd.DataFrame({
        "Feature": feature_names,
        "mean_abs": mean_abs,
        "mean_signed": mean_signed,
        "share": share
    }).sort_values("mean_abs", ascending=False).reset_index(drop=True)

    return out

def render_shap_pie_altair(global_df, top_k=8, title="SHAP Feature Impact Share (mean |SHAP|)"):
    df = global_df[["Feature", "share"]].copy()
    df = df.sort_values("share", ascending=False).reset_index(drop=True)

    top = df.head(top_k).copy()
    other = float(df["share"].iloc[top_k:].sum()) if len(df) > top_k else 0.0
    if other > 0:
        top = pd.concat([top, pd.DataFrame([{"Feature": "Other", "share": other}])], ignore_index=True)

    top["share_pct"] = top["share"] * 100

    pie = alt.Chart(top).mark_arc(innerRadius=40).encode(
        theta=alt.Theta("share:Q"),
        color=alt.Color("Feature:N", legend=None),
        tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("share_pct:Q", format=".1f", title="Share (%)")],
    ).properties(height=420, title=title).interactive()

    # Add labels for larger slices
    labels = alt.Chart(top[top["share_pct"] >= 6]).mark_text(radius=170, size=12).encode(
        theta=alt.Theta("share:Q"),
        text=alt.Text("Feature:N")
    )

    st.altair_chart(pie + labels, use_container_width=True)

def render_top6_bar(global_df, top_n=6, title="Top 6 drivers (mean |SHAP|)"):
    top = global_df.head(top_n).copy()
    top = top.iloc[::-1]  # show biggest at bottom for readability

    if ALTAIR_OK:
        bar = alt.Chart(top).mark_bar().encode(
            x=alt.X("mean_abs:Q", title="mean(|SHAP|)"),
            y=alt.Y("Feature:N", sort=None, title=None),
            tooltip=[
                alt.Tooltip("Feature:N"),
                alt.Tooltip("mean_abs:Q", format=".6f", title="mean(|SHAP|)"),
                alt.Tooltip("mean_signed:Q", format=".6f", title="mean(SHAP)"),
                alt.Tooltip("share:Q", format=".3%", title="share"),
            ],
        ).properties(height=260, title=title).interactive()
        st.altair_chart(bar, use_container_width=True)
    else:
        st.bar_chart(top.set_index("Feature")["mean_abs"])

def direction_arrow(mean_signed: float) -> str:
    # positive -> increases risk, negative -> decreases risk
    if mean_signed > 0:
        return "↑ increases risk (on average)"
    if mean_signed < 0:
        return "↓ decreases risk (on average)"
    return "≈ neutral (on average)"

# ======================================================================
# LOAD EVERYTHING
# ======================================================================

if not JST_XLSX.exists():
    st.error(f"Missing JST dataset file: {JST_XLSX.name}. Put it next to app.py.")
    st.stop()

bundle = train_pipeline(str(JST_XLSX))

df_target = bundle["df_target"]
macro = bundle["macro"]
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

# ======================================================================
# SIDEBAR CONTROLS (NO UPLOAD)
# ======================================================================

st.sidebar.header("🎛️ Model & Filters")

model_names = list(models.keys())
default_index = model_names.index(best_name) if best_name in model_names else 0

model_choice = st.sidebar.selectbox("Select model", options=model_names, index=default_index)
selected_model = models[model_choice]

threshold = st.sidebar.slider("Risk threshold", 0.05, 0.90, float(best_thresh), 0.01)

risk_countries = st.sidebar.multiselect(
    "Countries",
    ["USA", "UK", "Canada"],
    default=["USA", "UK", "Canada"]
)

min_year = int(df_target["year"].min())
max_year = int(df_target["year"].max())

from_year, to_year = st.sidebar.slider(
    "Year range",
    min_value=min_year,
    max_value=max_year,
    value=(1900, max_year)
)

show_crisis_years_list = st.sidebar.checkbox("Show crisis years list", value=True)

st.sidebar.subheader("🧠 SHAP (Tab 1)")
shap_sample_n = st.sidebar.slider("SHAP sample size", 50, 400, 150, 25)

# ======================================================================
# PREPARE RISK DF
# ======================================================================

risk_df = df_target[
    (df_target["country"].isin(risk_countries)) &
    (df_target["year"] >= from_year) &
    (df_target["year"] <= to_year)
].copy()

risk_scaled = X_full_scaled[risk_df.index.to_numpy()]
risk_df["crisis_prob"] = selected_model.predict_proba(risk_scaled)[:, 1]

crisis_years_df = (
    risk_df[risk_df["crisisJST"] == 1][["country", "year"]]
    .drop_duplicates()
    .copy()
)

macro_df = macro[
    (macro["country"].isin(risk_countries)) &
    (macro["year"] >= from_year) &
    (macro["year"] <= to_year)
].copy()

# ======================================================================
# TABS
# ======================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Crisis Risk",
    "📊 Model Results",
    "🧠 SHAP (Detailed)",
    "📂 Data Explorer"
])

# -----------------------------------------------------------------------------
# TAB 1: Crisis risk + GDP + SHAP pie + Top-6 SHAP + explanations
with tab1:
    st.header("Crisis risk over time", divider="gray")

    if risk_df.empty:
        st.warning("No data for the selected filters.")
    else:
        if ALTAIR_OK:
            chart = altair_risk_with_crisis_bands(
                risk_df.sort_values(["country", "year"]),
                crisis_years_df,
                title="Predicted crisis probability (shaded = JST crisis years)",
                threshold=threshold
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.line_chart(risk_df, x="year", y="crisis_prob", color="country")
            st.info("Install `altair` to enable RGB colors + crisis shading + interactive tooltips.")

        st.write("")
        st.subheader(f"Crisis summary ({to_year})", divider="gray")

        latest = risk_df[risk_df["year"] == to_year]
        cols = st.columns(3)

        for i, c in enumerate(["USA", "UK", "Canada"]):
            with cols[i]:
                if c not in risk_countries:
                    st.metric(label=f"{c} risk", value="(not selected)")
                    continue
                val = latest[latest["country"] == c]["crisis_prob"].mean()
                if pd.isna(val):
                    st.metric(label=f"{c} crisis risk", value="n/a")
                else:
                    val_pct = 100 * float(val)
                    thr_pct = 100 * float(threshold)
                    st.metric(
                        label=f"{c} crisis risk",
                        value=f"{val_pct:.1f}%",
                        delta=f"{(val_pct - thr_pct):+.1f} pp vs threshold",
                        delta_color="inverse" if val >= threshold else "normal",
                    )

        if show_crisis_years_list:
            with st.expander("Crisis years in selected window"):
                for c in risk_countries:
                    years = crisis_years_df[crisis_years_df["country"] == c]["year"].tolist()
                    st.write(f"**{c}:** " + (", ".join(map(str, years)) if years else "none"))

    st.write("")
    st.subheader("Macro context + explainability", divider="gray")

    left, right = st.columns(2)

    with left:
        st.markdown("### 🌍 GDP")
        if not macro_df.empty and "gdp" in macro_df.columns and not macro_df["gdp"].isna().all():
            if ALTAIR_OK:
                st.altair_chart(
                    altair_line(
                        macro_df.sort_values(["country", "year"]),
                        x_col="year",
                        y_col="gdp",
                        color_col="country",
                        title="GDP (JST)",
                        y_format=",.0f"
                    ),
                    use_container_width=True
                )
            else:
                st.line_chart(macro_df, x="year", y="gdp", color="country")
        else:
            st.info("GDP not available for selected range.")

        st.write("")
        st.markdown("### 🏠 Real house prices (hpnom / cpi)")
        if not macro_df.empty and "house_price_real" in macro_df.columns and not macro_df["house_price_real"].isna().all():
            if ALTAIR_OK:
                st.altair_chart(
                    altair_line(
                        macro_df.sort_values(["country", "year"]),
                        x_col="year",
                        y_col="house_price_real",
                        color_col="country",
                        title="Real house prices (JST)"
                    ),
                    use_container_width=True
                )
            else:
                st.line_chart(macro_df, x="year", y="house_price_real", color="country")
        else:
            st.info("House price data not available for selected range.")

    with right:
        st.markdown("### 🧠 SHAP Feature Importance")
        st.caption("SHAP allocates the prediction (relative to a baseline) across features. Larger **|SHAP|** = bigger influence.")

        # Compute SHAP once and reuse (pie + top6 + explanations)
        try:
            global_shap = compute_shap_global(
                selected_model=selected_model,
                Xs_test=Xs_test,
                feature_names=all_features,
                sample_n=shap_sample_n
            )

            # --- Pie chart (interactive if Altair available)
            st.markdown("#### Pie: global impact share (mean |SHAP|)")
            st.caption("This pie shows **share of total mean(|SHAP|)** across sampled test rows (global importance).")
            if ALTAIR_OK:
                render_shap_pie_altair(global_shap, top_k=8, title="SHAP Feature Impact Share (mean |SHAP|)")
            else:
                # fallback pie
                top = global_shap.head(8).copy()
                other = float(global_shap["share"].iloc[8:].sum()) if len(global_shap) > 8 else 0.0
                if other > 0:
                    top = pd.concat([top, pd.DataFrame([{"Feature": "Other", "mean_abs": np.nan, "mean_signed": 0.0, "share": other}])], ignore_index=True)
                plt.figure(figsize=(7, 6))
                plt.pie(top["share"], labels=top["Feature"], autopct=lambda p: f"{p:.1f}%" if p >= 4 else "", startangle=90)
                plt.title("SHAP Feature Impact Share (mean |SHAP|)")
                st.pyplot(plt.gcf(), clear_figure=True)
                st.info("Install `altair` for an interactive pie chart.")

            st.write("")
            st.markdown("#### Top 6 SHAP drivers (graph + explanation)")
            render_top6_bar(global_shap, top_n=6, title="Top 6 drivers (mean |SHAP|)")

            # Explanation right under the chart (as you asked)
            top6 = global_shap.head(6).copy()
            st.markdown("**What the top 6 represent**")
            st.markdown(
                "- The bar chart ranks features by **mean(|SHAP|)** across the sampled rows.\n"
                "- It answers: *which variables influence the model the most overall?*\n"
                "- The direction note uses **mean(SHAP)** (average sign)."
            )

            for _, row in top6.iterrows():
                feat = str(row["Feature"])
                dir_note = direction_arrow(float(row["mean_signed"]))
                st.markdown(
                    f"- **{feat}** — {dir_note}.  \n"
                    f"  *Meaning:* {explain_feature(feat)}"
                )

            with st.expander("Show SHAP importance table (top 25)"):
                tmp = global_shap.copy()
                tmp["share_pct"] = tmp["share"] * 100
                st.dataframe(tmp.head(25), use_container_width=True)

        except Exception:
            st.warning("SHAP could not be computed for this model. Try Gradient Boosting / Random Forest, or reduce SHAP sample size.")

# -----------------------------------------------------------------------------
# TAB 2: Model results
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
    st.dataframe(
        pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]),
        use_container_width=True
    )

# -----------------------------------------------------------------------------
# TAB 3: SHAP detailed
with tab3:
    st.header("SHAP explainability (detailed)", divider="gray")
    st.markdown(
        """
SHAP explains predictions by distributing the difference between the prediction and a baseline across features.

- **Positive SHAP** increases predicted crisis risk.
- **Negative SHAP** decreases predicted crisis risk.
- **Mean absolute SHAP** gives global importance (what the Tab 1 pie and Top-6 chart summarise).
"""
    )

    sample_n = st.slider("Sample size (rows)", 50, 500, 200, 25, key="shap_sample_n_tab3")
    plot_type = st.selectbox("Plot type", ["bar", "dot"], index=0, key="shap_plot_type_tab3")

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
# TAB 4: Data explorer
with tab4:
    st.header("Data explorer (processed dataset used for modelling)", divider="gray")
    show_cols = ["country", "year", "crisisJST", "target"] + all_features
    st.dataframe(
        df_target[show_cols].sort_values(["country", "year"]).tail(150),
        use_container_width=True
    )

    with st.expander("Show feature lists"):
        st.write("**Base (continuous) features**:", base_features)
        st.write("**Missing flags**:", missing_features)
        st.write("**All features (model input order)**:", all_features)
