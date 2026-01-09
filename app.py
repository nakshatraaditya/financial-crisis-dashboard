# ======================================================================
#  FINANCIAL CRISIS EARLY WARNING SYSTEM – STREAMLIT DASHBOARD (FULL)
#  + LLM POWERED CHATBOT "POPUP" (st.popover) that stays available across tabs
#
#  ✅ No upload UI anywhere
#  ✅ Chatbot opens as a popup (popover) and uses OpenAI Chat Completions API
#  ✅ Chat remembers conversation in st.session_state
#  ✅ Chat gets LIVE dashboard context (model, threshold, metrics, top SHAP)
#
#  IMPORTANT:
#  - Put JSTdatasetR6.xlsx next to this app.py
#  - Add OPENAI_API_KEY in Streamlit Secrets OR environment variable
# ======================================================================

import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional Altair for interactive RGB + shading + interactive pie
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

import requests

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

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
# SHAP feature explanations
FEATURE_EXPLANATIONS = {
    "housing_bubble": "Deviation of real house prices from a 10-year rolling trend (housing overvaluation proxy).",
    "credit_growth": "Growth in real bank credit (tloans/cpi). Rapid increases can signal overheating/leverage.",
    "banking_fragility": "Composite fragility index from noncore funding, LTD, and leverage-risk z-scores.",
    "sovereign_spread": "Long-term rate minus USA long-term rate (risk premium differential vs US benchmark).",
    "yield_curve": "Term spread (ltrate − stir). Flattening/inversion often reflects tighter conditions.",
    "money_expansion": "Change in money-to-GDP ratio (liquidity/credit conditions proxy).",
    "ca_gdp": "Current account balance scaled by GDP (external imbalance proxy).",
}

def explain_feature(name: str) -> str:
    if name.endswith("_missing"):
        base = name.replace("_missing", "")
        base_desc = FEATURE_EXPLANATIONS.get(base, "Base feature meaning.")
        return f"Missing-value flag for `{base}` (1 if missing, else 0). Base meaning: {base_desc}"
    return FEATURE_EXPLANATIONS.get(name, "Engineered macro-financial indicator used by the model.")

def direction_arrow(mean_signed: float) -> str:
    if mean_signed > 0:
        return "↑ increases risk (on average)"
    if mean_signed < 0:
        return "↓ decreases risk (on average)"
    return "≈ neutral (on average)"

# -----------------------------------------------------------------------------
# Behavioural proxy layman explanations (Explain-only tab)
BEHAVIOUR_PROXY_EXPLAIN = {
    "risk_appetite_z": (
        "Risk appetite = how willing investors are to take risks. "
        "When this is high, markets usually feel confident and people prefer risky assets. "
        "When it drops, people often ‘run to safety’, which can be an early warning sign."
    ),
    "market_volatility_z": (
        "Market volatility = how jumpy and unstable markets are. "
        "Higher volatility means prices are swinging more than usual, showing uncertainty and fear. "
        "Sustained spikes can signal stress building up in the financial system."
    ),
    "debt_service_risk_z": (
        "Debt service risk = how hard it is to repay debt when debt levels and interest rates are high. "
        "If this rises, borrowers may struggle to repay, and defaults become more likely. "
        "That can weaken banks and the wider economy."
    ),
}

OUTCOME_EXPLAIN = {
    "gdp_growth": "GDP growth shows whether the economy is expanding or slowing down.",
    "inflation": "Inflation shows how quickly prices are rising. High inflation can squeeze spending and push interest rates up.",
    "unemp_chg": "Unemployment change shows whether joblessness is worsening or improving.",
    "crisis_prob": "Crisis probability is the model’s estimated risk of a crisis (based on model inputs, not these behavioural proxies).",
}

def explain_proxy_short(proxy_col: str) -> str:
    return BEHAVIOUR_PROXY_EXPLAIN.get(proxy_col, "This proxy summarises a behavioural risk condition in the financial system.")

def explain_outcome_short(out_col: str) -> str:
    return OUTCOME_EXPLAIN.get(out_col, "This is an outcome used to understand how the economy typically behaves when behaviour changes.")

# -----------------------------------------------------------------------------
# HEADER
st.title("📉 Financial Crisis Early Warning System (EWS)")
st.caption("USA · UK · Canada | JST dataset | Multi-model pipeline + SHAP explainability")
st.write("")

with st.sidebar:
    if st.button("Clear cache & rerun"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.pop("chat_messages", None)
        st.session_state.pop("chat_last_error", None)
        st.session_state.pop("chat_api_key", None)
        st.rerun()

# ======================================================================
# DATA + PIPELINE FUNCTIONS
# ======================================================================

@st.cache_data
def load_data(jst_path: str):
    df = pd.read_excel(jst_path)
    df["country"] = df["country"].astype(str).str.strip()
    df = df[df["country"].isin(["USA", "UK", "Canada"])].copy()
    df = df.sort_values(["country", "year"])
    return df

def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series(np.nan, index=df.index)

def engineer_features(df):
    df = df.copy()

    if "crisisJST" not in df.columns:
        df["crisisJST"] = 0

    lev = _safe_series(df, "lev").astype(float)
    df["leverage_risk"] = 1 / (lev + 0.01)

    def expanding_z(df_inner, col):
        def z(s):
            mu = s.expanding().mean()
            sd = s.expanding().std().replace(0, np.nan)
            return (s - mu) / (sd + 1e-9)
        return df_inner.groupby("country")[col].transform(z)

    df["noncore_z"]  = expanding_z(df, "noncore") if "noncore" in df.columns else np.nan
    df["ltd_z"]      = expanding_z(df, "ltd") if "ltd" in df.columns else np.nan
    df["leverage_z"] = expanding_z(df, "leverage_risk")

    df["banking_fragility"] = (
        0.4 * df["noncore_z"] +
        0.3 * df["ltd_z"] +
        0.3 * df["leverage_z"]
    )

    hpnom = _safe_series(df, "hpnom").astype(float)
    cpi   = _safe_series(df, "cpi").astype(float)
    df["hp_real"] = hpnom / (cpi + 1e-9)
    df["hp_trend"] = df.groupby("country")["hp_real"].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )
    df["housing_bubble"] = (df["hp_real"] - df["hp_trend"]) / (df["hp_trend"] + 1e-9)

    tloans = _safe_series(df, "tloans").astype(float)
    df["real_credit"] = tloans / (cpi + 1e-9)
    df["credit_growth"] = df.groupby("country")["real_credit"].pct_change()

    ltrate = _safe_series(df, "ltrate").astype(float)
    stir   = _safe_series(df, "stir").astype(float)
    df["yield_curve"] = ltrate - stir

    us_ltrate = (
        df[df["country"] == "USA"]
        .drop_duplicates("year")
        .set_index("year")["ltrate"]
        .to_dict()
    ) if "ltrate" in df.columns else {}
    df["us_ltrate"] = df["year"].map(us_ltrate)
    df["sovereign_spread"] = ltrate - df["us_ltrate"].astype(float)

    money = _safe_series(df, "money").astype(float)
    gdp   = _safe_series(df, "gdp").astype(float)
    df["money_gdp"] = money / (gdp + 1e-9)
    df["money_expansion"] = df.groupby("country")["money_gdp"].pct_change()

    ca = _safe_series(df, "ca").astype(float)
    df["ca_gdp"] = ca / (gdp + 1e-9)

    base_features = [
        "housing_bubble", "credit_growth", "banking_fragility",
        "sovereign_spread", "yield_curve",
        "money_expansion", "ca_gdp"
    ]

    df = df[["country", "year", "crisisJST"] + base_features].copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    return df, base_features

def engineer_behavioural_proxies(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Behavioural proxies (EXPLAIN-ONLY, NOT fed into the model):
    - risk_appetite: risky_tr − safe_tr
    - market_volatility: 5y rolling std of Δ risky_tr
    - debt_service_risk: debtgdp × stir (simple stress proxy)
    Plus outcomes for correlation visuals: GDP growth, inflation, unemployment change.
    """
    df = df_raw.copy()
    df["country"] = df["country"].astype(str).str.strip()
    df = df[df["country"].isin(["USA", "UK", "Canada"])].copy()
    df = df.sort_values(["country", "year"])

    if "crisisJST" not in df.columns:
        df["crisisJST"] = 0

    # Outcomes (safe if cols missing)
    df["gdp_growth"] = df.groupby("country")["gdp"].pct_change() if "gdp" in df.columns else np.nan
    df["inflation"]  = df.groupby("country")["cpi"].pct_change() if "cpi" in df.columns else np.nan
    df["unemp_chg"]  = df.groupby("country")["unemp"].diff() if "unemp" in df.columns else np.nan

    risky_tr = _safe_series(df, "risky_tr").astype(float)
    safe_tr  = _safe_series(df, "safe_tr").astype(float)
    stir     = _safe_series(df, "stir").astype(float)
    debtgdp  = _safe_series(df, "debtgdp").astype(float)

    df["risk_appetite"] = risky_tr - safe_tr

    df["risky_tr_chg"] = df.groupby("country")["risky_tr"].diff() if "risky_tr" in df.columns else np.nan
    df["market_volatility"] = (
        df.groupby("country")["risky_tr_chg"]
          .transform(lambda s: s.rolling(5, min_periods=3).std())
        if "risky_tr" in df.columns else np.nan
    )

    df["debt_service_risk"] = debtgdp * stir

    def expanding_z(df_inner, col):
        def z(s):
            mu = s.expanding().mean()
            sd = s.expanding().std().replace(0, np.nan)
            return (s - mu) / (sd + 1e-9)
        return df_inner.groupby("country")[col].transform(z)

    for c in ["risk_appetite", "market_volatility", "debt_service_risk"]:
        df[f"{c}_z"] = expanding_z(df, c)

    keep = [
        "country", "year", "crisisJST",
        "risk_appetite", "market_volatility", "debt_service_risk",
        "risk_appetite_z", "market_volatility_z", "debt_service_risk_z",
        "gdp_growth", "inflation", "unemp_chg"
    ]
    out = df[keep].replace([np.inf, -np.inf], np.nan).copy()
    return out

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
    if all(c in df_raw.columns for c in ["country", "year", "gdp", "hpnom", "cpi", "crisisJST"]):
        macro = df_raw[["country", "year", "gdp", "hpnom", "cpi", "crisisJST"]].copy()
    else:
        macro = df_raw[["country", "year"]].copy()
        macro["gdp"] = _safe_series(df_raw, "gdp")
        macro["hpnom"] = _safe_series(df_raw, "hpnom")
        macro["cpi"] = _safe_series(df_raw, "cpi")
        macro["crisisJST"] = _safe_series(df_raw, "crisisJST")

    macro["house_price_real"] = _safe_series(macro, "hpnom").astype(float) / (_safe_series(macro, "cpi").astype(float) + 1e-9)
    macro = macro.sort_values(["country", "year"])

    # Behavioural (explain-only)
    behavioural = engineer_behavioural_proxies(df_raw)

    df_feat, base_features = engineer_features(df_raw)
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

    test_probs_by_model = {name: m.predict_proba(Xs_test)[:, 1] for name, m in fitted_models.items()}

    X_full = df_target[all_features]
    X_full_cont = scaler.transform(X_full[base_features])
    X_full_scaled = np.hstack([X_full_cont, X_full[missing_features].values])

    return {
        "df_target": df_target,
        "macro": macro,
        "behavioural": behavioural,
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
        rule = alt.Chart(pd.DataFrame({"y": [float(threshold)]})).mark_rule(strokeDash=[6, 4], opacity=0.95).encode(y="y:Q")
        layers.append(rule)

    return alt.layer(*layers).properties(height=320, title=title).interactive()

def compute_shap_global(selected_model, Xs_test, feature_names, sample_n=150):
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

    labels = alt.Chart(top[top["share_pct"] >= 6]).mark_text(radius=170, size=12).encode(
        theta=alt.Theta("share:Q"),
        text=alt.Text("Feature:N")
    )

    st.altair_chart(pie + labels, use_container_width=True)

def render_top6_bar(global_df, top_n=6, title="Top 6 drivers (mean |SHAP|)"):
    top = global_df.head(top_n).copy()
    top = top.iloc[::-1]
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

# ======================================================================
# LOAD EVERYTHING
# ======================================================================

if not JST_XLSX.exists():
    st.error(f"Missing JST dataset file: {JST_XLSX.name}. Put it next to app.py.")
    st.stop()

bundle = train_pipeline(str(JST_XLSX))

df_target = bundle["df_target"]
macro = bundle["macro"]
behavioural = bundle["behavioural"]
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
# SIDEBAR CONTROLS
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
# CHATBOT (POPOVER POPUP)
# ======================================================================

def get_openai_key() -> str | None:
    if st.session_state.get("chat_api_key"):
        return st.session_state["chat_api_key"]
    if "OPENAI_API_KEY" in st.secrets:
        return str(st.secrets["OPENAI_API_KEY"]).strip()
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    return env_key if env_key else None

def build_dashboard_context_text() -> str:
    probs = test_probs_by_model[model_choice]
    preds = (probs >= threshold).astype(int)

    roc = roc_auc_score(y_test, probs)
    pr  = average_precision_score(y_test, probs)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)

    top_shap = None
    try:
        g = compute_shap_global(selected_model, Xs_test, all_features, sample_n=min(150, shap_sample_n))
        top_shap = g.head(6)[["Feature", "share", "mean_signed"]].copy()
    except Exception:
        top_shap = None

    crisis_counts = crisis_years_df.groupby("country")["year"].count().to_dict() if not crisis_years_df.empty else {}

    ctx = []
    ctx.append("DASHBOARD CONTEXT (live):")
    ctx.append(f"- Selected model: {model_choice}")
    ctx.append(f"- Risk threshold: {threshold:.2f}")
    ctx.append(f"- Countries selected: {', '.join(risk_countries) if risk_countries else '(none)'}")
    ctx.append(f"- Year range selected: {from_year} to {to_year}")
    ctx.append(f"- Test metrics (1990+): ROC-AUC={roc:.3f}, PR-AUC={pr:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
    if crisis_counts:
        ctx.append(f"- Crisis years count in current window: {crisis_counts}")
    else:
        ctx.append("- Crisis years count in current window: none or not in selection")
    if top_shap is not None:
        rows = []
        for _, r in top_shap.iterrows():
            direction = "increases risk" if float(r["mean_signed"]) > 0 else "decreases risk" if float(r["mean_signed"]) < 0 else "neutral"
            rows.append(f"{r['Feature']} (share={float(r['share'])*100:.1f}%, avg sign={direction})")
        ctx.append("- Top SHAP drivers (global): " + "; ".join(rows))
    else:
        ctx.append("- Top SHAP drivers (global): unavailable (SHAP failed for this model).")

    ctx.append("Allowed topics: interpreting graphs, metrics, SHAP, crisis years, GDP/house prices, threshold effects, comparing models.")
    ctx.append("If asked something outside this dashboard/data, say so and suggest what to check.")
    return "\n".join(ctx)

def openai_chat_completion(api_key: str, messages: list[dict], model: str = "gpt-4o-mini") -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def init_chat():
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": "Hi! Ask me anything about this dashboard (metrics, SHAP, crisis years, GDP/house prices, threshold effects)."}
        ]

init_chat()

top_row_left, top_row_right = st.columns([0.78, 0.22], gap="small")
with top_row_left:
    st.write("")
with top_row_right:
    with st.popover("💬 Dashboard Chat"):
        st.markdown("### 💬 Chatbot")
        st.caption("LLM-powered assistant. It uses live dashboard context and your question to generate answers.")

        key_in = st.text_input(
            "OpenAI API key (optional if set in Secrets/ENV)",
            type="password",
            value=st.session_state.get("chat_api_key", ""),
            placeholder="sk-...",
            help="Stored only in session_state for this run."
        )
        if key_in.strip():
            st.session_state["chat_api_key"] = key_in.strip()

        api_key = get_openai_key()
        if not api_key:
            st.warning("No API key found. Set OPENAI_API_KEY in Streamlit Secrets or environment variable, or paste above.")

        st.markdown("**Suggested questions:**")
        sugg = st.selectbox(
            "Pick one",
            [
                "Why is PR-AUC different from ROC-AUC here?",
                "Explain what the top 6 SHAP features mean.",
                "What does changing the threshold do to precision/recall?",
                "Why do risk peaks appear before crisis years?",
                "Summarise results in 4 dissertation-ready sentences."
            ],
            index=0
        )
        if st.button("Use suggested question"):
            st.session_state["chat_draft"] = sugg

        st.divider()

        for m in st.session_state["chat_messages"][-12:]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_text = st.chat_input("Ask about the dashboard…")
        if user_text:
            st.session_state["chat_last_error"] = None
            st.session_state["chat_messages"].append({"role": "user", "content": user_text})

            if not api_key:
                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": "I can’t call the LLM yet because there is no API key set. Add OPENAI_API_KEY in Secrets/ENV or paste it above."
                })
                st.rerun()

            system_prompt = (
                "You are a helpful dissertation assistant embedded inside a Streamlit dashboard. "
                "Answer clearly and concretely. Use the provided dashboard context. "
                "If something cannot be inferred from the context, say so.\n\n"
                + build_dashboard_context_text()
            )
            messages = [{"role": "system", "content": system_prompt}] + st.session_state["chat_messages"][-10:]

            try:
                with st.status("Thinking…", expanded=False):
                    reply = openai_chat_completion(api_key=api_key, messages=messages, model="gpt-4o-mini")
                st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
            except Exception as e:
                st.session_state["chat_last_error"] = str(e)
                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": "I hit an API error. Check your API key and app logs. (Tip: on Streamlit Cloud, open Manage app → Logs.)"
                })

            st.rerun()

        if st.button("Clear chat history"):
            st.session_state["chat_messages"] = [{"role": "assistant", "content": "Chat cleared. Ask me anything about the dashboard."}]
            st.rerun()

        if st.session_state.get("chat_last_error"):
            st.caption(f"Last error: `{st.session_state['chat_last_error']}`")

# ======================================================================
# TABS
# ======================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Crisis Risk",
    "📊 Model Results",
    "🧠 SHAP (Detailed)",
    "📂 Data Explorer",
    "🧭 Behavioural Drivers (Explain-only)"
])

# -----------------------------------------------------------------------------
# TAB 1
with tab1:
    st.header("Crisis risk over time", divider="gray")

    if risk_df.empty:
        st.warning("No data for the selected filters.")
    else:
        if ALTAIR_OK:
            st.altair_chart(
                altair_risk_with_crisis_bands(
                    risk_df.sort_values(["country", "year"]),
                    crisis_years_df,
                    title="Predicted crisis probability (shaded = JST crisis years)",
                    threshold=threshold
                ),
                use_container_width=True
            )
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
                    st.metric(label=f"{c} crisis risk", value="(not selected)")
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
        st.markdown("### 🌍 GDP (from JST)")
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
        st.caption("SHAP splits the prediction (relative to a baseline) across features. Bigger |SHAP| = bigger influence.")

        try:
            global_shap = compute_shap_global(
                selected_model=selected_model,
                Xs_test=Xs_test,
                feature_names=all_features,
                sample_n=shap_sample_n
            )

            st.markdown("#### Pie: global impact share (mean |SHAP|)")
            st.caption("Share of total mean(|SHAP|) across sampled test rows (global importance).")

            if ALTAIR_OK:
                render_shap_pie_altair(global_shap, top_k=8)
            else:
                top = global_shap.head(8).copy()
                other = float(global_shap["share"].iloc[8:].sum()) if len(global_shap) > 8 else 0.0
                if other > 0:
                    top = pd.concat(
                        [top, pd.DataFrame([{"Feature": "Other", "mean_abs": np.nan, "mean_signed": 0.0, "share": other}])],
                        ignore_index=True
                    )
                plt.figure(figsize=(7, 6))
                plt.pie(top["share"], labels=top["Feature"], autopct=lambda p: f"{p:.1f}%" if p >= 4 else "", startangle=90)
                plt.title("SHAP Feature Impact Share (mean |SHAP|)")
                st.pyplot(plt.gcf(), clear_figure=True)
                st.info("Install `altair` for an interactive pie chart.")

            st.write("")
            st.markdown("#### SHAP explainability (right under the pie)")
            st.markdown(
                "- **Positive SHAP** → pushes predicted crisis risk **up**.\n"
                "- **Negative SHAP** → pushes predicted crisis risk **down**.\n"
                "- **mean(|SHAP|)** → global importance (how strongly a feature tends to matter overall).\n"
                "- The pie uses **share of mean(|SHAP|)**, so it’s a **global feature influence** summary."
            )

            st.write("")
            st.markdown("#### Top 6 SHAP drivers (graph + explanation)")
            render_top6_bar(global_shap, top_n=6, title="Top 6 drivers (mean |SHAP|)")

            top6 = global_shap.head(6).copy()
            st.markdown("**What the top 6 represent**")
            st.markdown(
                "- Ranked by **mean(|SHAP|)** over the sampled rows.\n"
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
# TAB 2
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
# TAB 3
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
# TAB 4
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

# -----------------------------------------------------------------------------
# TAB 5: Behavioural Drivers (Explain-only)
with tab5:
    st.header("Behavioural drivers (Explain-only)", divider="gray")
    st.caption("These variables are engineered for interpretation and visual insight. They are NOT used in the model pipeline.")

    beh = behavioural[
        (behavioural["country"].isin(risk_countries)) &
        (behavioural["year"] >= from_year) &
        (behavioural["year"] <= to_year)
    ].copy()

    if beh.empty:
        st.warning("No behavioural data for the selected filters.")
    else:
        proxy_options = {
            "Risk appetite (risky_tr − safe_tr)": "risk_appetite_z",
            "Market volatility (5y rolling std of Δ risky_tr)": "market_volatility_z",
            "Debt service risk (debtgdp × stir)": "debt_service_risk_z",
        }
        outcome_options = {
            "GDP growth (Δ gdp)": "gdp_growth",
            "Inflation (Δ cpi)": "inflation",
            "Unemployment change (Δ unemp)": "unemp_chg",
            "Model crisis probability (selected model)": "__MODEL_RISK__",
        }

        cL, cR = st.columns([0.45, 0.55], gap="large")
        with cL:
            proxy_label = st.selectbox("Select behavioural proxy", list(proxy_options.keys()), index=0)
            outcome_label = st.selectbox("Select outcome", list(outcome_options.keys()), index=0)
            lag = st.slider("Lag (years): compare proxy(t) with outcome(t+lag)", 0, 3, 1, 1)

        beh2 = beh.copy()
        if outcome_options[outcome_label] == "__MODEL_RISK__":
            tmp_risk = risk_df[["country", "year", "crisis_prob"]].copy()
            beh2 = beh2.merge(tmp_risk, on=["country", "year"], how="left")
            out_col = "crisis_prob"
        else:
            out_col = outcome_options[outcome_label]

        proxy_col = proxy_options[proxy_label]
        beh2["out_lagged"] = beh2.groupby("country")[out_col].shift(-lag)

        # One-line meaning under selectors
        with cL:
            st.caption(f"Proxy meaning: {explain_proxy_short(proxy_col)}")
            st.caption(f"Outcome meaning: {explain_outcome_short(out_col)}")

        # 1) Time series with crisis shading
        st.subheader("1) Proxy over time (shaded = crisis years)", divider="gray")
        if ALTAIR_OK:
            cdf = beh2[beh2["crisisJST"] == 1][["country", "year"]].drop_duplicates().copy()
            cdf["x_start"] = cdf["year"] - 0.5
            cdf["x_end"] = cdf["year"] + 0.5

            domain, range_colors = build_color_scale(beh2["country"].unique())

            rect = alt.Chart(cdf).mark_rect(opacity=0.15).encode(
                x="x_start:Q", x2="x_end:Q",
                y=alt.value(-10), y2=alt.value(10),
                color=alt.Color("country:N", scale=alt.Scale(domain=domain, range=range_colors), legend=None),
            )

            line = alt.Chart(beh2).mark_line().encode(
                x=alt.X("year:Q", title="year"),
                y=alt.Y(f"{proxy_col}:Q", title=proxy_label),
                color=alt.Color("country:N", scale=alt.Scale(domain=domain, range=range_colors),
                                legend=alt.Legend(title="country")),
                tooltip=["country:N", "year:Q", alt.Tooltip(f"{proxy_col}:Q", format=".3f")],
            )

            st.altair_chart((rect + line).properties(height=320).interactive(), use_container_width=True)
        else:
            st.line_chart(beh2, x="year", y=proxy_col, color="country")

        st.caption(
            f"🧠 What this means: {explain_proxy_short(proxy_col)} "
            f"Shaded bands mark crisis years, so you can see whether this behaviour tends to rise or fall around crises."
        )

        # 2) Lag correlation bars
        st.subheader("2) Lag correlation: proxy(t) vs outcome(t+lag)", divider="gray")
        corr_rows = []
        for c in sorted(beh2["country"].dropna().unique()):
            tmp = beh2[beh2["country"] == c][[proxy_col, "out_lagged"]].dropna()
            corr = tmp[proxy_col].corr(tmp["out_lagged"]) if len(tmp) >= 10 else np.nan
            corr_rows.append({"country": c, "corr": corr})

        pooled = beh2[[proxy_col, "out_lagged"]].dropna()
        pooled_corr = pooled[proxy_col].corr(pooled["out_lagged"]) if len(pooled) >= 20 else np.nan
        corr_rows.append({"country": "Pooled", "corr": pooled_corr})

        corr_df = pd.DataFrame(corr_rows)

        if ALTAIR_OK:
            st.altair_chart(
                alt.Chart(corr_df).mark_bar().encode(
                    x=alt.X("corr:Q", title="correlation", scale=alt.Scale(domain=[-1, 1])),
                    y=alt.Y("country:N", sort=None, title=None),
                    tooltip=["country:N", alt.Tooltip("corr:Q", format=".3f")]
                ).properties(height=180).interactive(),
                use_container_width=True
            )
        else:
            st.dataframe(corr_df, use_container_width=True)

        st.caption(
            f"🧭 What this means: This checks whether changes in **{proxy_label}** tend to come **before** changes in **{outcome_label}** "
            f"(using a {lag}-year lead). Values near +1 mean they usually move together; near −1 means they move opposite. "
            f"This is correlation (a pattern), not proof of cause."
        )

        # 3) Scatter with crisis highlighting
        st.subheader("3) Scatter: proxy vs outcome (crisis years highlighted)", divider="gray")
        scat = beh2[["country", "year", "crisisJST", proxy_col, "out_lagged"]].dropna().copy()
        if scat.empty:
            st.info("Not enough data for scatter (missing values after lagging). Try a wider window or smaller lag.")
        else:
            scat["crisis_flag"] = scat["crisisJST"].map({0: "Normal", 1: "Crisis year"})
            if ALTAIR_OK:
                domain, range_colors = build_color_scale(scat["country"].unique())
                chart = alt.Chart(scat).mark_circle(size=70).encode(
                    x=alt.X(f"{proxy_col}:Q", title=proxy_label),
                    y=alt.Y("out_lagged:Q", title=f"{outcome_label} (t+{lag})"),
                    color=alt.Color("country:N", scale=alt.Scale(domain=domain, range=range_colors)),
                    shape=alt.Shape("crisis_flag:N"),
                    tooltip=[
                        "country:N", "year:Q", "crisis_flag:N",
                        alt.Tooltip(f"{proxy_col}:Q", format=".3f"),
                        alt.Tooltip("out_lagged:Q", format=".3f")
                    ]
                ).properties(height=340).interactive()
                st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(scat.head(50), use_container_width=True)

            st.caption(
                "🎯 What this means: Each dot is a year. If crisis-year dots cluster at extreme proxy values, "
                "it suggests this behaviour is often present when crises happen (or just before, depending on the lag)."
            )

        # 4) Correlation heatmap
        st.subheader("4) Correlation heatmap (within selection)", divider="gray")
        heat_cols = [
            "risk_appetite_z", "market_volatility_z", "debt_service_risk_z",
            "gdp_growth", "inflation", "unemp_chg"
        ]
        if "crisis_prob" in beh2.columns:
            heat_cols = heat_cols + ["crisis_prob"]

        heat_data = beh2[heat_cols].dropna()
        if len(heat_data) < 25:
            st.info("Not enough complete rows for a stable heatmap. Try a wider year range.")
        else:
            corr = heat_data.corr(numeric_only=True)
            corr_long = corr.reset_index().melt(id_vars="index", var_name="var2", value_name="corr")
            corr_long = corr_long.rename(columns={"index": "var1"})

            if ALTAIR_OK:
                heat = alt.Chart(corr_long).mark_rect().encode(
                    x=alt.X("var2:N", title=None),
                    y=alt.Y("var1:N", title=None),
                    color=alt.Color("corr:Q", scale=alt.Scale(domain=[-1, 1])),
                    tooltip=["var1:N", "var2:N", alt.Tooltip("corr:Q", format=".3f")]
                ).properties(height=360).interactive()
                st.altair_chart(heat, use_container_width=True)
            else:
                st.dataframe(corr, use_container_width=True)

        st.caption(
            "🗺️ What this means: This heatmap summarises how strongly variables move together in your selected window. "
            "Values near +1 mean they move together, near −1 means they move opposite, near 0 means little relationship. "
            "It’s a quick overview, not a causal claim."
        )

        st.divider()
        st.markdown("**Note:** Correlation ≠ causation. These charts are included to explain relationships and timing, not to train the model.")
