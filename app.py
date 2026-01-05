# ======================================================================
#  FINANCIAL CRISIS EARLY WARNING SYSTEM – STREAMLIT DASHBOARD
#  Uses YOUR multi-model pipeline:
#   - Missing-value flags
#   - Proper scaling (continuous only)
#   - Multi-model validation + best model selection
#   - SHAP explainability
#
#  Current updates:
#   ✅ GDP chart uses JSTdatasetR6.xlsx (NO external gdp_data.csv needed)
#   ✅ SHAP pie chart shown on FIRST page
#   ✅ Pie chart visibility fixed (no legend squeeze, bigger chart, outside labels, dark theme)
#   ✅ SHAP explainability text placed RIGHT UNDER the pie chart
#   ✅ results_df Arrow-safe (no sklearn objects inside dataframe)
#
#  Note:
#   - Interactive pie requires plotly. If not installed, matplotlib fallback is used.
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
# HEADER
"""
# 📉 Financial Crisis Early Warning System (EWS)

This dashboard implements your **multi-model dissertation pipeline**:
- Missing-value flags + causal imputation  
- Proper scaling (continuous only)  
- Validation-based model selection + threshold optimization  
- SHAP explainability  
- Crisis risk timeline + JST macro context (GDP + Real House Prices)  
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
# 1) JST MACRO DATA (GDP + House prices) FOR CHARTS
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

    # Real house prices (if possible)
    if "hpnom" in out.columns and "cpi" in out.columns:
        out["house_price_real"] = out["hpnom"] / (out["cpi"] + 1e-9)

    return out


# ======================================================================
# 2) YOUR PIPELINE FUNCTIONS
# ======================================================================

def load_data(file="JSTdatasetR6.xlsx"):
    df = pd.read_excel(file)
    df = df[df["country"].isin(["USA", "UK", "Canada"])].copy()
    df = df.sort_values(["country", "year"])
    return df


def engineer_features(df):
    df = df.copy()

    # allow uploads without crisisJST
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

    df["hp_real"]  = df["hpnom"] / (df["cpi"] + 1e-9)
    df["hp_trend"] = df.groupby("country")["hp_real"].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )
    df["housing_bubble"] = (df["hp_real"] - df["hp_trend"]) / (df["hp_trend"] + 1e-9)

    df["real_credit"]   = df["tloans"] / (df["cpi"] + 1e-9)
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

    # Missing flags
    for col in base_features:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # Causal fill
    df[base_features] = df.groupby("country")[base_features].transform(
        lambda x: x.ffill(limit=3).bfill(limit=3)
    )

    # Median fallback
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
    df_feat, base_features = engineer_features(df_raw)
    df_clean = clean_data(df_feat, base_features)

    # reset index so alignment by row-position is safe
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
        # Arrow-safe (exclude model object)
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
    }


# ======================================================================
# 4) SIDEBAR UPLOAD & PREDICT
# ======================================================================

def run_upload_predict_sidebar(selected_model, scaler, threshold, base_features, missing_features, all_features):
    st.sidebar.header("📥 Upload & Predict")

    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="upload_csv")

    if not uploaded:
        st.sidebar.info("Upload a CSV to generate predictions.")
        return None

    user_df = pd.read_csv(uploaded)

    required_raw = {
        "country","year","lev","noncore","ltd","hpnom","cpi",
        "tloans","ltrate","stir","money","gdp","ca"
    }
    missing = sorted(list(required_raw - set(user_df.columns)))
    if missing:
        st.sidebar.error(
            "Missing columns (showing up to 12): " + ", ".join(missing[:12]) +
            ("..." if len(missing) > 12 else "")
        )
        return None

    df_feat, base_feats = engineer_features(user_df)
    df_clean = clean_data(df_feat, base_feats)

    X_u = df_clean[all_features].copy()

    X_u_cont = scaler.transform(X_u[base_features])
    X_u_scaled = np.hstack([X_u_cont, X_u[missing_features].values])

    probs = selected_model.predict_proba(X_u_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)

    out = df_clean[["country", "year"]].copy()
    out["predicted_prob"] = probs
    out["predicted_class"] = preds

    st.sidebar.success(f"{int(preds.sum())} high-risk rows ({100 * preds.mean():.1f}%)")
    return out


# ======================================================================
# 5) SHAP PIE (cached per model + sample size)
# ======================================================================

def get_shap_pie_data(model_key: str, selected_model, Xs_test: np.ndarray, feature_names: list, sample_n: int):
    """
    Returns df: Feature, Share using mean(|SHAP|) across sampled test rows.
    Cached in st.session_state to avoid recomputation on reruns.
    """
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
        values = values[:, :, 1]  # positive class

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

macro_df = get_jst_macro_series(str(JST_XLSX))

# ======================================================================
# 7) SIDEBAR CONTROLS
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

risk_countries = st.sidebar.multiselect(
    "Countries (risk timeline)",
    ["USA", "UK", "Canada"],
    default=["USA", "UK", "Canada"],
    key="risk_countries"
)

min_year = int(df_target["year"].min())
max_year = int(df_target["year"].max())

from_year, to_year = st.sidebar.slider(
    "Year range (risk timeline)",
    min_value=min_year,
    max_value=max_year,
    value=(1900, max_year),
    key="risk_year_slider"
)

show_crisis_years = st.sidebar.checkbox("Show crisis years list", value=True)

st.sidebar.subheader("🧠 SHAP Pie (first page)")
shap_pie_sample_n = st.sidebar.slider("SHAP sample size", 50, 400, 150, 25)

# Upload & predict in sidebar
uploaded_preds = run_upload_predict_sidebar(
    selected_model=selected_model,
    scaler=scaler,
    threshold=threshold,
    base_features=base_features,
    missing_features=missing_features,
    all_features=all_features
)

# ======================================================================
# 8) PREPARE RISK DF (timeline probs)
# ======================================================================

risk_df = df_target[
    (df_target["country"].isin(risk_countries)) &
    (df_target["year"] >= from_year) &
    (df_target["year"] <= to_year)
].copy()

risk_scaled = X_full_scaled[risk_df.index.to_numpy()]
risk_df["crisis_prob"] = selected_model.predict_proba(risk_scaled)[:, 1]

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
# TAB 1: Crisis Risk + JST GDP + Real House Prices + SHAP pie + explanation under pie
with tab1:
    st.header("Crisis risk over time", divider="gray")

    if risk_df.empty:
        st.warning("No data for the selected filters.")
    else:
        st.line_chart(risk_df, x="year", y="crisis_prob", color="country")

        st.write("")
        st.subheader(f"Risk summary ({to_year})", divider="gray")

        latest = risk_df[risk_df["year"] == to_year]
        cols = st.columns(3)

        for i, c in enumerate(risk_countries):
            with cols[i % 3]:
                val = latest[latest["country"] == c]["crisis_prob"].mean()
                if pd.isna(val):
                    st.metric(label=f"{c} risk", value="n/a")
                else:
                    st.metric(
                        label=f"{c} risk",
                        value=f"{val:.2f}",
                        delta="HIGH" if val >= threshold else "LOW",
                        delta_color="inverse",
                    )

        if show_crisis_years:
            with st.expander("Crisis years in selected window"):
                for c in risk_countries:
                    years = risk_df[(risk_df["country"] == c) & (risk_df["crisisJST"] == 1)]["year"].tolist()
                    st.write(f"**{c}:** " + (", ".join(map(str, years)) if years else "none"))

        st.write("")
        st.subheader("JST macro context + explainability", divider="gray")

        left, right = st.columns(2)

        # --- LEFT: GDP & House prices from JSTdatasetR6.xlsx ---
        with left:
            st.markdown("### 🌍 GDP (JST variable)")
            macro_filtered = macro_df[
                (macro_df["country"].isin(risk_countries)) &
                (macro_df["year"] >= from_year) &
                (macro_df["year"] <= to_year)
            ].copy()

            if "gdp" in macro_filtered.columns:
                st.line_chart(macro_filtered, x="year", y="gdp", color="country")
                st.caption("This chart uses the **JST `gdp` variable** (check JST codebook for exact definition/units).")
            else:
                st.warning("Column `gdp` not found in JSTdatasetR6.xlsx")

            st.write("")
            st.markdown("### 🏠 Real house prices (hpnom / cpi)")
            if "house_price_real" in macro_filtered.columns:
                st.line_chart(macro_filtered, x="year", y="house_price_real", color="country")
            else:
                st.info("House price columns not available to compute real house prices.")

        # --- RIGHT: SHAP pie (visibility fixed) + SHAP explanation under the pie ---
        with right:
            st.markdown("### 🧠 SHAP Explainability")
            st.caption("Pie uses **mean(|SHAP|)** across sampled test rows (share of total influence).")

            top_k = 8  # fewer slices => clearer pie
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

                    # Fix visibility: no legend squeeze, bigger plot, dark theme, outside labels
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

                # SHAP explainability UNDER the pie chart
                st.markdown(
                    """
**What SHAP values represent**

- SHAP (SHapley Additive exPlanations) decomposes a model prediction into **feature contributions** around a baseline.  
- A **positive SHAP value** means that feature pushes the prediction **towards crisis (higher risk)**.  
- A **negative SHAP value** means it pushes the prediction **away from crisis (lower risk)**.  
- Larger **|SHAP|** means the feature has **stronger influence** on the model output.

**What this pie chart shows**

This pie chart uses **mean absolute SHAP** (mean(|SHAP|)) over many test observations.  
So, it reflects **global feature importance (share of total influence)**, not the direction of effect.  
For direction and dispersion, use the **SHAP dot/bar plots** in the SHAP tab.
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
# TAB 3: Uploaded Predictions
with tab3:
    st.header("Predictions from uploaded CSV", divider="gray")

    if uploaded_preds is None:
        st.info("Upload a CSV from the sidebar to view predictions here.")
    else:
        st.dataframe(uploaded_preds, use_container_width=True)

        st.write("")
        st.subheader("Uploaded risk over time", divider="gray")
        st.line_chart(
            uploaded_preds.sort_values(["country", "year"]),
            x="year",
            y="predicted_prob",
            color="country",
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
