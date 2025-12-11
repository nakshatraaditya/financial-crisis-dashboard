# ======================================================================
#  FINANCIAL CRISIS EARLY WARNING SYSTEM – STREAMLIT DASHBOARD
#  Multi-Model, Tabs, SHAP, Heatmaps, Regime Segmentation
# ======================================================================

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import shap

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

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Crisis Early Warning System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📉 Financial Crisis Early Warning Dashboard")
st.markdown(
    "Early-warning system for **USA, UK, and Canada**, using macro-financial indicators and multiple ML models."
)

# ----------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------
@st.cache_data
def load_data(file="JSTdatasetR6.xlsx"):
    df = pd.read_excel(file)
    df = df[df["country"].isin(["USA", "UK", "Canada"])].copy()
    df = df.sort_values(["country", "year"])
    return df

# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------
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
        "money_expansion", "ca_gdp",
    ]

    df = df[["country", "year", "crisisJST"] + features]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df, features

# ----------------------------------------------------------------------
# 3. CLEANING (CAUSAL IMPUTATION)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 4. TARGET VARIABLE – Crisis in next 2 years
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 5. BUILD MODEL SET
# ----------------------------------------------------------------------
def build_model_set():
    return {
        "Logistic Regression (Rare Event)": LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=5000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=600,
            max_depth=4,
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
        "SVM (RBF, calibrated)": SVC(
            kernel="rbf",
            probability=True,
            C=2.0,
            gamma="scale",
            class_weight="balanced",
            random_state=42
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=2000,
            random_state=42
        )
    }

# ----------------------------------------------------------------------
# 6. TRAIN ALL MODELS + METRICS
# ----------------------------------------------------------------------
@st.cache_resource
def train_all_models(df_target, features):
    # Time-based split
    train = df_target[df_target["year"] < 1970]
    val   = df_target[(df_target["year"] >= 1970) & (df_target["year"] < 1990)]
    test  = df_target[df_target["year"] >= 1990]

    X_train = train[features]
    y_train = train["target"]
    X_val   = val[features]
    y_val   = val["target"]
    X_test  = test[features]
    y_test  = test["target"]

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_val   = scaler.transform(X_val)
    Xs_test  = scaler.transform(X_test)

    models = build_model_set()
    results = {}
    comparison_rows = []

    for name, clf in models.items():
        st.write(f"🔧 Training: **{name}**")

        sample_weights = np.where(y_train == 1, 10, 1)

        if name == "Neural Network (MLP)":
            clf.fit(Xs_train, y_train)
        else:
            clf.fit(Xs_train, y_train, sample_weight=sample_weights)

        # Validation probabilities
        if hasattr(clf, "predict_proba"):
            val_probs = clf.predict_proba(Xs_val)[:, 1]
        else:
            raw = clf.decision_function(Xs_val)
            val_probs = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

        # Threshold tuning on validation (F1)
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.05, 0.90, 0.01):
            preds = (val_probs >= t).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        val_preds = (val_probs >= best_t).astype(int)
        val_roc = roc_auc_score(y_val, val_probs)
        val_pr  = average_precision_score(y_val, val_probs)
        val_prec = precision_score(y_val, val_preds, zero_division=0)
        val_rec  = recall_score(y_val, val_preds, zero_division=0)

        # Test probabilities
        if hasattr(clf, "predict_proba"):
            test_probs = clf.predict_proba(Xs_test)[:, 1]
        else:
            raw_t = clf.decision_function(Xs_test)
            test_probs = (raw_t - raw_t.min()) / (raw_t.max() - raw_t.min() + 1e-9)

        test_preds = (test_probs >= best_t).astype(int)
        test_roc = roc_auc_score(y_test, test_probs)
        test_pr  = average_precision_score(y_test, test_probs)
        test_prec = precision_score(y_test, test_preds, zero_division=0)
        test_rec  = recall_score(y_test, test_preds, zero_division=0)
        test_f1   = f1_score(y_test, test_preds, zero_division=0)

        results[name] = {
            "clf": clf,
            "threshold": best_t,
            "val_probs": val_probs,
            "test_probs": test_probs,
            "val_metrics": {
                "ROC-AUC": val_roc,
                "PR-AUC": val_pr,
                "Precision": val_prec,
                "Recall": val_rec,
                "F1": best_f1,
            },
            "test_metrics": {
                "ROC-AUC": test_roc,
                "PR-AUC": test_pr,
                "Precision": test_prec,
                "Recall": test_rec,
                "F1": test_f1,
            },
        }

        comparison_rows.append({
            "model": name,
            "VAL_ROC-AUC": val_roc,
            "VAL_PR-AUC": val_pr,
            "VAL_F1": best_f1,
            "TEST_ROC-AUC": test_roc,
            "TEST_PR-AUC": test_pr,
            "TEST_F1": test_f1,
            "threshold": best_t
        })

    comparison_df = pd.DataFrame(comparison_rows)
    return results, comparison_df, scaler, train, val, test, features

# ----------------------------------------------------------------------
# RUN PIPELINE
# ----------------------------------------------------------------------
df_raw = load_data()
df_feat, feature_list = engineer_features(df_raw)
df_clean = clean_data(df_feat)
df_target = create_target(df_clean)

results, comparison_df, scaler, train_df, val_df, test_df, features = train_all_models(
    df_target, feature_list
)

# ----------------------------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

# Theme (simple CSS tweak)
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
if theme == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #111111; color: #EEEEEE; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Model selection
model_names = list(results.keys())
selected_model_name = st.sidebar.selectbox("Select model", model_names, index=model_names.index("Gradient Boosting") if "Gradient Boosting" in model_names else 0)
selected_model_info = results[selected_model_name]
selected_clf = selected_model_info["clf"]
selected_threshold = selected_model_info["threshold"]

st.sidebar.markdown(f"**Selected Model:** {selected_model_name}")
st.sidebar.markdown(f"**Optimal Threshold:** `{selected_threshold:.2f}`")

# ----------------------------------------------------------------------
# TABS LAYOUT
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Model Results", "🔍 SHAP Explainability", "📈 Risk Timeline", "📂 Data Explorer"]
)

# ----------------------------------------------------------------------
# TAB 1 – MODEL RESULTS
# ----------------------------------------------------------------------
with tab1:
    st.header("📊 Model Comparison & Performance")

    st.subheader("Validation vs Test Performance (All Models)")
    st.dataframe(comparison_df.style.format({
        "VAL_ROC-AUC": "{:.3f}",
        "VAL_PR-AUC": "{:.3f}",
        "VAL_F1": "{:.3f}",
        "TEST_ROC-AUC": "{:.3f}",
        "TEST_PR-AUC": "{:.3f}",
        "TEST_F1": "{:.3f}",
        "threshold": "{:.2f}"
    }))

    # Show detailed metrics for selected model
    st.subheader(f"Detailed Test Metrics – {selected_model_name}")
    test_metrics = selected_model_info["test_metrics"]
    st.write(pd.DataFrame(test_metrics, index=["Test"]).T)

    # Confusion matrix for selected model
    st.subheader("Confusion Matrix (Test Set)")
    X_test = scaler.transform(test_df[features])
    if hasattr(selected_clf, "predict_proba"):
        test_probs_sel = selected_clf.predict_proba(X_test)[:, 1]
    else:
        raw_t = selected_clf.decision_function(X_test)
        test_probs_sel = (raw_t - raw_t.min()) / (raw_t.max() - raw_t.min() + 1e-9)

    test_preds_sel = (test_probs_sel >= selected_threshold).astype(int)
    cm = confusion_matrix(test_df["target"], test_preds_sel)

    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                xticklabels=["No Crisis", "Crisis"],
                yticklabels=["No Crisis", "Crisis"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # Precision–Recall curve for selected model
    st.subheader("Precision–Recall Curve (Test Set)")
    prec_curve, rec_curve, _ = precision_recall_curve(test_df["target"], test_probs_sel)
    fig_pr, ax_pr = plt.subplots(figsize=(5, 3))
    ax_pr.plot(rec_curve, prec_curve, label="Model")
    baseline = test_df["target"].mean()
    ax_pr.axhline(baseline, linestyle="--", color="red", label=f"Baseline ({baseline:.2%})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(alpha=0.3)
    ax_pr.legend()
    st.pyplot(fig_pr)

    # Crisis heatmap (by country & decade)
    st.subheader("Crisis & Risk Heatmap (by Decade)")
    test_risk_df = test_df.copy()
    test_risk_df["predicted_prob"] = test_probs_sel
    test_risk_df["decade"] = (test_risk_df["year"] // 10) * 10

    heat = test_risk_df.groupby(["country", "decade"]).agg(
        crisis_rate=("crisisJST", "mean"),
        avg_risk=("predicted_prob", "mean")
    ).reset_index()

    heat_pivot = heat.pivot(index="country", columns="decade", values="avg_risk")

    fig_hm, ax_hm = plt.subplots(figsize=(6, 3))
    sns.heatmap(heat_pivot, annot=True, fmt=".2f", cmap="Reds", ax=ax_hm)
    ax_hm.set_title("Average Predicted Crisis Risk by Country & Decade")
    st.pyplot(fig_hm)

# ----------------------------------------------------------------------
# TAB 2 – SHAP EXPLAINABILITY
# ----------------------------------------------------------------------
with tab2:
    st.header("🔍 SHAP Global Explainability")

    if selected_model_name != "Logistic Regression (Rare Event)":
        st.info("SHAP is enabled only for **Logistic Regression** in this app for efficiency and stability. Select that model in the sidebar to view SHAP explanations.")
    else:
        st.markdown("Global feature importance for **Logistic Regression (Rare Event)**.")
        X_train = scaler.transform(train_df[features])
        X_test = scaler.transform(test_df[features])

        try:
            shap.initjs()
            explainer = shap.LinearExplainer(results["Logistic Regression (Rare Event)"]["clf"], X_train)
            shap_vals = explainer.shap_values(X_test)

            st.subheader("SHAP Summary Plot (Test Set)")
            shap.summary_plot(shap_vals, test_df[features], show=False)
            fig = plt.gcf()
            st.pyplot(fig, clear_figure=True)

            st.subheader("SHAP Global Feature Importance (Bar)")
            shap.summary_plot(shap_vals, test_df[features], plot_type="bar", show=False)
            fig2 = plt.gcf()
            st.pyplot(fig2, clear_figure=True)
        except Exception as e:
            st.error(f"SHAP failed to compute: {e}")

# ----------------------------------------------------------------------
# TAB 3 – RISK TIMELINE
# ----------------------------------------------------------------------
with tab3:
    st.header("📈 Crisis Risk Timeline")

    # Risk for selected model
    test_probs_all = selected_model_info["test_probs"]
    test_df_model = test_df.copy()
    test_df_model["predicted_prob"] = test_probs_all
    test_df_model["predicted_class"] = (test_probs_all >= selected_threshold).astype(int)

    # Country selector
    country = st.selectbox("Select country", ["USA", "UK", "Canada"])
    df_c = test_df_model[test_df_model["country"] == country].copy()

    # Single country timeline + regimes
    st.subheader(f"{country} — Crisis Risk Timeline (Test Period)")
    fig_rt, ax_rt = plt.subplots(figsize=(10, 4))
    ax_rt.plot(df_c["year"], df_c["predicted_prob"], color="purple", label="Risk Score")
    ax_rt.axhline(selected_threshold, color="orange", linestyle="--", label=f"Threshold ({selected_threshold:.2f})")

    # Crisis shading
    for y in df_c[df_c["crisisJST"] == 1]["year"]:
        ax_rt.axvspan(y - 0.5, y + 0.5, color="red", alpha=0.3)

    # Simple regime segmentation (vertical lines)
    for year, label in [(1990, "Post-Cold War"), (2000, "Dot-com"), (2007, "Pre-GFC"), (2008, "GFC"), (2010, "Post-GFC")]:
        ax_rt.axvline(year, color="grey", alpha=0.3, linestyle=":")
        ax_rt.text(year + 0.1, 0.95, label, rotation=90, fontsize=7, alpha=0.6)

    ax_rt.set_ylim(0, 1)
    ax_rt.set_xlabel("Year")
    ax_rt.set_ylabel("Crisis Probability")
    ax_rt.legend(loc="upper left")
    ax_rt.grid(alpha=0.3)
    st.pyplot(fig_rt)

    # Country comparison
    st.subheader("🇺🇸🇬🇧🇨🇦 Risk Comparison Across Countries")
    fig_cmp, ax_cmp = plt.subplots(figsize=(10, 4))
    for c in ["USA", "UK", "Canada"]:
        tmp = test_df_model[test_df_model["country"] == c]
        ax_cmp.plot(tmp["year"], tmp["predicted_prob"], label=c)

    ax_cmp.axhline(selected_threshold, color="orange", linestyle="--", label="Threshold")
    ax_cmp.set_xlabel("Year")
    ax_cmp.set_ylabel("Crisis Probability")
    ax_cmp.set_ylim(0, 1)
    ax_cmp.legend()
    ax_cmp.grid(alpha=0.3)
    st.pyplot(fig_cmp)

# ----------------------------------------------------------------------
# TAB 4 – DATA EXPLORER
# ----------------------------------------------------------------------
with tab4:
    st.header("📂 Data Explorer")

    test_export = test_df.copy()
    test_export["predicted_prob"] = results[selected_model_name]["test_probs"]
    test_export["predicted_class"] = (test_export["predicted_prob"] >= selected_threshold).astype(int)

    country_filter = st.selectbox("Filter by Country", ["All", "USA", "UK", "Canada"])
    if country_filter != "All":
        test_view = test_export[test_export["country"] == country_filter]
    else:
        test_view = test_export

    st.dataframe(test_view)

    csv_data = test_view.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_data,
        file_name=f"crisis_predictions_{selected_model_name.replace(' ', '_')}.csv",
        mime="text/csv"
    )

    st.markdown("Tip: Use this export in Power BI for additional custom visuals.")

