# app.py
# ======================================================================
#  FINANCIAL CRISIS EARLY WARNING SYSTEM – STREAMLIT DASHBOARD
#  Multi-Model Pipeline (USA, UK, Canada)
#  Based on your final dissertation-ready modelling code
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

import shap
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 1. CORE PIPELINE FUNCTIONS (your logic, wrapped for Streamlit)
# ----------------------------------------------------------------------

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
            mu = s.expanding().mean()
            sd = s.expanding().std().replace(0, np.nan)
            return (s - mu) / (sd + 1e-9)
        return df_inner.groupby("country")[col].transform(z)

    df["noncore_z"] = expanding_z(df, "noncore")
    df["ltd_z"] = expanding_z(df, "ltd")
    df["leverage_z"] = expanding_z(df, "leverage_risk")

    df["banking_fragility"] = (
        0.4 * df["noncore_z"]
        + 0.3 * df["ltd_z"]
        + 0.3 * df["leverage_z"]
    )

    df["hp_real"] = df["hpnom"] / df["cpi"]
    df["hp_trend"] = df.groupby("country")["hp_real"].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )
    df["housing_bubble"] = (df["hp_real"] - df["hp_trend"]) / (df["hp_trend"] + 1e-9)

    df["real_credit"] = df["tloans"] / df["cpi"]
    df["credit_growth"] = df.groupby("country")["real_credit"].pct_change()

    df["yield_curve"] = df["ltrate"] - df["stir"]

    us_ltrate = df[df["country"] == "USA"].drop_duplicates("year").set_index("year")["ltrate"]
    df["us_ltrate"] = df["year"].map(us_ltrate.to_dict())
    df["sovereign_spread"] = df["ltrate"] - df["us_ltrate"]

    df["money_gdp"] = df["money"] / df["gdp"]
    df["money_expansion"] = df.groupby("country")["money_gdp"].pct_change()

    df["ca_gdp"] = df["ca"] / df["gdp"]

    feature_list = [
        "housing_bubble", "credit_growth", "banking_fragility",
        "sovereign_spread", "yield_curve", "money_expansion", "ca_gdp"
    ]

    df = df[["country", "year", "crisisJST"] + feature_list].copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    return df, feature_list


def clean_data(df):
    df = df.copy()

    df = df[~df["year"].between(1914, 1918)]
    df = df[~df["year"].between(1939, 1945)]

    features = [c for c in df.columns if c not in ["country", "year", "crisisJST"]]

    for col in features:
        df[f"{col}_missing"] = df[col].isna().astype(int)

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


def build_model_set():
    return {
        "Logistic Regression (Rare Event)": LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=5000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=600, max_depth=4, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.01, max_depth=3, random_state=42
        ),
        "SVM (RBF, calibrated)": SVC(
            kernel="rbf", probability=True, C=2.0,
            gamma="scale", class_weight="balanced"
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            learning_rate_init=0.001,
            max_iter=2000
        )
    }


def evaluate_model(model_name, model, X_train, y_train, X_val, y_val):
    weights = np.where(y_train == 1, 10, 1)

    if model_name == "Neural Network (MLP)":
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=weights)

    if hasattr(model, "predict_proba"):
        val_probs = model.predict_proba(X_val)[:, 1]
    else:
        raw = model.decision_function(X_val)
        val_probs = (raw - raw.min()) / (raw.max() - raw.min())

    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.05, 0.90, 0.01):
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    preds = (val_probs >= best_thresh).astype(int)

    return {
        "model": model_name,
        "ROC-AUC": roc_auc_score(y_val, val_probs),
        "PR-AUC": average_precision_score(y_val, val_probs),
        "Precision": precision_score(y_val, preds, zero_division=0),
        "Recall": recall_score(y_val, preds, zero_division=0),
        "F1": best_f1,
        "threshold": best_thresh,
        "clf": model
    }


def get_probs(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    return (raw - raw.min()) / (raw.max() - raw.min())


# cache the full training so Streamlit doesn't retrain on every interaction
@st.cache_data
def run_full_pipeline(file="JSTdatasetR6.xlsx"):
    df_raw = load_data(file)
    df_feat, features = engineer_features(df_raw)
    df_clean = clean_data(df_feat)
    df_target = create_target(df_clean)

    train = df_target[df_target["year"] < 1970]
    val   = df_target[(df_target["year"] >= 1970) & (df_target["year"] < 1990)]
    test  = df_target[df_target["year"] >= 1990]

    X_train = train.drop(columns=["target", "crisisJST", "country", "year"])
    X_val   = val.drop(columns=["target", "crisisJST", "country", "year"])
    X_test  = test.drop(columns=["target", "crisisJST", "country", "year"])

    y_train = train["target"]
    y_val   = val["target"]
    y_test  = test["target"]

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_val   = scaler.transform(X_val)
    Xs_test  = scaler.transform(X_test)

    model_set = build_model_set()
    results = []
    models = {}

    for name, clf in model_set.items():
        res = evaluate_model(name, clf, Xs_train, y_train, Xs_val, y_val)
        results.append(res)
        models[name] = res  # store full info

    results_df = pd.DataFrame(results)
    best_row = results_df.sort_values("F1", ascending=False).iloc[0]
    best_name = best_row["model"]
    best_thresh = best_row["threshold"]

    # compute test metrics for all models
    test_rows = []
    for name, info in models.items():
        clf = info["clf"]
        thresh = info["threshold"]
        probs = get_probs(clf, Xs_test)
        preds = (probs >= thresh).astype(int)

        test_rows.append({
            "model": name,
            "ROC-AUC": roc_auc_score(y_test, probs),
            "PR-AUC": average_precision_score(y_test, probs),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
            "threshold": thresh
        })

    test_df_metrics = pd.DataFrame(test_rows)

    # add risk scores for best model
    df_full = df_target.copy()
    X_full = df_full.drop(columns=["target", "crisisJST", "country", "year"])
    Xs_full = scaler.transform(X_full)
    df_full["risk_score"] = get_probs(models[best_name]["clf"], Xs_full)

    return {
        "df_raw": df_raw,
        "df_target": df_target,
        "train_df": train,
        "val_df": val,
        "test_df": test,
        "features": features,
        "scaler": scaler,
        "Xs_train": Xs_train,
        "Xs_val": Xs_val,
        "Xs_test": Xs_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "model_results_val": results_df,
        "model_results_test": test_df_metrics,
        "models": models,
        "best_model_name": best_name,
        "best_threshold": best_thresh,
        "df_full": df_full,
    }


# ----------------------------------------------------------------------
# 2. STREAMLIT APP LAYOUT
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Crisis Early Warning Dashboard",
    layout="wide"
)

st.title("📉 Financial Crisis Early Warning System")
st.caption("Multi-model early warning system for USA, UK and Canada (1870–2020)")

# Run pipeline
pipeline = run_full_pipeline()
df_raw      = pipeline["df_raw"]
df_target   = pipeline["df_target"]
train_df    = pipeline["train_df"]
val_df      = pipeline["val_df"]
test_df     = pipeline["test_df"]
features    = pipeline["features"]
scaler      = pipeline["scaler"]
Xs_train    = pipeline["Xs_train"]
Xs_val      = pipeline["Xs_val"]
Xs_test     = pipeline["Xs_test"]
y_train     = pipeline["y_train"]
y_val       = pipeline["y_val"]
y_test      = pipeline["y_test"]
results_val = pipeline["model_results_val"]
results_test= pipeline["model_results_test"]
models      = pipeline["models"]
best_name   = pipeline["best_model_name"]
best_thresh = pipeline["best_threshold"]
df_full     = pipeline["df_full"]

# Sidebar controls
st.sidebar.header("⚙️ Controls")

model_options = list(models.keys())
default_idx = model_options.index(best_name)
selected_model_name = st.sidebar.selectbox(
    "Select model to inspect", model_options, index=default_idx
)
selected_model_info = models[selected_model_name]
selected_model = selected_model_info["clf"]
selected_threshold = selected_model_info["threshold"]

selected_countries = st.sidebar.multiselect(
    "Select countries for risk timeline",
    options=["USA", "UK", "Canada"],
    default=["USA", "UK", "Canada"]
)

st.sidebar.markdown(f"**Best model (by F1 on validation):** `{best_name}`")
st.sidebar.markdown(f"**Best threshold (val):** `{best_thresh:.2f}`")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Results",
    "🔍 SHAP Explainability",
    "📈 Risk Timeline",
    "📂 Data Explorer"
])


# ----------------------------------------------------------------------
# TAB 1 – MODEL RESULTS
# ----------------------------------------------------------------------
with tab1:
    st.header("📊 Model Performance")

    st.subheader("Validation Set (1970–1990)")
    st.dataframe(results_val.style.format({
        "ROC-AUC": "{:.3f}",
        "PR-AUC": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1": "{:.3f}",
        "threshold": "{:.2f}"
    }))

    st.subheader("Test Set (1990–2020)")
    st.dataframe(results_test.style.format({
        "ROC-AUC": "{:.3f}",
        "PR-AUC": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1": "{:.3f}",
        "threshold": "{:.2f}"
    }))

    st.markdown(f"### Selected Model: `{selected_model_name}` – Test Performance")

    # compute test metrics for selected model
    probs_sel = get_probs(selected_model, Xs_test)
    preds_sel = (probs_sel >= selected_threshold).astype(int)

    roc_sel = roc_auc_score(y_test, probs_sel)
    pr_sel  = average_precision_score(y_test, probs_sel)
    prec_sel= precision_score(y_test, preds_sel, zero_division=0)
    rec_sel = recall_score(y_test, preds_sel, zero_division=0)
    f1_sel  = f1_score(y_test, preds_sel, zero_division=0)

    colA, colB, colC, colD, colE = st.columns(5)
    colA.metric("ROC-AUC", f"{roc_sel:.3f}")
    colB.metric("PR-AUC", f"{pr_sel:.3f}")
    colC.metric("Precision", f"{prec_sel:.3f}")
    colD.metric("Recall", f"{rec_sel:.3f}")
    colE.metric("F1", f"{f1_sel:.3f}")

    # Confusion matrix
    st.subheader("Confusion Matrix (Test Set)")
    cm = confusion_matrix(y_test, preds_sel)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                xticklabels=["No Crisis", "Crisis"],
                yticklabels=["No Crisis", "Crisis"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # Precision–Recall curve
    st.subheader("Precision–Recall Curve (Test Set)")
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs_sel)
    fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
    ax_pr.plot(rec_curve, prec_curve, lw=2)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(alpha=0.3)
    st.pyplot(fig_pr)


# ----------------------------------------------------------------------
# TAB 2 – SHAP EXPLAINABILITY
# ----------------------------------------------------------------------
with tab2:
    st.header("🔍 SHAP Global Explainability")

    if selected_model_name not in ["Logistic Regression (Rare Event)", "Gradient Boosting", "Random Forest"]:
        st.info("SHAP is only enabled for Logistic Regression and tree-based models (Gradient Boosting, Random Forest). Please select one of those in the sidebar.")
    else:
        st.write(f"Global feature contribution for **{selected_model_name}**.")

        X_train_feats = train_df[features]
        X_test_feats  = test_df[features]

        # use scaled versions if needed
        Xs_train_feats = Xs_train
        Xs_test_feats  = Xs_test

        try:
            if selected_model_name in ["Gradient Boosting", "Random Forest"]:
                explainer = shap.TreeExplainer(selected_model)
                shap_vals = explainer.shap_values(Xs_test_feats)
            else:
                explainer = shap.LinearExplainer(selected_model, Xs_train_feats)
                shap_vals = explainer.shap_values(Xs_test_feats)

            st.subheader("SHAP Summary Plot")
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            shap.summary_plot(shap_vals, X_test_feats, show=False)
            st.pyplot(fig1)

            st.subheader("SHAP Global Feature Importance (Bar)")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            shap.summary_plot(shap_vals, X_test_feats, plot_type="bar", show=False)
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"SHAP computation failed: {e}")


# ----------------------------------------------------------------------
# TAB 3 – RISK TIMELINE
# ----------------------------------------------------------------------
with tab3:
    st.header("📈 Crisis Risk Timeline")

    st.markdown(f"Using **best model** (by validation F1): `{best_name}` with threshold `{best_thresh:.2f}`")

    for c in selected_countries:
        df_c = df_full[df_full["country"] == c].copy()
        if df_c.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df_c["year"], df_c["risk_score"], color="purple", lw=2, label="Risk Score")
        ax.axhline(best_thresh, color="orange", linestyle="--", label="Threshold")

        crisis_years = df_c[df_c["crisisJST"] == 1]["year"]
        for y in crisis_years:
            ax.axvspan(y - 0.5, y + 0.5, color="red", alpha=0.3)

        ax.axvline(1990, lw=2, color="black")
        ax.text(df_c["year"].min(), 0.95, "TRAINING", color="gray", fontsize=8)
        ax.text(1992, 0.95, "TEST", color="black", fontsize=8)

        ax.set_title(f"{c} – Crisis Risk Timeline")
        ax.set_ylabel("Predicted Probability")
        ax.set_xlabel("Year")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        st.pyplot(fig)


# ----------------------------------------------------------------------
# TAB 4 – DATA EXPLORER
# ----------------------------------------------------------------------
with tab4:
    st.header("📂 Test Data Explorer (1990–2020)")
    st.markdown("Explore test-period predictions, actual crises, and features.")

    probs_best = get_probs(models[best_name]["clf"], Xs_test)
    preds_best = (probs_best >= best_thresh).astype(int)

    explorer_df = test_df.copy()
    explorer_df["predicted_prob"] = probs_best
    explorer_df["predicted_class"] = preds_best
    explorer_df["prediction_type"] = "TN"
    explorer_df.loc[(explorer_df["predicted_class"] == 1) & (explorer_df["target"] == 1), "prediction_type"] = "TP"
    explorer_df.loc[(explorer_df["predicted_class"] == 1) & (explorer_df["target"] == 0), "prediction_type"] = "FP"
    explorer_df.loc[(explorer_df["predicted_class"] == 0) & (explorer_df["target"] == 1), "prediction_type"] = "FN"

    country_filter = st.multiselect(
        "Filter by country",
        options=["USA", "UK", "Canada"],
        default=["USA", "UK", "Canada"]
    )

    type_filter = st.multiselect(
        "Filter by prediction type",
        options=["TP", "FP", "TN", "FN"],
        default=["TP", "FP", "TN", "FN"]
    )

    df_filtered = explorer_df[
        explorer_df["country"].isin(country_filter)
        & explorer_df["prediction_type"].isin(type_filter)
    ]

    st.dataframe(df_filtered.sort_values(["country", "year"]).reset_index(drop=True))

    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv,
        file_name="test_predictions_filtered.csv",
        mime="text/csv"
    )


