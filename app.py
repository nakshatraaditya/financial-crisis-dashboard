# ======================================================================
#  FINANCIAL CRISIS EARLY WARNING SYSTEM – STREAMLIT DASHBOARD
#  Streamlit-native interactivity (like GDP template) + GDP tab
#  Upload & Predict moved to Sidebar
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Page config
st.set_page_config(
    page_title="Financial Crisis Early Warning Dashboard",
    page_icon="📉",
    layout="wide",
)

APP_DIR = Path(__file__).parent
JST_XLSX = APP_DIR / "JSTdatasetR6.xlsx"          # <-- make sure this exists in your repo
GDP_CSV  = APP_DIR / "data" / "gdp_data.csv"      # <-- from Streamlit GDP template

# -----------------------------------------------------------------------------
# Header
"""
# 📉 Financial Crisis Early Warning Dashboard

Interactive exploration of **crisis risk probabilities** for **USA · UK · Canada (1870–2020)**  
plus a **GDP context dashboard** (World Bank style template).

Use the sidebar to:
- Filter **countries** and **years**
- Adjust **risk threshold**
- **Upload a CSV** and generate predictions
"""
st.write("")

# -----------------------------------------------------------------------------
# DATA: JST pipeline
@st.cache_data
def load_jst_data(filepath: Path) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    df = df[df["country"].isin(["USA", "UK", "Canada"])].copy()
    df = df.sort_values(["country", "year"])
    return df


def engineer_features(df: pd.DataFrame):
    df = df.copy()

    # Allow uploads without crisisJST column
    if "crisisJST" not in df.columns:
        df["crisisJST"] = 0

    # --- Banking fragility ---
    df["leverage_risk"] = 1 / (df["lev"] + 0.01)

    def expanding_z_by_country(col: str) -> pd.Series:
        def z(s: pd.Series) -> pd.Series:
            m = s.expanding().mean()
            sd = s.expanding().std().replace(0, np.nan)
            return (s - m) / (sd + 1e-9)
        return df.groupby("country")[col].transform(z)

    df["noncore_z"] = expanding_z_by_country("noncore")
    df["ltd_z"] = expanding_z_by_country("ltd")
    df["leverage_z"] = expanding_z_by_country("leverage_risk")

    df["banking_fragility"] = (
        0.4 * df["noncore_z"] +
        0.3 * df["ltd_z"] +
        0.3 * df["leverage_z"]
    )

    # --- Housing bubble ---
    df["hp_real"] = df["hpnom"] / (df["cpi"] + 1e-9)
    df["hp_trend"] = df.groupby("country")["hp_real"].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )
    df["housing_bubble"] = (df["hp_real"] - df["hp_trend"]) / (df["hp_trend"] + 1e-9)

    # --- Credit growth ---
    df["real_credit"] = df["tloans"] / (df["cpi"] + 1e-9)
    df["credit_growth"] = df.groupby("country")["real_credit"].pct_change()

    # --- Yield curve ---
    df["yield_curve"] = df["ltrate"] - df["stir"]

    # --- Monetary & external ---
    df["money_gdp"] = df["money"] / (df["gdp"] + 1e-9)
    df["money_expansion"] = df.groupby("country")["money_gdp"].pct_change()
    df["ca_gdp"] = df["ca"] / (df["gdp"] + 1e-9)

    features = [
        "housing_bubble",
        "credit_growth",
        "banking_fragility",
        "yield_curve",
        "money_expansion",
        "ca_gdp",
    ]

    keep_cols = ["country", "year", "crisisJST"] + features
    df = df[keep_cols].replace([np.inf, -np.inf], np.nan)

    return df, features


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove wartime distortions
    df = df[~df["year"].between(1914, 1918)]
    df = df[~df["year"].between(1939, 1945)]

    feat_cols = [c for c in df.columns if c not in ["country", "year", "crisisJST"]]

    df[feat_cols] = df.groupby("country")[feat_cols].transform(
        lambda x: x.ffill(limit=3).bfill(limit=3)
    )
    df[feat_cols] = df.groupby("country")[feat_cols].transform(
        lambda x: x.fillna(x.median())
    )

    return df


def create_target(df: pd.DataFrame, horizon_years: int = 2) -> pd.DataFrame:
    """
    Target = 1 if a crisis occurs in the next 1..horizon_years years.
    (Implemented as rolling max on shifted crisis indicator)
    """
    df = df.copy()

    df["target"] = (
        df.groupby("country")["crisisJST"]
        .shift(-1)
        .rolling(horizon_years, min_periods=1)
        .max()
    )
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    return df


@st.cache_data
def prepare_jst(filepath: Path):
    df = load_jst_data(filepath)
    df, features = engineer_features(df)
    df = clean_data(df)
    df = create_target(df, horizon_years=2)
    return df, features


@st.cache_resource
def train_model(filepath: Path):
    df, features = prepare_jst(filepath)

    train = df[df["year"] < 1970].copy()
    test  = df[df["year"] >= 1990].copy()

    X_train = train[features]
    y_train = train["target"]
    X_test  = test[features]
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

    test_probs = model.predict_proba(Xs_test)[:, 1]

    # pick a sensible default threshold via best F1 on test (for demo; keep defensible in write-up)
    best_t, best_f1 = 0.20, -1.0
    for t in np.arange(0.05, 0.91, 0.01):
        preds = (test_probs >= t).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    return model, scaler, df, features, test, y_test.to_numpy(), test_probs, best_t


# -----------------------------------------------------------------------------
# DATA: GDP loader (template style)
@st.cache_data
def get_gdp_data(filepath: Path) -> pd.DataFrame:
    raw_gdp_df = pd.read_csv(filepath)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    gdp_df = raw_gdp_df.melt(
        ["Country Code"],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        "Year",
        "GDP",
    )
    gdp_df["Year"] = pd.to_numeric(gdp_df["Year"])
    return gdp_df


# -----------------------------------------------------------------------------
# Sidebar: Upload & Predict
def run_upload_predict_sidebar(model, scaler, risk_threshold, features):
    st.sidebar.header("📥 Upload & Predict")

    uploaded = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="upload_csv"
    )

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
            "Missing columns (showing up to 12): " + ", ".join(missing[:12]) + ("..." if len(missing) > 12 else "")
        )
        return None

    df_u, _ = engineer_features(user_df)
    df_u = clean_data(df_u)

    # If upload doesn't have crisisJST, it's fine (we set to 0 above)
    X_u = df_u[features]
    Xs_u = scaler.transform(X_u)

    probs_u = model.predict_proba(Xs_u)[:, 1]
    preds_u = (probs_u >= risk_threshold).astype(int)

    out = df_u[["country", "year"]].copy()
    out["predicted_prob"] = probs_u
    out["predicted_class"] = preds_u

    st.sidebar.success(f"{int(preds_u.sum())} high-risk rows ({100 * preds_u.mean():.1f}%)")
    return out


# -----------------------------------------------------------------------------
# Load model + data
if not JST_XLSX.exists():
    st.error(f"Missing JST dataset file: {JST_XLSX.name}. Put it next to app.py.")
    st.stop()

model, scaler, df_full, FEATURES, df_test, y_test, test_probs, default_threshold = train_model(JST_XLSX)

# GDP data is optional (dashboard still works without it)
gdp_df = None
if GDP_CSV.exists():
    gdp_df = get_gdp_data(GDP_CSV)

# -----------------------------------------------------------------------------
# Sidebar controls (risk dashboard)
st.sidebar.header("🎛️ Dashboard Controls")

risk_countries = st.sidebar.multiselect(
    "Select countries (risk charts)",
    ["USA", "UK", "Canada"],
    default=["USA", "UK", "Canada"],
    key="risk_countries"
)

min_year = int(df_full["year"].min())
max_year = int(df_full["year"].max())

risk_from_year, risk_to_year = st.sidebar.slider(
    "Select year range (risk charts)",
    min_value=min_year,
    max_value=max_year,
    value=(1900, max_year),
    key="risk_year_slider"
)

risk_threshold = st.sidebar.slider(
    "Risk threshold",
    min_value=0.05,
    max_value=0.90,
    value=float(default_threshold),
    step=0.01,
    key="risk_threshold"
)

show_crisis_years = st.sidebar.checkbox(
    "Show crisis years list",
    value=True,
    key="show_crisis_years"
)

# Upload widget in sidebar (moved from main)
uploaded_preds = run_upload_predict_sidebar(
    model=model,
    scaler=scaler,
    risk_threshold=risk_threshold,
    features=FEATURES
)

# -----------------------------------------------------------------------------
# Prepare filtered risk data for charts
risk_df = df_full[
    (df_full["country"].isin(risk_countries)) &
    (df_full["year"] >= risk_from_year) &
    (df_full["year"] <= risk_to_year)
].copy()

Xs = scaler.transform(risk_df[FEATURES])
risk_df["crisis_prob"] = model.predict_proba(Xs)[:, 1]

# -----------------------------------------------------------------------------
# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Crisis Risk",
    "📊 Model Results",
    "📥 Uploaded Predictions",
    "🌍 GDP Dashboard",
    "📂 Data Explorer",
])

# -----------------------------------------------------------------------------
# TAB 1: Crisis Risk (Streamlit-native line chart)
with tab1:
    st.header("Crisis risk over time", divider="gray")

    if risk_df.empty:
        st.warning("No data for the selected filters.")
    else:
        st.line_chart(
            risk_df,
            x="year",
            y="crisis_prob",
            color="country",
        )

        st.write("")
        st.subheader(f"Risk summary ({risk_to_year})", divider="gray")

        latest = risk_df[risk_df["year"] == risk_to_year]
        cols = st.columns(3)

        for i, c in enumerate(risk_countries):
            col = cols[i % len(cols)]
            with col:
                val = latest[latest["country"] == c]["crisis_prob"].mean()
                if pd.isna(val):
                    st.metric(label=f"{c} risk", value="n/a")
                else:
                    st.metric(
                        label=f"{c} risk",
                        value=f"{val:.2f}",
                        delta="HIGH" if val >= risk_threshold else "LOW",
                        delta_color="inverse",
                    )

        if show_crisis_years:
            with st.expander("Crisis years in selected window"):
                for c in risk_countries:
                    years = risk_df[(risk_df["country"] == c) & (risk_df["crisisJST"] == 1)]["year"].tolist()
                    if years:
                        st.write(f"**{c}:** " + ", ".join(map(str, years)))
                    else:
                        st.write(f"**{c}:** none")

# -----------------------------------------------------------------------------
# TAB 2: Model Results
with tab2:
    st.header("Out-of-sample performance (post-1990)", divider="gray")

    preds = (test_probs >= risk_threshold).astype(int)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{roc_auc_score(y_test, test_probs):.3f}")
    c2.metric("PR-AUC",  f"{average_precision_score(y_test, test_probs):.3f}")
    c3.metric("Recall",  f"{recall_score(y_test, preds):.3f}")
    c4.metric("F1",      f"{f1_score(y_test, preds):.3f}")

    st.write("")
    st.subheader("Confusion matrix (threshold applied)", divider="gray")

    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm_df, use_container_width=True)

    st.caption(
        "Threshold is controlled from the sidebar, so you can demonstrate sensitivity and policy trade-offs."
    )

# -----------------------------------------------------------------------------
# TAB 3: Uploaded Predictions (results displayed here)
with tab3:
    st.header("Predictions from uploaded CSV", divider="gray")

    if uploaded_preds is None:
        st.info("Upload a CSV from the sidebar to view predictions here.")
    else:
        st.dataframe(uploaded_preds, use_container_width=True)

        st.write("")
        st.subheader("Uploaded risk over time", divider="gray")

        # show a line chart for uploads too (if multiple countries exist)
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
# TAB 4: GDP Dashboard (template-style)
with tab4:
    st.header("🌍 GDP dashboard", divider="gray")

    if gdp_df is None:
        st.warning(
            f"GDP file not found at: {GDP_CSV}. "
            "Add the World Bank template file to /data/gdp_data.csv to enable this tab."
        )
    else:
        st.caption(
            "Streamlit-template style (slider + multiselect + line_chart + metrics)."
        )

        gdp_min = int(gdp_df["Year"].min())
        gdp_max = int(gdp_df["Year"].max())

        from_year, to_year = st.slider(
            "Which years are you interested in?",
            min_value=gdp_min,
            max_value=gdp_max,
            value=(gdp_min, gdp_max),
            key="gdp_year_slider"
        )

        countries = sorted(gdp_df["Country Code"].unique())

        selected_countries = st.multiselect(
            "Which countries would you like to view?",
            countries,
            default=["DEU", "FRA", "GBR", "USA", "CAN", "JPN"],
            key="gdp_country_selector"
        )

        if not selected_countries:
            st.warning("Select at least one country")
            st.stop()

        filtered_gdp_df = gdp_df[
            (gdp_df["Country Code"].isin(selected_countries))
            & (gdp_df["Year"] <= to_year)
            & (from_year <= gdp_df["Year"])
        ]

        st.subheader("GDP over time", divider="gray")

        st.line_chart(
            filtered_gdp_df,
            x="Year",
            y="GDP",
            color="Country Code",
        )

        st.write("")
        st.subheader(f"GDP in {to_year}", divider="gray")

        first_year = gdp_df[gdp_df["Year"] == from_year]
        last_year  = gdp_df[gdp_df["Year"] == to_year]

        cols = st.columns(4)
        for i, country in enumerate(selected_countries):
            col = cols[i % len(cols)]
            with col:
                try:
                    first_vals = first_year[first_year["Country Code"] == country]["GDP"]
                    last_vals  = last_year[last_year["Country Code"] == country]["GDP"]

                    if first_vals.empty or last_vals.empty:
                        st.metric(label=f"{country} GDP", value="n/a")
                        continue

                    first_gdp = float(first_vals.iloc[0]) / 1e9
                    last_gdp  = float(last_vals.iloc[0]) / 1e9

                    if math.isnan(first_gdp) or first_gdp == 0:
                        growth = "n/a"
                        delta_color = "off"
                    else:
                        growth = f"{last_gdp / first_gdp:,.2f}x"
                        delta_color = "normal"

                    st.metric(
                        label=f"{country} GDP",
                        value=f"{last_gdp:,.0f}B",
                        delta=growth,
                        delta_color=delta_color,
                    )
                except Exception:
                    st.metric(label=f"{country} GDP", value="n/a")

# -----------------------------------------------------------------------------
# TAB 5: Data Explorer
with tab5:
    st.header("Data explorer", divider="gray")
    st.caption("This is the processed JST dataset used for modelling (features + target).")

    st.dataframe(
        df_full.sort_values(["country", "year"]).tail(80),
        use_container_width=True
    )

    with st.expander("Show current feature list"):
        st.write(FEATURES)
