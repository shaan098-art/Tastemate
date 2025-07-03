# ──────────────────────────────────────────────────────────────────────────────
# apps.py – Streamlit dashboard for Cloud-Kitchen consumer data
# All five tabs in a single file – deploy directly on Streamlit Cloud
# ──────────────────────────────────────────────────────────────────────────────
import io
import base64
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             mean_squared_error, r2_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Cloud Kitchen Analytics", layout="wide")

# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(csv_path):
    return pd.read_csv(csv_path)

def download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

def dense_onehot():
    """Return OneHotEncoder that always outputs dense arrays, no matter the version."""
    try:        # scikit-learn ≥1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:                 # scikit-learn ≤1.1
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def prep_features(df, target, drop_multiselect=True):
    X = df.copy()
    y = X.pop(target)
    if target == "subscribe_intent":          # binarise for classification
        y = (y >= 4).astype(int)

    if drop_multiselect:
        ms_cols = [c for c in X.columns
                   if X[c].dtype == object and X[c].str.contains(",").any()]
        X = X.drop(columns=ms_cols)

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", dense_onehot(),   cat_cols)
    ])
    return X, y, pre

# ═════════════════════════════════════════════════════════════════════════════
# Sidebar – data source
# ═════════════════════════════════════════════════════════════════════════════
st.sidebar.header("Dataset")
src = st.sidebar.radio("Data source", ["GitHub raw CSV", "Upload CSV"])

if src == "GitHub raw CSV":
    default_url = ("https://raw.githubusercontent.com/<USERNAME>/<REPO>"
                   "/main/cloud_kitchen_survey_synthetic.csv")
    url = st.sidebar.text_input("Raw CSV URL", value=default_url)
    df = load_data(url) if url else None
else:
    up = st.sidebar.file_uploader("Upload CSV", type="csv")
    df = pd.read_csv(up) if up is not None else None

if df is None:
    st.info("⬅️  Provide a CSV via link or upload to begin.")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# Tabs
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data Visualisation", "Classification", "Clustering",
     "Association Rules", "Regression"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – Visualisation
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Exploratory Insights")
    st.dataframe(df.head())

    charts = [
        "Age distribution", "Diet style counts",
        "Subscribe intent by fitness goal", "Average spend by income bracket",
        "Workout vs Subscribe intent scatter", "Heatmap of correlations",
        "Orders per week distribution", "Spice preference counts",
        "Pause likelihood by distance", "NPS distribution"
    ]
    show = st.multiselect("Choose insights", charts, default=charts)

    if "Age distribution" in show:
        fig, ax = plt.subplots()
        sns.countplot(x="age_group", data=df, ax=ax)
        st.pyplot(fig)

    if "Diet style counts" in show:
        fig, ax = plt.subplots()
        sns.countplot(x="diet_style", data=df, ax=ax)
        st.pyplot(fig)

    if "Subscribe intent by fitness goal" in show:
        fig, ax = plt.subplots()
        sns.boxplot(x="fitness_goal", y="subscribe_intent", data=df, ax=ax)
        st.pyplot(fig)

    if "Average spend by income bracket" in show:
        fig, ax = plt.subplots()
        sns.barplot(x="income_bracket_aed", y="avg_spend_aed", data=df, ax=ax)
        st.pyplot(fig)

    if "Workout vs Subscribe intent scatter" in show:
        fig, ax = plt.subplots()
        sns.scatterplot(x="workouts_per_week", y="subscribe_intent",
                        hue="fitness_goal", data=df, ax=ax)
        st.pyplot(fig)

    if "Heatmap of correlations" in show:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.select_dtypes("number").corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    if "Orders per week distribution" in show:
        fig, ax = plt.subplots()
        sns.histplot(df["orders_per_week"], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    if "Spice preference counts" in show:
        fig, ax = plt.subplots()
        sns.countplot(x="spice_level", data=df, ax=ax)
        st.pyplot(fig)

    if "Pause likelihood by distance" in show:
        fig, ax = plt.subplots()
        sns.boxplot(x="distance_km", y="pause_likelihood", data=df, ax=ax)
        st.pyplot(fig)

    if "NPS distribution" in show:
        fig, ax = plt.subplots()
        sns.histplot(df["nps_intent"], bins=11, ax=ax)
        st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – Classification
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Predict Subscriber Intent")

    X, y, pre = prep_features(df, "subscribe_intent")
    split = st.slider("Test set size", 0.1, 0.4, 0.3, 0.05)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=split, random_state=42, stratify=y)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42)
    }

    results, trained = [], {}
    y_true = np.asarray(yte).ravel().astype(int)   # ensure clean 1-D int array

    for name, mdl in models.items():
        pipe = Pipeline([("prep", pre), ("clf", mdl)])
        pipe.fit(Xtr, ytr)

        y_pred = np.asarray(pipe.predict(Xte)).ravel().astype(int)
        y_prob = (pipe.predict_proba(Xte)[:, 1]
                  if hasattr(mdl, "predict_proba") else None)

        trained[name] = {"pipe": pipe, "y_pred": y_pred, "y_prob": y_prob}

        results.append({
            "Algorithm": name,
            "Accuracy":  accuracy_score(y_true, y_pred).round(3),
            "Precision": precision_score(y_true, y_pred).round(3),
            "Recall":    recall_score(y_true, y_pred).round(3),
            "F1":        f1_score(y_true, y_pred).round(3)
        })

    st.subheader("Performance Summary")
    st.dataframe(pd.DataFrame(results))

    algo_cm = st.selectbox("Algorithm for confusion matrix", list(models))
    if st.checkbox("Show confusion matrix"):
        cm = confusion_matrix(y_true, trained[algo_cm]["y_pred"])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not-Likely", "Likely"],
                    yticklabels=["Not-Likely", "Likely"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        st.pyplot(fig)

    st.subheader("ROC Curves")
    fig, ax = plt.subplots()
    for name, t in trained.items():
        if t["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(y_true, t["y_prob"])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    st.pyplot(fig)

    st.subheader("Upload new CSV to predict")
    new_csv = st.file_uploader("CSV without target column", type="csv")
    if new_csv is not None:
        new_df = pd.read_csv(new_csv)
        mdl_choice = st.selectbox("Model", list(models), key="pred_model")
        pipe = trained[mdl_choice]["pipe"]
        preds = pipe.predict(new_df)
        out = new_df.copy()
        out["predicted_subscriber"] = preds
        st.dataframe(out.head())
        st.markdown(download_link(out, "predictions.csv", "📥 Download predictions"),
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – Clustering
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Customer Segmentation (K-Means)")

    num_df = df.select_dtypes("number")
    scaled = StandardScaler().fit_transform(num_df)

    inertias = [KMeans(k, n_init="auto", random_state=42).fit(scaled).inertia_
                for k in range(2, 11)]
    fig, ax = plt.subplots()
    ax.plot(range(2, 11), inertias, marker="o")
    ax.set_xlabel("k"); ax.set_ylabel("Inertia"); ax.set_title("Elbow chart")
    st.pyplot(fig)

    k = st.slider("Number of clusters", 2, 10, 3)
    km = KMeans(k, n_init="auto", random_state=42).fit(scaled)
    df["cluster"] = km.labels_

    st.subheader("Cluster personas")
    persona = df.groupby("cluster").agg({
        "age_group": "median",
        "orders_per_week": "mean",
        "avg_spend_aed": "mean",
        "subscribe_intent": "mean",
        "workouts_per_week": "mean"
    }).round(2).rename(columns={
        "age_group": "Median Age Group",
        "orders_per_week": "Avg Orders/Wk",
        "avg_spend_aed": "Avg Spend (AED)",
        "subscribe_intent": "Mean Subscribe Intent",
        "workouts_per_week": "Avg Workouts/Wk"
    })
    st.dataframe(persona)
    st.markdown(download_link(df, "clustered_data.csv",
                              "📥 Download data with clusters"),
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – Association Rules
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Association Rule Mining (Apriori)")

    multi_cols = [c for c in df.columns
                  if df[c].dtype == object and df[c].str.contains(",").any()]
    choose = st.multiselect("Columns to mine", multi_cols, default=multi_cols[:2])
    min_sup = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)

    if choose:
        basket = pd.DataFrame()
        for col in choose:
            dummies = df[col].str.get_dummies(sep=",").add_prefix(f"{col}_")
            basket = pd.concat([basket, dummies], axis=1)

        freq = apriori(basket.astype(bool), min_support=min_sup,
                       use_colnames=True)
        rules = association_rules(freq, "confidence", min_conf)
        top10 = rules.sort_values("confidence", ascending=False).head(10)
        st.dataframe(top10[["antecedents", "consequents",
                            "support", "confidence", "lift"]])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 – Regression
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("Budget Prediction (Regression)")

    target = st.selectbox("Target numeric variable",
                          ["avg_spend_aed", "max_budget_aed"])
    Xr, yr, pre_r = prep_features(df, target)

    Xtr, Xte, ytr, yte = train_test_split(
        Xr, yr, test_size=0.3, random_state=42)

    regs = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }

    reg_results, trained_r = [], {}
    for name, reg in regs.items():
        pipe = Pipeline([("prep", pre_r), ("reg", reg)])
        pipe.fit(Xtr, ytr)
        y_pred = pipe.predict(Xte)
        trained_r[name] = pipe

        try:
            rmse = mean_squared_error(yte, y_pred, squared=False)
        except TypeError:
            rmse = np.sqrt(mean_squared_error(yte, y_pred))

        reg_results.append({
            "Model": name,
            "R2": r2_score(yte, y_pred).round(3),
            "RMSE": round(rmse, 3)
        })

    st.subheader("Regression performance")
    st.dataframe(pd.DataFrame(reg_results))

    mdl_plot = st.selectbox("Model for residual plot", list(regs))
    preds = trained_r[mdl_plot].predict(Xr)
    fig, ax = plt.subplots()
    ax.scatter(yr, preds, alpha=0.3)
    ax.plot([yr.min(), yr.max()], [yr.min(), yr.max()], "--", color="red")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted — {mdl_plot}")
    st.pyplot(fig)
