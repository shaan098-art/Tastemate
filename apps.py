# ──────────────────────────────────────────────────────────────────────────────
# apps.py  –  Streamlit dashboard for Cloud-Kitchen consumer data
# Fully self-contained, ready for Streamlit Cloud
# ──────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np

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
import base64

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cloud Kitchen Analytics", layout="wide")

# -----------------------------------------------------------------------------#
# Utilities
# -----------------------------------------------------------------------------#
@st.cache_data
def load_data(url_or_bytes):
    """Read CSV from GitHub raw link OR from BytesIO upload."""
    if isinstance(url_or_bytes, bytes):
        return pd.read_csv(io.BytesIO(url_or_bytes))
    return pd.read_csv(url_or_bytes)

def download_link(df, filename, link_text):
    """Create a download-as-CSV link."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'

def dense_onehot():
    """Return an OneHotEncoder that *always* outputs dense arrays.

    scikit-learn <=1.1 :  parameter is `sparse`
    scikit-learn >=1.2 :  parameter is `sparse_output`
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:                          # old API
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def prep_features(df, target, drop_multiselect=True):
    X = df.copy()
    y = X.pop(target)

    # turn multi-class subscribe_intent (1-5) into binary {0,1}
    if target == "subscribe_intent":
        y = (y >= 4).astype(int)

    if drop_multiselect:
        multi_cols = [c for c in X.columns
                      if X[c].dtype == object and X[c].str.contains(",").any()]
        X = X.drop(columns=multi_cols)

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", dense_onehot(), cat_cols)
    ])

    return X, y, pre

# -----------------------------------------------------------------------------#
# Sidebar – choose data source
# -----------------------------------------------------------------------------#
st.sidebar.header("Dataset")
data_choice = st.sidebar.radio("Data source",
                               ["GitHub raw CSV", "Upload CSV"])
if data_choice == "GitHub raw CSV":
    default = "https://raw.githubusercontent.com/<USERNAME>/<REPO>/main/cloud_kitchen_survey_synthetic.csv"
    data_url = st.sidebar.text_input("GitHub raw URL to CSV", value=default)
    df = load_data(data_url) if data_url else None
else:
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    df = pd.read_csv(up) if up is not None else None

if df is None:
    st.info("⬅️  Provide a CSV via GitHub raw link or upload one to begin.")
    st.stop()

# -----------------------------------------------------------------------------#
# Tabs
# -----------------------------------------------------------------------------#
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data Visualisation", "Classification", "Clustering",
     "Association Rules", "Regression"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 – Visualisation
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Exploratory Insights")
    st.dataframe(df.head())

    vis_opts = [
        "Age distribution", "Diet style counts",
        "Subscribe intent by fitness goal", "Average spend by income bracket",
        "Workout vs Subscribe intent scatter", "Heatmap of correlations",
        "Orders per week distribution", "Spice preference counts",
        "Pause likelihood by distance", "NPS distribution"
    ]
    sel = st.multiselect("Choose charts to display", vis_opts, default=vis_opts)

    if "Age distribution" in sel:
        fig, ax = plt.subplots()
        sns.countplot(x="age_group", data=df, ax=ax)
        st.pyplot(fig)

    if "Diet style counts" in sel:
        fig, ax = plt.subplots()
        sns.countplot(x="diet_style", data=df, ax=ax)
        st.pyplot(fig)

    if "Subscribe intent by fitness goal" in sel:
        fig, ax = plt.subplots()
        sns.boxplot(x="fitness_goal", y="subscribe_intent", data=df, ax=ax)
        st.pyplot(fig)

    if "Average spend by income bracket" in sel:
        fig, ax = plt.subplots()
        sns.barplot(x="income_bracket_aed", y="avg_spend_aed", data=df, ax=ax)
        st.pyplot(fig)

    if "Workout vs Subscribe intent scatter" in sel:
        fig, ax = plt.subplots()
        sns.scatterplot(x="workouts_per_week", y="subscribe_intent",
                        hue="fitness_goal", data=df, ax=ax)
        st.pyplot(fig)

    if "Heatmap of correlations" in sel:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.select_dtypes("number").corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    if "Orders per week distribution" in sel:
        fig, ax = plt.subplots()
        sns.histplot(df["orders_per_week"], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    if "Spice preference counts" in sel:
        fig, ax = plt.subplots()
        sns.countplot(x="spice_level", data=df, ax=ax)
        st.pyplot(fig)

    if "Pause likelihood by distance" in sel:
        fig, ax = plt.subplots()
        sns.boxplot(x="distance_km", y="pause_likelihood", data=df, ax=ax)
        st.pyplot(fig)

    if "NPS distribution" in sel:
        fig, ax = plt.subplots()
        sns.histplot(df["nps_intent"], bins=11, ax=ax)
        st.pyplot(fig)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 – Classification
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Predict Subscriber Intent")

    X, y, pre = prep_features(df, "subscribe_intent")

    test_size = st.slider("Test size fraction", 0.1, 0.4, 0.3, 0.05)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42)
    }

    results, trained = [], {}
    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("clf", model)])
        pipe.fit(Xtr, ytr)
        y_pred = pipe.predict(Xte)
        # some models may not have predict_proba under very old sklearn
        y_prob = (pipe.predict_proba(Xte)[:, 1]
                  if hasattr(pipe.named_steps["clf"], "predict_proba")
                  else None)
        trained[name] = (pipe, y_pred, y_prob)

        results.append({
            "Algorithm": name,
            "Accuracy": accuracy_score(yte, y_pred).round(3),
            "Precision": precision_score(yte, y_pred).round(3),
            "Recall": recall_score(yte, y_pred).round(3),
            "F1": f1_score(yte, y_pred).round(3)
        })

    st.subheader("Performance Summary")
    st.dataframe(pd.DataFrame(results))

    # Confusion matrix toggle
    algo_cm = st.selectbox("Choose algorithm for confusion matrix", list(models))
    if st.checkbox("Show confusion matrix"):
        _, y_pred_cm, _ = trained[algo_cm]
        cm = confusion_matrix(yte, y_pred_cm)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not-Likely", "Likely"],
                    yticklabels=["Not-Likely", "Likely"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        st.pyplot(fig)

    # ROC curves
    st.subheader("ROC Curves")
    fig, ax = plt.subplots()
    for name, (_, _, y_prob) in trained.items():
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(yte, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    st.pyplot(fig)

    # Predict on new data
    st.subheader("Upload new CSV (without target) for prediction")
    new_csv = st.file_uploader("Upload CSV", type="csv", key="pred_csv")
    if new_csv is not None:
        new_df = pd.read_csv(new_csv)
        model_choice = st.selectbox("Model", list(models), key="pred_model")
        pred_pipe = trained[model_choice][0]
        new_preds = pred_pipe.predict(new_df)
        out = new_df.copy()
        out["predicted_subscriber"] = new_preds
        st.dataframe(out.head())
        st.markdown(download_link(out, "predictions.csv", "📥 Download predictions"),
                    unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 – Clustering
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Customer Segmentation (K-Means)")

    num_df = df.select_dtypes("number")
    scaled = StandardScaler().fit_transform(num_df)

    # Elbow
    inertias = [KMeans(n_clusters=k, random_state=42, n_init="auto").fit(scaled).inertia_
                for k in range(2, 11)]
    fig, ax = plt.subplots()
    ax.plot(range(2, 11), inertias, marker="o")
    ax.set_xlabel("k"); ax.set_ylabel("Inertia"); ax.set_title("Elbow chart")
    st.pyplot(fig)

    k = st.slider("Choose number of clusters", 2, 10, 3)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(scaled)
    df["cluster"] = km.labels_

    st.subheader("Cluster persona table")
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

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 – Association Rules
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Association Rule Mining (Apriori)")

    multi_cols = [c for c in df.columns
                  if df[c].dtype == object and df[c].str.contains(",").any()]
    chosen_cols = st.multiselect("Multi-select columns to mine",
                                 multi_cols, default=multi_cols[:2])

    min_sup = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)

    if chosen_cols:
        basket = pd.DataFrame()
        for col in chosen_cols:
            onehot = df[col].str.get_dummies(sep=",").add_prefix(f"{col}_")
            basket = pd.concat([basket, onehot], axis=1)

        freq = apriori(basket.astype(bool), min_support=min_sup,
                       use_colnames=True)
        rules = association_rules(freq, metric="confidence",
                                  min_threshold=min_conf)
        st.subheader("Top 10 rules by confidence")
        st.dataframe(rules.sort_values("confidence", ascending=False)
                          .head(10)[["antecedents", "consequents",
                                     "support", "confidence", "lift"]])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 – Regression
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("Budget Prediction (Regression)")

    target_reg = st.selectbox("Target numeric variable",
                              ["avg_spend_aed", "max_budget_aed"])
    Xr, yr, pre_r = prep_features(df, target_reg)

    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        Xr, yr, test_size=0.3, random_state=42)

    reg_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }

    reg_results, trained_r = [], {}
    for name, reg in reg_models.items():
        pipe = Pipeline([("prep", pre_r), ("reg", reg)])
        pipe.fit(Xtr_r, ytr_r)
        y_pred = pipe.predict(Xte_r)
        trained_r[name] = pipe

        # robust RMSE no matter the sklearn version
        try:
            rmse_val = mean_squared_error(yte_r, y_pred, squared=False)
        except TypeError:  # old sklearn fallback
            rmse_val = np.sqrt(mean_squared_error(yte_r, y_pred))

        reg_results.append({
            "Model": name,
            "R2": r2_score(yte_r, y_pred).round(3),
            "RMSE": round(rmse_val, 3)
        })

    st.subheader("Regression performance")
    st.dataframe(pd.DataFrame(reg_results))

    # Residual plot
    model_sel = st.selectbox("Model for residual plot", list(reg_models))
    preds_full = trained_r[model_sel].predict(Xr)
    fig, ax = plt.subplots()
    ax.scatter(yr, preds_full, alpha=0.3)
    ax.plot([yr.min(), yr.max()], [yr.min(), yr.max()], "--", color="red")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted – {model_sel}")
    st.pyplot(fig)
