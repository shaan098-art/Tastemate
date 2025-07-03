
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report, mean_squared_error,
                             r2_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import io
import base64

st.set_page_config(page_title="Cloud Kitchen Analytics", layout="wide")

###############################################################################
# Helper functions
###############################################################################
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

def get_dataset():
    data_source = st.sidebar.radio("Data source", ["GitHub URL", "Upload CSV"])
    if data_source == "GitHub URL":
        default_url = "https://raw.githubusercontent.com/<USERNAME>/<REPO>/main/cloud_kitchen_survey_synthetic.csv"
        url = st.sidebar.text_input("Raw CSV URL", value=default_url)
        if url:
            try:
                df = load_data(url)
                st.sidebar.success("Loaded data from GitHub")
                return df
            except Exception as e:
                st.sidebar.error(f"Error loading data: {e}")
                return None
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.sidebar.success("Loaded uploaded data")
            return df
    return None

def download_link(dataframe, filename, link_text):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def prep_features(df, target, drop_multiselect=True):
    X = df.copy()
    y = X.pop(target)
    # Binary target for classification convenience
    if y.nunique() > 2:
        y = (y >= 4).astype(int)
    # Remove columns not useful
    if drop_multiselect:
        multiselect_cols = [c for c in X.columns if X[c].dtype == object and (X[c].str.contains(",").any())]
        X = X.drop(columns=multiselect_cols)
    numeric_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])
    return X, y, pre

###############################################################################
# Main App
###############################################################################
df = get_dataset()
if df is None:
    st.info("Provide a GitHub raw CSV link or upload a CSV to get started.")
    st.stop()

st.title("📊 Cloud Kitchen Analytics Dashboard")
tabs = st.tabs(["Data Visualisation", "Classification", "Clustering",
                "Association Rules", "Regression"])

###############################################################################
# Data Visualisation
###############################################################################
with tabs[0]:
    st.header("Exploratory Insights")
    # Show data preview
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    vis_options = [
        "Age distribution",
        "Diet style counts",
        "Subscribe intent by fitness goal",
        "Average spend by income bracket",
        "Workout vs Subscribe intent scatter",
        "Heatmap of correlations",
        "Orders per week distribution",
        "Spice preference counts",
        "Pause likelihood by distance",
        "NPS distribution"
    ]
    selected_vis = st.multiselect(
        "Select insights to display (10 available)", vis_options, default=vis_options)

    if "Age distribution" in selected_vis:
        st.subheader("Age Group Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="age_group", data=df, ax=ax)
        st.pyplot(fig)
        st.caption("Most respondents are in the 25‑34 age band.")

    if "Diet style counts" in selected_vis:
        st.subheader("Diet Style Popularity")
        fig, ax = plt.subplots()
        sns.countplot(x="diet_style", data=df, ax=ax)
        st.pyplot(fig)

    if "Subscribe intent by fitness goal" in selected_vis:
        st.subheader("Subscribe Likelihood vs Fitness Goal")
        fig, ax = plt.subplots()
        sns.boxplot(x="fitness_goal", y="subscribe_intent", data=df, ax=ax)
        st.pyplot(fig)

    if "Average spend by income bracket" in selected_vis:
        st.subheader("Average Spend per Meal vs Income")
        fig, ax = plt.subplots()
        sns.barplot(x="income_bracket_aed", y="avg_spend_aed", data=df, ax=ax)
        st.pyplot(fig)

    if "Workout vs Subscribe intent scatter" in selected_vis:
        st.subheader("Workouts vs Subscribe Intent")
        fig, ax = plt.subplots()
        sns.scatterplot(x="workouts_per_week", y="subscribe_intent", hue="fitness_goal", data=df, ax=ax)
        st.pyplot(fig)

    if "Heatmap of correlations" in selected_vis:
        st.subheader("Correlation Heatmap (Numerics)")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df.select_dtypes(include=["int64","float64"]).corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    if "Orders per week distribution" in selected_vis:
        st.subheader("Orders per Week Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["orders_per_week"], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    if "Spice preference counts" in selected_vis:
        st.subheader("Spice Level Preferences")
        fig, ax = plt.subplots()
        sns.countplot(x="spice_level", data=df, ax=ax)
        st.pyplot(fig)

    if "Pause likelihood by distance" in selected_vis:
        st.subheader("Pause Likelihood by Distance")
        fig, ax = plt.subplots()
        sns.boxplot(x="distance_km", y="pause_likelihood", data=df, ax=ax)
        st.pyplot(fig)

    if "NPS distribution" in selected_vis:
        st.subheader("Net Promoter Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["nps_intent"], bins=11, kde=False, ax=ax)
        st.pyplot(fig)

###############################################################################
# Classification
###############################################################################
with tabs[1]:
    st.header("Predict Subscriber Intent (Classification)")
    target = "subscribe_intent"
    X, y, pre = prep_features(df, target)

    test_size = st.slider("Test size", 0.1, 0.4, 0.3, 0.05)
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "GBRT": GradientBoostingClassifier(random_state=random_state)
    }

    results = []
    trained = {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else None
        trained[name] = (pipe, y_pred, y_prob)
        results.append({
            "Algorithm": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred)
        })
    res_df = pd.DataFrame(results).round(3)
    st.subheader("Performance Summary")
    st.dataframe(res_df)

    # Confusion matrix
    algo = st.selectbox("Select algorithm for confusion matrix", list(models.keys()))
    show_cm = st.checkbox("Show confusion matrix")
    if show_cm:
        pipe, y_pred, _ = trained[algo]
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not‑Likely", "Likely"],
                    yticklabels=["Not‑Likely", "Likely"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix – {algo}")
        st.pyplot(fig)

    # ROC
    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    for name, (pipe, _, y_prob) in trained.items():
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0,1], [0,1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    st.pyplot(fig)

    # Prediction on new data
    st.subheader("Upload New Data for Prediction")
    new_file = st.file_uploader("Upload CSV without target column", type=["csv"], key="pred_upload")
    if new_file is not None:
        new_df = pd.read_csv(new_file)
        model_choice = st.selectbox("Choose model for prediction", list(models.keys()), key="pred_model")
        pipe = trained[model_choice][0]
        preds = pipe.predict(new_df)
        pred_df = new_df.copy()
        pred_df["predicted_subscriber"] = preds
        st.dataframe(pred_df.head())
        # download
        st.markdown(download_link(pred_df, "predictions.csv", "📥 Download predictions"), unsafe_allow_html=True)

###############################################################################
# Clustering
###############################################################################
with tabs[2]:
    st.header("Customer Segmentation (K‑Means)")
    numeric_df = df.select_dtypes(include=["int64","float64"])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)

    # Elbow chart
    st.subheader("Elbow Chart")
    inertias = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(scaled)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(2,11), inertias, marker="o")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

    # Slider for K
    k = st.slider("Select number of clusters", 2, 10, 3)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["cluster"] = km.fit_predict(scaled)

    # Persona table
    st.subheader("Cluster Personas")
    persona = df.groupby("cluster").agg({
        "age_group":"median",
        "orders_per_week":"mean",
        "avg_spend_aed":"mean",
        "subscribe_intent":"mean",
        "workouts_per_week":"mean"
    }).round(2).rename(columns={
        "age_group":"Median Age Group",
        "orders_per_week":"Avg Orders/Wk",
        "avg_spend_aed":"Avg Spend (AED)",
        "subscribe_intent":"Mean Subscribe Intent",
        "workouts_per_week":"Avg Workouts/Wk"
    })
    st.dataframe(persona)

    # Download clustered data
    st.markdown(download_link(df, "clustered_data.csv", "📥 Download data with clusters"),
                unsafe_allow_html=True)

###############################################################################
# Association Rules
###############################################################################
with tabs[3]:
    st.header("Association Rule Mining (Apriori)")
    multiselect_cols = [c for c in df.columns if df[c].dtype == object and df[c].str.contains(",").any()]
    default_cols = multiselect_cols[:2]
    cols_to_use = st.multiselect("Choose columns to mine", multiselect_cols,
                                 default=default_cols)
    min_sup = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)

    if cols_to_use:
        # One‑hot encode multi‑select columns
        basket_sets = pd.DataFrame()
        for col in cols_to_use:
            exploded = df[col].str.get_dummies(sep=",")
            # Prefix column name
            exploded = exploded.add_prefix(f"{col}_")
            basket_sets = pd.concat([basket_sets, exploded], axis=1)
        # Apriori
        frequent = apriori(basket_sets.astype(bool), min_support=min_sup,
                           use_colnames=True)
        rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
        top_rules = rules.sort_values(by="confidence", ascending=False).head(10)
        st.subheader("Top 10 Rules by Confidence")
        st.dataframe(top_rules[["antecedents","consequents","support","confidence","lift"]])

###############################################################################
# Regression
###############################################################################
with tabs[4]:
    st.header("Budget Prediction (Regression)")
    target_reg = st.selectbox("Choose target numeric variable",
                              ["avg_spend_aed","max_budget_aed"])
    Xr, yr, pre_r = prep_features(df, target_reg)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        Xr, yr, test_size=0.3, random_state=42)

    reg_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }

    reg_results = []
    trained_r = {}
    for name, reg in reg_models.items():
        pipe = Pipeline([("pre", pre_r), ("reg", reg)])
        pipe.fit(X_train_r, y_train_r)
        y_pred = pipe.predict(X_test_r)
        trained_r[name] = pipe
      rmse_val = mean_squared_error(y_test_r, y_pred, squared=False)
    except TypeError:
        # Old scikit-learn (<0.22) path – manually take the square root
        rmse_val = np.sqrt(mean_squared_error(y_test_r, y_pred))
        reg_results.append({
            "Model": name,
            "R2": r2_score(y_test_r, y_pred),
            "RMSE": rmse_val
        })
    reg_df = pd.DataFrame(reg_results).round(3)
    st.subheader("Regression Performance")
    st.dataframe(reg_df)

    # Quick insights: scatter actual vs predicted
    model_sel = st.selectbox("Select model for residual plot", list(reg_models.keys()))
    pipe = trained_r[model_sel]
    y_pred_full = pipe.predict(Xr)
    fig, ax = plt.subplots()
    ax.scatter(yr, y_pred_full, alpha=0.3)
    ax.plot([yr.min(), yr.max()], [yr.min(), yr.max()], "--", color="red")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted – {model_sel}")
    st.pyplot(fig)

    st.caption("The closer the points are to the red dashed line, the better the predictions.")
