
# Cloud Kitchen Analytics Dashboard

Streamlit dashboard for analysing consumer survey data for the **Data‑Ready Cloud Kitchen** concept.

## ✨ Features
* **Data Visualisation** – 10 ready‑made exploratory charts with filters.
* **Classification** – KNN, Decision Tree, Random Forest, Gradient Boosting + metrics, confusion matrix toggle, ROC curves. Upload new data to predict and download results.
* **Clustering** – K‑means with dynamic cluster slider, elbow chart, persona table, downloadable cluster‑labelled data.
* **Association Rules** – Apriori mining on multi‑select columns with adjustable support & confidence.
* **Regression** – Linear, Ridge, Lasso, Decision‑Tree regressors with performance table and residual plot.

Everything lives in a single `apps.py` for plug‑and‑play deployment on **Streamlit Cloud**.

## 🚀 Quick start

```bash
# clone your repo & install deps
pip install -r requirements.txt

# run locally
streamlit run apps.py
```

## 📂 Data
The app expects the CSV at a GitHub **raw‑content** URL (default set in the sidebar)  
or you can upload a file via the UI.

## 🗂️ Structure
```
.
├── apps.py            # Streamlit application – all code here
├── requirements.txt   # Python dependencies
└── README.md          # This guide
```

## 🌐 Deploy on Streamlit Cloud
1. Create a new repo containing these three files **plus** your dataset CSV.  
2. On [streamlit.io](https://share.streamlit.io/), click **New app**, pick the repo and set `apps.py` as the entry point.  
3. Done – Streamlit Cloud builds from `requirements.txt` and serves your dashboard!

Enjoy exploring your cloud‑kitchen data 🔥
