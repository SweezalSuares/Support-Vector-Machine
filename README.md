# SVM Retail Customer Segmentation — Streamlit App

Interactive Streamlit version of the Lab 9 SVM binary classification
assignment (Parts A–H), built on `retail_customer_segmentation.csv`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure `retail_customer_segmentation.csv` is in the same folder as
`app.py` — it's the data source the app reads on startup.

## What's inside

- Sidebar navigation between Part A (dataset creation) through Part H
  (final analysis), plus an Overview page.
- All data loading, preprocessing, and model training is cached
  (`@st.cache_data` / `@st.cache_resource`) so switching between
  sections after the first load is fast.
- Part G lets you type your own comma-separated C values to re-run the
  hyperparameter experiment interactively.
- Every plot from the original notebook (class distribution, scatter,
  pairplot, correlation heatmap, confusion matrix, decision boundary,
  C vs accuracy curve, kernel comparison bar chart) is reproduced.

