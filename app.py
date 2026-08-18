"""
SVM Binary Classification – Retail Customer Segmentation Platform
Streamlit app version of Lab 9 (Parts A-H)
Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

sns.set_style("whitegrid")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SOURCE_FILE = "retail_customer_segmentation.csv"
OUTPUT_FILE = "svm_retail_dataset.csv"

NUMERIC_FEATURES = [
    "age", "annual_income", "months_active", "avg_monthly_spend",
    "purchase_frequency", "avg_order_value", "discount_usage_rate",
    "return_rate", "browsing_time_minutes", "support_interactions",
]
CATEGORICAL_FEATURES = ["payment_method", "region"]
TARGET_RAW = "customer_segment"
TARGET = "high_engagement_customer"
HIGH_ENGAGEMENT = {"Loyal", "High_Value"}

st.set_page_config(
    page_title="SVM Retail Customer Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Cached data / model building blocks (so re-runs on widget changes are fast)
# ----------------------------------------------------------------------

@st.cache_data
def build_dataset(n_total=1000, source_file=SOURCE_FILE):
    df_full = pd.read_csv(source_file)
    df_full[TARGET] = df_full[TARGET_RAW].apply(
        lambda s: 1 if s in HIGH_ENGAGEMENT else 0
    )
    n_per_class = n_total // 2
    class1 = df_full[df_full[TARGET] == 1].sample(n=n_per_class, random_state=RANDOM_STATE)
    class0 = df_full[df_full[TARGET] == 0].sample(n=n_per_class, random_state=RANDOM_STATE)
    df = pd.concat([class1, class0], axis=0).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]]


@st.cache_data
def preprocess(df):
    df = df.drop_duplicates().reset_index(drop=True)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    X_processed = preprocessor.fit_transform(X)
    feature_names = NUMERIC_FEATURES + list(
        preprocessor.named_transformers_["cat"].named_steps["encoder"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    X_processed = pd.DataFrame(X_processed, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, X_processed, y


@st.cache_resource
def grid_search_linear(X_train, y_train):
    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(
        SVC(kernel="linear", random_state=RANDOM_STATE),
        param_grid, cv=5, scoring="f1", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


@st.cache_resource
def train_final_model(_X_train, y_train, C):
    model = SVC(kernel="linear", C=C, probability=True, random_state=RANDOM_STATE)
    model.fit(_X_train, y_train)
    return model


@st.cache_data
def c_experiment(_X_train, _X_test, y_train, y_test, C_values):
    rows = []
    for C in C_values:
        m = SVC(kernel="linear", C=C, random_state=RANDOM_STATE)
        m.fit(_X_train, y_train)
        rows.append({
            "C": C,
            "train_accuracy": round(accuracy_score(y_train, m.predict(_X_train)), 4),
            "test_accuracy": round(accuracy_score(y_test, m.predict(_X_test)), 4),
            "test_f1": round(f1_score(y_test, m.predict(_X_test)), 4),
            "n_support_vectors": len(m.support_vectors_),
        })
    return pd.DataFrame(rows)


@st.cache_data
def kernel_comparison(_X_train, _X_test, y_train, y_test):
    kernels = {
        "linear": SVC(kernel="linear", C=0.1, random_state=RANDOM_STATE),
        "rbf": SVC(kernel="rbf", C=1, gamma="scale", random_state=RANDOM_STATE),
        "poly": SVC(kernel="poly", degree=3, C=1, random_state=RANDOM_STATE),
    }
    results = {}
    for name, m in kernels.items():
        m.fit(_X_train, y_train)
        pred = m.predict(_X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        }
    return results


# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------

st.sidebar.title("SVM Lab 9 Navigation")
st.sidebar.caption("Retail Customer Segmentation Platform")
section = st.sidebar.radio(
    "Jump to section",
    [
        "Overview",
        "Part A – Dataset Creation",
        "Part B – Preprocessing",
        "Part C – Visualization",
        "Part D – SVM Model (Linear)",
        "Part E – Model Evaluation",
        "Part F – Decision Boundary",
        "Part G – Effect of C",
        "Part H – Final Analysis",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Sweezal Suares · MSc AIM · Lab 9")

# ----------------------------------------------------------------------
# Shared pipeline (runs once, cached)
# ----------------------------------------------------------------------

df = build_dataset()
X_train, X_test, y_train, y_test, X_all, y_all = preprocess(df)
grid = grid_search_linear(X_train, y_train)
best_C = grid.best_params_["C"]
model = train_final_model(X_train, y_train, best_C)

# ========================================================================
# OVERVIEW
# ========================================================================
if section == "Overview":
    st.title("SVM Binary Classification — Retail Customer Segmentation")
    st.markdown(
        """
This app walks through the full SVM lab (Parts A–H) built on the
**retail customer segmentation** dataset, predicting whether a customer
is a **high-engagement customer** (`Loyal` / `High_Value` segment) from
behavioural and spend features.

Use the sidebar to jump between Parts A through H. All computation is
cached, so switching sections is fast after the first load.
        """
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{df.shape[0]:,}")
    col2.metric("Features", f"{X_all.shape[1]}")
    col3.metric("Best C (Linear)", f"{best_C}")
    y_pred_preview = model.predict(X_test)
    col4.metric("Test Accuracy", f"{accuracy_score(y_test, y_pred_preview):.2%}")

    st.info(
        "Binary target: `high_engagement_customer` = 1 if segment is "
        "Loyal/High_Value, else 0 (Occasional/Regular). Sampled to a "
        "balanced 500/500 from the full 50,000-row source dataset."
    )

# ========================================================================
# PART A
# ========================================================================
elif section == "Part A – Dataset Creation":
    st.title("Part A – Dataset Creation")
    st.caption(
        "Generate ~1,000 instances, 5+ predictors, 1 binary target, "
        "meaningful names, balanced classes."
    )

    st.subheader("Predictor & target variables")
    c1, c2 = st.columns(2)
    c1.write("**Numeric predictors:**")
    c1.code("\n".join(NUMERIC_FEATURES))
    c2.write("**Categorical predictors:**")
    c2.code("\n".join(CATEGORICAL_FEATURES))
    st.write(f"**Target variable:** `{TARGET}`")

    st.subheader("First 10 records")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Dataset shape")
    st.write(df.shape)

    st.subheader("Data types")
    st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)

    st.subheader("Class distribution")
    dist_col1, dist_col2 = st.columns(2)
    dist_col1.dataframe(df[TARGET].value_counts().rename("count"))
    dist_col2.dataframe(
        (df[TARGET].value_counts(normalize=True) * 100).round(2).rename("percent")
    )

    st.subheader("Descriptive statistics")
    st.dataframe(df.describe(include="all"), use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download generated dataset (svm_retail_dataset.csv)",
        data=csv_bytes,
        file_name=OUTPUT_FILE,
        mime="text/csv",
    )

# ========================================================================
# PART B
# ========================================================================
elif section == "Part B – Preprocessing":
    st.title("Part B – Data Preprocessing")

    st.subheader("Missing values per column")
    st.dataframe(df.isnull().sum().rename("missing_count"), use_container_width=True)

    n_dupes = df.duplicated().sum()
    st.subheader("Duplicate records")
    st.write(f"**{n_dupes}** duplicate records found (dropped before splitting).")

    st.subheader("Attribute types identified")
    c1, c2 = st.columns(2)
    c1.write("**Categorical:**")
    c1.code("\n".join(CATEGORICAL_FEATURES))
    c2.write("**Numeric:**")
    c2.code("\n".join(NUMERIC_FEATURES))

    st.subheader("Encoding, scaling & split")
    st.write(f"X shape after imputation/encoding/scaling: **{X_all.shape}**")
    st.write(f"Train set: **{X_train.shape}**  |  Test set: **{X_test.shape}**")
    st.caption("80/20 split, stratified on the target, random_state=42.")

    with st.expander("Why is feature scaling important for SVM?", expanded=True):
        st.markdown(
            """
SVM finds the maximum-margin hyperplane using distances (dot products)
between points. A feature measured in large raw units (e.g.
`annual_income` in the tens of thousands) would numerically dominate a
feature like `return_rate` (0–1), and the margin would be driven almost
entirely by scale rather than actual predictive value. Standardizing
every feature to mean 0 / std 1 lets each feature contribute based on
its real relationship with the target, not its arbitrary unit.
            """
        )

# ========================================================================
# PART C
# ========================================================================
elif section == "Part C – Visualization":
    st.title("Part C – Exploratory Data Visualization")

    st.subheader("1. Class distribution")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x=TARGET, hue=TARGET, palette="Set2", legend=False, ax=ax)
    ax.set_title("Class Distribution: High Engagement Customer")
    ax.set_xlabel("0 = Occasional/Regular, 1 = Loyal/High_Value")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("2. Scatter plot — two important features")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=df, x="purchase_frequency", y="months_active",
        hue=TARGET, palette="Set1", alpha=0.6, ax=ax
    )
    ax.set_title("purchase_frequency vs months_active")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("3. Pairplot of selected features")
    pair_features = ["purchase_frequency", "months_active", "avg_order_value", "avg_monthly_spend"]
    with st.spinner("Rendering pairplot..."):
        pp = sns.pairplot(
            df[pair_features + [TARGET]], hue=TARGET, palette="Set1",
            plot_kws={"alpha": 0.5, "s": 20}
        )
        st.pyplot(pp.figure)
        plt.close(pp.figure)

    st.subheader("4. Correlation heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[NUMERIC_FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Observations from visualizations")
    st.markdown(
        """
1. **Class distribution:** the target variable is perfectly balanced (500/500).
2. **Scatter plot & pairplot:** significant overlap between classes suggests
   non-linear separability, hinting non-linear kernels might help.
3. **Correlation heatmap:** `months_active` (+0.47), `purchase_frequency`
   (+0.49), and `avg_order_value` (-0.35) show the strongest relationships
   with the target — these are the most discriminative features.
        """
    )

# ========================================================================
# PART D
# ========================================================================
elif section == "Part D – SVM Model (Linear)":
    st.title("Part D – SVM Model Development (Linear Kernel)")

    st.write(f"**Grid search over C:** {[0.01, 0.1, 1, 10, 100]}")
    st.write(f"**Best C (5-fold CV, F1-scoring):** {best_C}")
    st.write(f"**Best CV F1-score:** {grid.best_score_:.4f}")

    st.subheader("Full grid search results")
    cv_results = pd.DataFrame(grid.cv_results_)[
        ["param_C", "mean_test_score", "std_test_score", "rank_test_score"]
    ].rename(columns={"param_C": "C", "mean_test_score": "mean_F1", "std_test_score": "std_F1"})
    st.dataframe(cv_results, use_container_width=True)

    st.success(f"Final model trained: `SVC(C={best_C}, kernel='linear')`")

# ========================================================================
# PART E
# ========================================================================
elif section == "Part E – Model Evaluation":
    st.title("Part E – Model Evaluation")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision", f"{prec:.4f}")
    c3.metric("Recall", f"{rec:.4f}")
    c4.metric("F1-score", f"{f1:.4f}")

    st.subheader("Classification report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"], ax=ax)
    ax.set_title("Confusion Matrix - Linear SVM")
    st.pyplot(fig)
    plt.close(fig)

# ========================================================================
# PART F
# ========================================================================
elif section == "Part F – Decision Boundary":
    st.title("Part F – SVM Decision Boundary Visualization")

    FEATURE_X, FEATURE_Y = "purchase_frequency", "months_active"

    data2 = SimpleImputer(strategy="median").fit_transform(df[[FEATURE_X, FEATURE_Y]])
    X2 = StandardScaler().fit_transform(data2)
    y2 = df[TARGET].values

    model2d = SVC(kernel="linear", C=0.1, random_state=RANDOM_STATE)
    model2d.fit(X2, y2)

    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model2d.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z > 0, alpha=0.15, cmap="coolwarm")
    ax.contour(xx, yy, Z, colors="k", levels=[-1, 0, 1], linestyles=["--", "-", "--"], linewidths=1.2)
    ax.scatter(X2[y2 == 0, 0], X2[y2 == 0, 1], c="royalblue", s=25, alpha=0.6, label="Class 0 (Occasional/Regular)")
    ax.scatter(X2[y2 == 1, 0], X2[y2 == 1, 1], c="crimson", s=25, alpha=0.6, label="Class 1 (Loyal/High_Value)")
    ax.scatter(model2d.support_vectors_[:, 0], model2d.support_vectors_[:, 1],
               s=110, facecolors="none", edgecolors="black", linewidths=1.5, label="Support Vectors")
    ax.set_xlabel(f"{FEATURE_X} (scaled)")
    ax.set_ylabel(f"{FEATURE_Y} (scaled)")
    ax.set_title("Linear SVM Decision Boundary (2 features)")
    ax.legend(loc="best")
    st.pyplot(fig)
    plt.close(fig)

    st.write(f"**Number of support vectors:** {len(model2d.support_vectors_)}")

    st.subheader("SVM decision boundary explanation")
    st.markdown(
        """
1. **Hyperplane / Decision Boundary:** separates data points into different
   classes within the feature space — a dividing line in 2D, a plane in 3D.
   Points on one side are classified as one class, points on the other as
   the second class. Its position is determined by the support vectors.
2. **Support Vectors:** training data points closest to the hyperplane,
   critical for defining the boundary (circled in the plot above).
3. **Margin:** the region between the two parallel dashed hyperplanes
   through the closest support vectors — maximized for robust separation.
4. **Importance of support vectors:** only support vectors influence the
   decision boundary; every other point could be removed without changing
   it, making SVM efficient and robust.
5. **How the boundary separates classes:** a new point is classified by
   which side of the solid hyperplane line it falls on (the sign of the
   decision function).
        """
    )

# ========================================================================
# PART G
# ========================================================================
elif section == "Part G – Effect of C":
    st.title("Part G – Effect of Hyperparameter C")

    st.caption("Try your own C values, or use the default experiment (0.01, 1, 100).")
    c_input = st.text_input("C values (comma-separated)", value="0.01, 1, 100")
    try:
        C_values = tuple(float(x.strip()) for x in c_input.split(",") if x.strip())
    except ValueError:
        st.error("Please enter valid numbers separated by commas.")
        C_values = (0.01, 1, 100)

    g_results = c_experiment(X_train, X_test, y_train, y_test, C_values)
    st.dataframe(g_results, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(g_results["C"], g_results["train_accuracy"], marker="o", label="Train accuracy")
    ax.plot(g_results["C"], g_results["test_accuracy"], marker="o", label="Test accuracy")
    ax.set_xscale("log")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Effect of C on train vs test accuracy")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Experiment 1 — effect of C on Linear SVM")
    st.markdown(
        """
Increasing `C` (penalty for misclassification) in a linear SVM generally
leads to a **smaller margin**, **fewer training misclassifications**, and
a **decrease in the number of support vectors**, but also a **higher risk
of overfitting**. Conversely, a low `C` results in a **larger margin**,
**more misclassifications**, and potential **underfitting**.

From the results table above, increasing C improves training accuracy but
shows fluctuating test performance, indicating that a moderately low C
(like the 0.1 found by GridSearchCV in Part D) can be optimal for
generalization.
        """
    )

# ========================================================================
# PART H
# ========================================================================
elif section == "Part H – Final Analysis":
    st.title("Part H – Final Analysis")

    kresults = kernel_comparison(X_train, X_test, y_train, y_test)
    kdf = pd.DataFrame(kresults).T.rename_axis("kernel").reset_index()
    st.subheader("Kernel comparison")
    st.dataframe(kdf, use_container_width=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(kdf["kernel"], kdf["f1"], color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("Test F1-score")
    ax.set_title("Kernel comparison (F1-score)")
    st.pyplot(fig)
    plt.close(fig)

    best_kernel = max(kresults, key=lambda k: kresults[k]["f1"])

    st.markdown(f"""
**1. Which SVM kernel produced the best classification performance?**
Based on F1-scores, the **{best_kernel.upper()}** kernel achieved the best
classification performance with an F1-score of **{kresults[best_kernel]['f1']:.4f}**.
The linear SVM scored {kresults['linear']['f1']:.4f}, and the polynomial
SVM scored {kresults['poly']['f1']:.4f}.

**2. Why did the selected kernel perform better than the other kernels?**
The RBF kernel likely performed better because the decision boundary
between classes is non-linear and complex. Given the scatter and pair
plots in Part C, classes were not perfectly linearly separable. RBF maps
data into a higher-dimensional space to capture these non-linear
relationships effectively; the linear kernel is limited to linear
boundaries, while the polynomial kernel wasn't as well matched to the
actual shape of the overlap here.

**3. Which features appear to be most useful for classification?**
The correlation heatmap indicates that `months_active`, `purchase_frequency`,
and `avg_order_value` are most useful — they show the highest absolute
correlation with the target variable. The scatter plot of
`purchase_frequency` vs `months_active` visually supports these features
as discriminative.

**4. Why is feature scaling important in SVM?**
Feature scaling is crucial because SVM calculates distances between data
points. Without scaling, features with larger numerical values (e.g.
`annual_income`) could disproportionately influence these distance
calculations, biasing the model. Scaling ensures all features contribute
equally to the hyperplane position.

**5. How does the value of C affect the SVM model?**
`C` controls the trade-off between maximizing the margin and minimizing
classification errors. A small `C` leads to a wider margin and more
misclassifications (higher bias / underfitting risk). A large `C` strives
for fewer misclassifications on training data with a smaller margin
(higher variance / overfitting risk).

**6. How does gamma affect an SVM?**
`gamma` (for non-linear kernels like RBF) controls the influence of
individual training examples. A small gamma gives a smoother, simpler
boundary (underfitting risk); a large gamma gives a highly complex,
wiggly boundary tailored to individual points (overfitting risk).

**7. What happens when C is excessively high?**
The model heavily penalizes any misclassification, creating an extremely
complex boundary that fits training data (including noise) almost
perfectly. Margin becomes very narrow, training error is minimized, but
generalization to unseen data suffers.

**8. What happens when gamma is excessively high?**
Each training point gets a tiny radius of influence, so the model builds
a boundary that wraps tightly around individual points — essentially
memorizing the training set. Performance on new data drops sharply.

**9. How do support vectors influence the final decision boundary?**
Support vectors are the critical points closest to (or on the wrong side
of) the decision boundary. They alone determine the position and
orientation of the hyperplane and margin, making the SVM boundary robust
and efficient — every other training point could be removed without
changing it.

**10. Is the final model overfitting or underfitting? Justify your answer.**
The final model ({best_kernel.upper()} kernel, F1={kresults[best_kernel]['f1']:.4f})
shows reasonable, balanced test performance rather than a large train-test
gap. Combined with the genuinely overlapping feature space seen in Part C,
this points to a reasonably well-fit model rather than severe overfitting
or underfitting — remaining error reflects real overlap between
Loyal/High_Value and Occasional/Regular customers on these features.
""")
