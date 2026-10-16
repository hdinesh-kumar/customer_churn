# ============================================================
# CUSTOMER CHURN INTELLIGENCE SYSTEM
# MACHINE LEARNING PROJECT - GOOGLE COLAB
# ============================================================


# ============================================================
# CELL 1 - IMPORT LIBRARIES
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from google.colab import drive, files

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef
)

print("Libraries imported successfully!")


# ============================================================
# CELL 2 - MOUNT GOOGLE DRIVE
# ============================================================

drive.mount("/content/drive")

print("Google Drive mounted successfully!")


# ============================================================
# CELL 3 - LOAD DATASET
# ============================================================

file_path = "/content/drive/MyDrive/ML/WA_Fn-UseC_-Telco-Customer-Churn.csv"

if os.path.exists(file_path):

    df = pd.read_csv(file_path)

    print("Dataset loaded from Google Drive!")

else:

    print("Dataset not found in Google Drive.")
    print("Please upload the CSV file.")

    uploaded = files.upload()

    uploaded_file = list(uploaded.keys())[0]

    df = pd.read_csv(uploaded_file)

    print("Dataset uploaded successfully!")


# ============================================================
# CELL 4 - BASIC DATASET INFORMATION
# ============================================================

print("\n================ DATASET SHAPE ================\n")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


print("\n================ FIRST 5 ROWS ================\n")

display(df.head())


print("\n================ COLUMN NAMES ================\n")

print(df.columns.tolist())


print("\n================ DATA TYPES ================\n")

display(
    pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })
)


print("\n================ DATASET INFORMATION ================\n")

df.info()


# ============================================================
# CELL 5 - CHECK MISSING VALUES
# ============================================================

print("\n================ MISSING VALUES BEFORE CLEANING ================\n")

display(
    df.isnull()
    .sum()
    .to_frame("Missing Values")
)


# ============================================================
# CELL 6 - DATA CLEANING
# ============================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

missing_totalcharges = df["TotalCharges"].isnull().sum()

print(
    "Missing values detected in TotalCharges:",
    missing_totalcharges
)

median_totalcharges = df["TotalCharges"].median()

df["TotalCharges"] = df["TotalCharges"].fillna(
    median_totalcharges
)

print(
    "Median used to fill missing TotalCharges:",
    round(median_totalcharges, 2)
)

print("\nMissing values after cleaning:")

display(
    df.isnull()
    .sum()
    .to_frame("Missing Values")
)


# ============================================================
# CELL 7 - CHECK DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)


# ============================================================
# CELL 8 - DESCRIPTIVE STATISTICS
# ============================================================

print("\n================ NUMERICAL STATISTICS ================\n")

display(
    df[
        [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]
    ].describe()
)


# ============================================================
# CELL 9 - CHURN DISTRIBUTION
# ============================================================

churn_count = df["Churn"].value_counts()

churn_percentage = (
    df["Churn"]
    .value_counts(normalize=True)
    * 100
)

churn_summary = pd.DataFrame({
    "Customer Count": churn_count,
    "Percentage": churn_percentage
})

print("\n================ CHURN DISTRIBUTION ================\n")

display(
    churn_summary.round(2)
)


# ============================================================
# CELL 10 - CHURN COUNT GRAPH
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title(
    "Customer Churn Distribution",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Churn Status")

plt.ylabel("Number of Customers")

plt.show()


# ============================================================
# CELL 11 - CHURN PERCENTAGE PIE CHART
# ============================================================

plt.figure(figsize=(7, 7))

df["Churn"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title(
    "Churn Percentage",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel("")

plt.show()


# ============================================================
# CELL 12 - CONTRACT VS CHURN
# ============================================================

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\n================ CONTRACT VS CHURN ================\n")

display(
    contract_churn.round(2)
)

contract_churn.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title(
    "Contract Type vs Churn",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Contract Type")

plt.ylabel("Percentage of Customers")

plt.xticks(rotation=0)

plt.legend(title="Churn")

plt.tight_layout()

plt.show()


# ============================================================
# CELL 13 - INTERNET SERVICE VS CHURN
# ============================================================

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print("\n================ INTERNET SERVICE VS CHURN ================\n")

display(
    internet_churn.round(2)
)

internet_churn.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title(
    "Internet Service vs Churn",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Internet Service")

plt.ylabel("Percentage of Customers")

plt.xticks(rotation=0)

plt.legend(title="Churn")

plt.tight_layout()

plt.show()


# ============================================================
# CELL 14 - TENURE VS CHURN
# ============================================================

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    kde=True,
    multiple="layer"
)

plt.title(
    "Customer Tenure Distribution by Churn",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Tenure in Months")

plt.ylabel("Number of Customers")

plt.show()


# ============================================================
# CELL 15 - MONTHLY CHARGES VS CHURN
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title(
    "Monthly Charges vs Churn",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Churn")

plt.ylabel("Monthly Charges")

plt.show()


# ============================================================
# CELL 16 - TOTAL CHARGES VS CHURN
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="TotalCharges"
)

plt.title(
    "Total Charges vs Churn",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Churn")

plt.ylabel("Total Charges")

plt.show()


# ============================================================
# CELL 17 - PAYMENT METHOD VS CHURN
# ============================================================

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print("\n================ PAYMENT METHOD VS CHURN ================\n")

display(
    payment_churn.round(2)
)

payment_churn.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title(
    "Payment Method vs Churn",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Payment Method")

plt.ylabel("Percentage of Customers")

plt.xticks(
    rotation=25,
    ha="right"
)

plt.legend(title="Churn")

plt.tight_layout()

plt.show()


# ============================================================
# CELL 18 - CORRELATION HEATMAP
# ============================================================

numeric_data = df[
    [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]
]

plt.figure(figsize=(8, 6))

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Correlation Between Numerical Features",
    fontsize=14,
    fontweight="bold"
)

plt.show()


# ============================================================
# CELL 19 - CREATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[
        "customerID",
        "Churn"
    ]
)

y = df["Churn"]

print("Feature dataset shape:", X.shape)

print("Target shape:", y.shape)


# ============================================================
# CELL 20 - IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical Columns:")

print(categorical_columns)


print("\nNumerical Columns:")

print(numerical_columns)


# ============================================================
# CELL 21 - TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n================ TRAIN TEST SPLIT ================\n")

print("Training records :", X_train.shape[0])

print("Testing records  :", X_test.shape[0])


print("\nTraining target distribution:")

display(
    y_train.value_counts()
)


print("\nTesting target distribution:")

display(
    y_test.value_counts()
)


# ============================================================
# CELL 22 - PREPROCESSING
# ============================================================

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

numerical_transformer = StandardScaler()


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_transformer,
            categorical_columns
        ),

        (
            "numerical",
            numerical_transformer,
            numerical_columns
        )
    ]
)


# ============================================================
# CELL 23 - LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ============================================================
# CELL 24 - CREATE MACHINE LEARNING PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            model
        )
    ]
)


print("Machine Learning Pipeline created successfully!")


# ============================================================
# CELL 25 - TRAIN MODEL
# ============================================================

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed successfully!")


# ============================================================
# CELL 26 - MODEL PREDICTION
# ============================================================

y_pred = pipeline.predict(
    X_test
)


classes = list(
    pipeline.classes_
)


yes_index = classes.index(
    "Yes"
)


y_probability = pipeline.predict_proba(
    X_test
)[:, yes_index]


print("Prediction completed successfully!")


# ============================================================
# CELL 27 - BASIC MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "Pipeline Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# CELL 28 - CALCULATE ALL EVALUATION METRICS
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


y_test_binary = (
    y_test == "Yes"
).astype(int)


roc_auc = roc_auc_score(
    y_test_binary,
    y_probability
)


average_precision = average_precision_score(
    y_test_binary,
    y_probability
)


mcc = matthews_corrcoef(
    y_test,
    y_pred
)


# ============================================================
# CELL 29 - CONFUSION MATRIX VALUES
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        "No",
        "Yes"
    ]
)

tn, fp, fn, tp = cm.ravel()


specificity = tn / (
    tn + fp
)


negative_predictive_value = tn / (
    tn + fn
)


false_positive_rate = fp / (
    fp + tn
)


false_negative_rate = fn / (
    fn + tp
)


# ============================================================
# CELL 30 - DISPLAY EVALUATION METRICS
# ============================================================

metrics_data = pd.DataFrame({

    "Evaluation Metric": [

        "Accuracy",

        "Precision",

        "Recall / Sensitivity",

        "F1 Score",

        "Specificity",

        "Balanced Accuracy",

        "ROC-AUC Score",

        "Average Precision",

        "Matthews Correlation Coefficient",

        "Negative Predictive Value",

        "False Positive Rate",

        "False Negative Rate"

    ],

    "Value": [

        accuracy,

        precision,

        recall,

        f1,

        specificity,

        balanced_accuracy,

        roc_auc,

        average_precision,

        mcc,

        negative_predictive_value,

        false_positive_rate,

        false_negative_rate
    ]

})


metrics_data["Percentage"] = (
    metrics_data["Value"] * 100
).round(2)


print(
    "\n================ MODEL EVALUATION METRICS ================\n"
)

display(
    metrics_data
)


# ============================================================
# CELL 31 - PRINT IMPORTANT METRICS SEPARATELY
# ============================================================

print("\n================ IMPORTANT RESULTS ================\n")

print(
    f"Accuracy       : {accuracy:.4f} ({accuracy * 100:.2f}%)"
)

print(
    f"Precision      : {precision:.4f} ({precision * 100:.2f}%)"
)

print(
    f"Recall         : {recall:.4f} ({recall * 100:.2f}%)"
)

print(
    f"F1 Score       : {f1:.4f} ({f1 * 100:.2f}%)"
)

print(
    f"Specificity    : {specificity:.4f} ({specificity * 100:.2f}%)"
)

print(
    f"ROC-AUC Score  : {roc_auc:.4f}"
)

print(
    f"Average Precision : {average_precision:.4f}"
)

print(
    f"MCC            : {mcc:.4f}"
)


# ============================================================
# CELL 32 - CONFUSION MATRIX VALUES
# ============================================================

print(
    "\n================ CONFUSION MATRIX VALUES ================\n"
)

print("True Negative  (TN):", tn)

print("False Positive (FP):", fp)

print("False Negative (FN):", fn)

print("True Positive  (TP):", tp)


# ============================================================
# CELL 33 - CONFUSION MATRIX GRAPH
# ============================================================

plt.figure(
    figsize=(7, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "No Churn",
        "Churn"
    ],
    yticklabels=[
        "No Churn",
        "Churn"
    ]
)

plt.title(
    "Confusion Matrix",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel(
    "Predicted Value"
)

plt.ylabel(
    "Actual Value"
)

plt.show()


# ============================================================
# CELL 34 - CLASSIFICATION REPORT
# ============================================================

print(
    "\n================ CLASSIFICATION REPORT ================\n"
)

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# ============================================================
# CELL 35 - CLASSIFICATION REPORT AS TABLE
# ============================================================

classification_report_table = pd.DataFrame(
    classification_report(
        y_test,
        y_pred,
        output_dict=True
    )
).transpose()


print(
    "\n================ CLASSIFICATION REPORT TABLE ================\n"
)

display(
    classification_report_table.round(4)
)


# ============================================================
# CELL 36 - METRICS COMPARISON GRAPH
# ============================================================

important_metrics = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Specificity",
        "ROC-AUC"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        specificity,
        roc_auc
    ]
})


plt.figure(
    figsize=(10, 6)
)

ax = sns.barplot(
    data=important_metrics,
    x="Metric",
    y="Score"
)


plt.ylim(
    0,
    1
)

plt.title(
    "Model Evaluation Metrics",
    fontsize=15,
    fontweight="bold"
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Evaluation Metric"
)


for container in ax.containers:

    ax.bar_label(
        container,
        fmt="%.3f"
    )


plt.tight_layout()

plt.show()


# ============================================================
# CELL 37 - ROC CURVE
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test_binary,
    y_probability
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate / Recall"
)

plt.title(
    "ROC Curve",
    fontsize=15,
    fontweight="bold"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.show()


# ============================================================
# CELL 38 - PRECISION RECALL CURVE
# ============================================================

precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
    y_test_binary,
    y_probability
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall_curve,
    precision_curve,
    linewidth=2,
    label=f"Average Precision = {average_precision:.3f}"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve",
    fontsize=15,
    fontweight="bold"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.show()


# ============================================================
# CELL 39 - ACTUAL VS PREDICTED COUNTS
# ============================================================

actual_counts = y_test.value_counts()

predicted_counts = pd.Series(
    y_pred
).value_counts()


comparison = pd.DataFrame({

    "Actual": actual_counts,

    "Predicted": predicted_counts

}).fillna(0)


print(
    "\n================ ACTUAL VS PREDICTED ================\n"
)

display(
    comparison
)


comparison.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title(
    "Actual vs Predicted Customer Classes",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel(
    "Churn Status"
)

plt.ylabel(
    "Number of Customers"
)

plt.xticks(
    rotation=0
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# CELL 40 - CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


cv_accuracy_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


print(
    "\n================ 5-FOLD CROSS VALIDATION ================\n"
)


print(
    "Accuracy for each fold:"
)

print(
    np.round(
        cv_accuracy_scores,
        4
    )
)


print(
    "\nMean Cross Validation Accuracy:",
    round(
        cv_accuracy_scores.mean(),
        4
    )
)


print(
    "Mean Cross Validation Accuracy Percentage:",
    round(
        cv_accuracy_scores.mean() * 100,
        2
    ),
    "%"
)


print(
    "Cross Validation Standard Deviation:",
    round(
        cv_accuracy_scores.std(),
        4
    )
)


# ============================================================
# CELL 41 - CROSS VALIDATION GRAPH
# ============================================================

cv_results = pd.DataFrame({

    "Fold": [
        "Fold 1",
        "Fold 2",
        "Fold 3",
        "Fold 4",
        "Fold 5"
    ],

    "Accuracy": cv_accuracy_scores
})


plt.figure(
    figsize=(8, 5)
)

ax = sns.barplot(
    data=cv_results,
    x="Fold",
    y="Accuracy"
)


plt.ylim(
    0,
    1
)

plt.title(
    "5-Fold Cross Validation Accuracy",
    fontsize=14,
    fontweight="bold"
)


for container in ax.containers:

    ax.bar_label(
        container,
        fmt="%.3f"
    )


plt.ylabel(
    "Accuracy"
)

plt.show()


# ============================================================
# CELL 42 - LOGISTIC REGRESSION FEATURE IMPORTANCE
# ============================================================

feature_names = (
    pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


coefficients = (
    pipeline
    .named_steps["classifier"]
    .coef_[0]
)


feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Coefficient": coefficients

})


feature_importance["Feature"] = (
    feature_importance["Feature"]
    .str.replace(
        "categorical__",
        "",
        regex=False
    )
    .str.replace(
        "numerical__",
        "",
        regex=False
    )
)


feature_importance = feature_importance.sort_values(
    "Coefficient",
    ascending=False
)


print(
    "\n================ FEATURES INCREASING CHURN ================\n"
)

display(
    feature_importance.head(10)
)


print(
    "\n================ FEATURES REDUCING CHURN ================\n"
)

display(
    feature_importance.tail(10)
)


# ============================================================
# CELL 43 - FEATURE COEFFICIENT GRAPH
# ============================================================

top_positive = feature_importance.nlargest(
    10,
    "Coefficient"
).copy()


top_negative = feature_importance.nsmallest(
    10,
    "Coefficient"
).copy()


top_features = pd.concat(
    [
        top_negative,
        top_positive
    ]
)


top_features["Effect"] = np.where(
    top_features["Coefficient"] > 0,
    "Increases Churn",
    "Reduces Churn"
)


plt.figure(
    figsize=(11, 8)
)


sns.barplot(
    data=top_features,
    x="Coefficient",
    y="Feature",
    hue="Effect",
    dodge=False
)


plt.axvline(
    x=0,
    linestyle="--"
)


plt.title(
    "Important Features Affecting Customer Churn",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel(
    "Logistic Regression Coefficient"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.show()


# ============================================================
# CELL 44 - CREATE PREDICTION RESULT TABLE
# ============================================================

prediction_results = X_test.copy()


prediction_results["Actual_Churn"] = y_test.values


prediction_results["Predicted_Churn"] = y_pred


prediction_results["Churn_Probability"] = (
    y_probability * 100
)


def assign_risk(probability):

    if probability < 30:

        return "LOW"

    elif probability < 70:

        return "MEDIUM"

    else:

        return "HIGH"


prediction_results["Risk_Level"] = (
    prediction_results[
        "Churn_Probability"
    ]
    .apply(
        assign_risk
    )
)


print(
    "\n================ SAMPLE PREDICTION RESULTS ================\n"
)


display(
    prediction_results[
        [
            "tenure",
            "Contract",
            "InternetService",
            "PaymentMethod",
            "MonthlyCharges",
            "Actual_Churn",
            "Predicted_Churn",
            "Churn_Probability",
            "Risk_Level"
        ]
    ]
    .head(20)
)


# ============================================================
# CELL 45 - RISK LEVEL DISTRIBUTION
# ============================================================

risk_distribution = (
    prediction_results[
        "Risk_Level"
    ]
    .value_counts()
    .reindex(
        [
            "LOW",
            "MEDIUM",
            "HIGH"
        ]
    )
    .fillna(0)
)


print(
    "\n================ CHURN RISK DISTRIBUTION ================\n"
)


display(
    risk_distribution.to_frame(
        "Number of Customers"
    )
)


plt.figure(
    figsize=(8, 5)
)


ax = sns.barplot(
    x=risk_distribution.index,
    y=risk_distribution.values
)


plt.title(
    "Predicted Customer Churn Risk Levels",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel(
    "Risk Level"
)

plt.ylabel(
    "Number of Customers"
)


for container in ax.containers:

    ax.bar_label(
        container
    )


plt.show()


# ============================================================
# CELL 46 - SAVE TRAINED PIPELINE
# ============================================================

model_path = (
    "/content/drive/MyDrive/ML/churn_pipeline.pkl"
)


joblib.dump(
    pipeline,
    model_path
)


print(
    "churn_pipeline.pkl saved successfully!"
)

print(
    "Saved location:",
    model_path
)


# ============================================================
# CELL 47 - SAVE MODEL EVALUATION RESULTS
# ============================================================

metrics_path = (
    "/content/drive/MyDrive/ML/churn_model_metrics.csv"
)


classification_path = (
    "/content/drive/MyDrive/ML/classification_report.csv"
)


prediction_path = (
    "/content/drive/MyDrive/ML/churn_test_predictions.csv"
)


metrics_data.to_csv(
    metrics_path,
    index=False
)


classification_report_table.to_csv(
    classification_path
)


prediction_results.to_csv(
    prediction_path,
    index=False
)


print(
    "\nEvaluation files saved successfully!"
)

print(
    "Metrics:",
    metrics_path
)

print(
    "Classification Report:",
    classification_path
)

print(
    "Predictions:",
    prediction_path
)


# ============================================================
# CELL 48 - FUNCTION TO PREDICT A NEW CUSTOMER
# ============================================================

def predict_customer_churn(customer_data):

    customer_df = pd.DataFrame(
        [customer_data]
    )


    predicted_class = pipeline.predict(
        customer_df
    )[0]


    classes = list(
        pipeline.classes_
    )


    churn_index = classes.index(
        "Yes"
    )


    churn_probability = (
        pipeline.predict_proba(
            customer_df
        )[0][churn_index]
        * 100
    )


    if churn_probability < 30:

        risk_level = "LOW"


    elif churn_probability < 70:

        risk_level = "MEDIUM"


    else:

        risk_level = "HIGH"


    result = pd.DataFrame({

        "Predicted Churn": [
            predicted_class
        ],

        "Churn Probability": [
            round(
                churn_probability,
                2
            )
        ],

        "Risk Level": [
            risk_level
        ]
    })


    return result


# ============================================================
# CELL 49 - SAMPLE CUSTOMER PREDICTION
# ============================================================

sample_customer = {

    "gender": "Male",

    "SeniorCitizen": 0,

    "Partner": "No",

    "Dependents": "No",

    "tenure": 3,

    "PhoneService": "Yes",

    "MultipleLines": "No",

    "InternetService": "Fiber optic",

    "OnlineSecurity": "No",

    "OnlineBackup": "No",

    "DeviceProtection": "No",

    "TechSupport": "No",

    "StreamingTV": "Yes",

    "StreamingMovies": "Yes",

    "Contract": "Month-to-month",

    "PaperlessBilling": "Yes",

    "PaymentMethod": "Electronic check",

    "MonthlyCharges": 95.00,

    "TotalCharges": 285.00
}


sample_prediction = predict_customer_churn(
    sample_customer
)


print(
    "\n================ NEW CUSTOMER CHURN PREDICTION ================\n"
)


display(
    sample_prediction
)


# ============================================================
# CELL 50 - FINAL PROJECT SUMMARY
# ============================================================

print(
    "\n======================================================"
)

print(
    "CUSTOMER CHURN MACHINE LEARNING PROJECT - FINAL RESULTS"
)

print(
    "======================================================"
)


print(
    "\nMachine Learning Type : Supervised Learning"
)


print(
    "Problem Type          : Binary Classification"
)


print(
    "Algorithm             : Logistic Regression"
)


print(
    "Training Records      :",
    len(X_train)
)


print(
    "Testing Records       :",
    len(X_test)
)


print(
    f"Accuracy              : {accuracy * 100:.2f}%"
)


print(
    f"Precision             : {precision * 100:.2f}%"
)


print(
    f"Recall                : {recall * 100:.2f}%"
)


print(
    f"F1 Score              : {f1 * 100:.2f}%"
)


print(
    f"Specificity           : {specificity * 100:.2f}%"
)


print(
    f"ROC-AUC Score         : {roc_auc:.4f}"
)


print(
    f"Cross Validation Acc. : {cv_accuracy_scores.mean() * 100:.2f}%"
)


print(
    "\nModel successfully trained, evaluated and saved."
)


print(
    "======================================================"
)

# ============================================================
# CELL 51 - CLASS IMBALANCE CHECK
# ============================================================

class_counts = df["Churn"].value_counts()

no_churn_count = class_counts.get("No", 0)
yes_churn_count = class_counts.get("Yes", 0)

total_count = len(df)

no_percentage = (
    no_churn_count / total_count
) * 100

yes_percentage = (
    yes_churn_count / total_count
) * 100


print(
    "\n================ CLASS IMBALANCE CHECK ================\n"
)

print(
    "No Churn Customers  :",
    no_churn_count,
    f"({no_percentage:.2f}%)"
)

print(
    "Churn Customers     :",
    yes_churn_count,
    f"({yes_percentage:.2f}%)"
)


if yes_churn_count > 0:

    imbalance_ratio = (
        no_churn_count /
        yes_churn_count
    )

    print(
        f"\nClass Ratio (No : Yes) = {imbalance_ratio:.2f} : 1"
    )


if abs(
    no_percentage - yes_percentage
) > 10:

    print(
        "\nObservation:"
    )

    print(
        "The dataset is imbalanced because the number of "
        "non-churn customers is considerably higher than "
        "the number of churn customers."
    )

    print(
        "Therefore, evaluation metrics such as Precision, "
        "Recall, F1 Score and ROC-AUC are important in "
        "addition to Accuracy."
    )

else:

    print(
        "\nObservation: The dataset is relatively balanced."
    )


imbalance_table = pd.DataFrame({

    "Class": [
        "No Churn",
        "Churn"
    ],

    "Customer Count": [
        no_churn_count,
        yes_churn_count
    ],

    "Percentage": [
        no_percentage,
        yes_percentage
    ]
})


print(
    "\nClass Distribution Table:"
)


display(
    imbalance_table.round(2)
)


plt.figure(
    figsize=(7, 5)
)

ax = sns.barplot(
    data=imbalance_table,
    x="Class",
    y="Customer Count"
)


for container in ax.containers:

    ax.bar_label(
        container
    )


plt.title(
    "Class Distribution - Churn vs No Churn",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel(
    "Customer Class"
)

plt.ylabel(
    "Number of Customers"
)

plt.show()


# ============================================================
# CELL 52 - TOP CHURN RISK AND PROTECTIVE FACTORS
# ============================================================

risk_factor_table = (
    feature_importance
    .sort_values(
        "Coefficient",
        ascending=False
    )
    .head(10)
    .copy()
)


protective_factor_table = (
    feature_importance
    .sort_values(
        "Coefficient",
        ascending=True
    )
    .head(10)
    .copy()
)


risk_factor_table[
    "Effect"
] = "Increases Churn Risk"


protective_factor_table[
    "Effect"
] = "Reduces Churn Risk"


risk_factor_table[
    "Odds Ratio"
] = np.exp(
    risk_factor_table[
        "Coefficient"
    ]
)


protective_factor_table[
    "Odds Ratio"
] = np.exp(
    protective_factor_table[
        "Coefficient"
    ]
)


print(
    "\n================ TOP 10 CHURN RISK FACTORS ================\n"
)


display(
    risk_factor_table[
        [
            "Feature",
            "Coefficient",
            "Odds Ratio",
            "Effect"
        ]
    ]
    .reset_index(
        drop=True
    )
    .round(4)
)


print(
    "\n================ TOP 10 PROTECTIVE FACTORS ================\n"
)


display(
    protective_factor_table[
        [
            "Feature",
            "Coefficient",
            "Odds Ratio",
            "Effect"
        ]
    ]
    .reset_index(
        drop=True
    )
    .round(4)
)


print(
    "\nInterpretation:"
)

print(
    "A positive Logistic Regression coefficient is associated "
    "with increased churn tendency."
)

print(
    "A negative coefficient is associated with reduced churn tendency."
)

print(
    "An Odds Ratio above 1 indicates higher churn odds, while an "
    "Odds Ratio below 1 indicates lower churn odds."
)


# ============================================================
# CELL 53 - INDIVIDUAL CUSTOMER RISK FACTOR EXPLANATION
# ============================================================

def explain_customer_risk(
    customer_data,
    top_n=10
):

    customer_df = pd.DataFrame(
        [customer_data]
    )


    transformed_customer = (
        pipeline
        .named_steps[
            "preprocessor"
        ]
        .transform(
            customer_df
        )
    )


    if hasattr(
        transformed_customer,
        "toarray"
    ):

        transformed_customer = (
            transformed_customer
            .toarray()
        )


    transformed_customer = (
        np.asarray(
            transformed_customer
        )[0]
    )


    customer_feature_names = (
        pipeline
        .named_steps[
            "preprocessor"
        ]
        .get_feature_names_out()
    )


    customer_coefficients = (
        pipeline
        .named_steps[
            "classifier"
        ]
        .coef_[0]
    )


    contributions = (
        transformed_customer
        * customer_coefficients
    )


    explanation = pd.DataFrame({

        "Feature": customer_feature_names,

        "Transformed Value": transformed_customer,

        "Coefficient": customer_coefficients,

        "Contribution": contributions
    })


    explanation[
        "Feature"
    ] = (
        explanation[
            "Feature"
        ]
        .str.replace(
            "categorical__",
            "",
            regex=False
        )
        .str.replace(
            "numerical__",
            "",
            regex=False
        )
    )


    explanation = explanation[
        explanation[
            "Transformed Value"
        ] != 0
    ].copy()


    explanation[
        "Effect"
    ] = np.where(

        explanation[
            "Contribution"
        ] > 0,

        "Increases Churn Risk",

        "Reduces Churn Risk"
    )


    increasing_risk = (
        explanation[
            explanation[
                "Contribution"
            ] > 0
        ]
        .sort_values(
            "Contribution",
            ascending=False
        )
        .head(
            top_n
        )
    )


    reducing_risk = (
        explanation[
            explanation[
                "Contribution"
            ] < 0
        ]
        .sort_values(
            "Contribution",
            ascending=True
        )
        .head(
            top_n
        )
    )


    classes = list(
        pipeline.classes_
    )


    churn_index = classes.index(
        "Yes"
    )


    churn_probability = (
        pipeline
        .predict_proba(
            customer_df
        )[0][
            churn_index
        ]
        * 100
    )


    predicted_class = (
        pipeline.predict(
            customer_df
        )[0]
    )


    if churn_probability < 30:

        risk_level = "LOW"

    elif churn_probability < 70:

        risk_level = "MEDIUM"

    else:

        risk_level = "HIGH"


    print(
        "\n======================================================"
    )

    print(
        "INDIVIDUAL CUSTOMER CHURN EXPLANATION"
    )

    print(
        "======================================================"
    )


    print(
        "\nPredicted Churn :",
        predicted_class
    )


    print(
        f"Churn Probability : {churn_probability:.2f}%"
    )


    print(
        "Risk Level :",
        risk_level
    )


    print(
        "\n================ FACTORS INCREASING RISK ================\n"
    )


    if len(
        increasing_risk
    ) > 0:

        display(

            increasing_risk[
                [
                    "Feature",
                    "Coefficient",
                    "Contribution",
                    "Effect"
                ]
            ]
            .reset_index(
                drop=True
            )
            .round(4)
        )

    else:

        print(
            "No positive feature contributions identified for this customer."
        )


    print(
        "\n================ FACTORS REDUCING RISK ================\n"
    )


    if len(
        reducing_risk
    ) > 0:

        display(

            reducing_risk[
                [
                    "Feature",
                    "Coefficient",
                    "Contribution",
                    "Effect"
                ]
            ]
            .reset_index(
                drop=True
            )
            .round(4)
        )

    else:

        print(
            "No negative feature contributions identified for this customer."
        )


    print(
        "\nNote: Contribution values explain how this customer's encoded "
        "features influence the Logistic Regression score."
    )


    return (
        increasing_risk,
        reducing_risk
    )


customer_risk_factors, customer_protective_factors = (
    explain_customer_risk(
        sample_customer
    )
)


# ============================================================
# CELL 54 - CHURN RISK THRESHOLD EXPLANATION
# ============================================================

risk_threshold_table = pd.DataFrame({

    "Churn Probability": [
        "0% - below 30%",
        "30% - below 70%",
        "70% - 100%"
    ],

    "Risk Level": [
        "LOW",
        "MEDIUM",
        "HIGH"
    ],

    "Interpretation": [

        "Customer has relatively low predicted churn probability",

        "Customer should be monitored and may require retention attention",

        "Customer has high predicted churn probability and requires priority retention action"
    ]
})


print(
    "\n================ CHURN RISK THRESHOLDS ================\n"
)


display(
    risk_threshold_table
)


print(
    "\nImportant Note:"
)

print(
    "Logistic Regression produces the churn probability."
)

print(
    "The LOW, MEDIUM and HIGH categories are application-level "
    "business rules used to make the probability easier to interpret."
)

print(
    "The model itself does not automatically learn the 30% and 70% boundaries."
)

print(
    "These thresholds can be changed later according to business cost, "
    "retention capacity or validation requirements."
)


print(
    "\nRisk Classification Used:"
)

print(
    "Probability < 30%       -> LOW RISK"
)

print(
    "Probability 30% to <70% -> MEDIUM RISK"
)

print(
    "Probability >= 70%      -> HIGH RISK"
)


# ============================================================
# CELL 55 - FINAL MODEL CONCLUSION
# ============================================================

print(
    "\n=========================================================="
)

print(
    "FINAL MODEL CONCLUSION"
)

print(
    "=========================================================="
)


print(
    "\nModel Used:"
)

print(
    "Logistic Regression"
)


print(
    "\nMachine Learning Type:"
)

print(
    "Supervised Learning - Binary Classification"
)


print(
    "\nFinal Evaluation Results:"
)


print(
    f"Accuracy     : {accuracy * 100:.2f}%"
)

print(
    f"Precision    : {precision * 100:.2f}%"
)

print(
    f"Recall       : {recall * 100:.2f}%"
)

print(
    f"F1 Score     : {f1 * 100:.2f}%"
)

print(
    f"Specificity  : {specificity * 100:.2f}%"
)

print(
    f"ROC-AUC      : {roc_auc:.4f}"
)

print(
    f"Balanced Accuracy : {balanced_accuracy * 100:.2f}%"
)

print(
    f"Average Precision : {average_precision:.4f}"
)

print(
    f"Cross Validation Accuracy : {cv_accuracy_scores.mean() * 100:.2f}%"
)


print(
    "\nConclusion:"
)


if (
    accuracy >= 0.75
    and roc_auc >= 0.75
):

    print(
        "The Logistic Regression model demonstrates useful predictive "
        "performance for identifying customers who may be at risk of churn."
    )

else:

    print(
        "The model demonstrates predictive ability, but its performance "
        "could be improved through additional feature engineering, "
        "model tuning or alternative classification algorithms."
    )


print(
    "\nThe model should not be evaluated using accuracy alone."
)

print(
    "Precision, Recall, F1 Score, ROC-AUC and the Confusion Matrix "
    "are also considered because the churn dataset contains an unequal "
    "distribution of churn and non-churn customers."
)

print(
    "\nThe trained model provides churn probability, which is converted "
    "into LOW, MEDIUM and HIGH risk levels using application-defined thresholds."
)

print(
    "Logistic Regression coefficients are used to identify global factors "
    "associated with increasing or reducing churn risk."
)

print(
    "Individual feature contributions are also used to explain why a "
    "specific customer receives a particular churn-risk score."
)

print(
    "\nOverall, the model can be used as an analytical system to identify "
    "potentially high-risk customers and support customer-retention decisions."
)

print(
    "\n=========================================================="
)
