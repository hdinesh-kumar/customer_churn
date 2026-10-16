# Customer Churn Intelligence System

## Project Overview

> **Note:** Keep the `images` folder in the same repository as this `README.md`.  
> The relative Markdown image paths used in this file will then render automatically on GitHub.


This project is a **Customer Churn Prediction and Analysis System** built using **Supervised Machine Learning**.

The goal of the project is to analyze telecom customer behaviour and predict whether a customer is likely to **churn** or **stay** with the company.

The project uses the **IBM Telco Customer Churn dataset** and applies **Logistic Regression** for binary classification.

The complete workflow includes:

- Dataset loading and inspection
- Data cleaning
- Exploratory Data Analysis (EDA)
- Class imbalance analysis
- Feature preprocessing
- Train-test splitting
- Logistic Regression model training
- Model evaluation
- Confusion matrix
- Classification report
- Precision, Recall and F1-score
- ROC-AUC analysis
- Precision-Recall analysis
- 5-fold cross-validation
- Global churn risk-factor analysis
- Individual customer churn explanation
- Churn probability calculation
- Low / Medium / High churn-risk classification

---

## Machine Learning Problem

This project uses:

- **Machine Learning Type:** Supervised Learning
- **Problem Type:** Binary Classification
- **Algorithm:** Logistic Regression
- **Target Variable:** `Churn`

The target has two possible values:

- `No` → Customer stayed
- `Yes` → Customer churned

The model learns from existing customer records and predicts the churn status of unseen customers.

---

## Dataset

The dataset contains:

| Item | Value |
|---|---:|
| Total Customers | 7,043 |
| Total Columns | 21 |
| Input Features Used | 19 |
| Target Column | Churn |
| Customer ID Column | Excluded from training |

### Important Features

Some of the important customer attributes include:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

---

## Data Cleaning

The `TotalCharges` column was initially stored as an object/text column.

It was converted into numeric format.

During conversion, **11 missing values** were detected.

These missing values were filled using the median value of `TotalCharges`.

- Missing `TotalCharges` values: **11**
- Median used: **1397.47**
- Duplicate rows: **0**

After cleaning, the dataset contained no missing values.

---

## Exploratory Data Analysis

### 1. Customer Churn Distribution

![Customer Churn Distribution](images/customer_churn_distribution.png)

![Class Distribution - Churn vs No Churn](images/class_distribution_churn_vs_no_churn.png)


The dataset contains:

| Churn Status | Customers | Percentage |
|---|---:|---:|
| No | 5,174 | 73.46% |
| Yes | 1,869 | 26.54% |

The dataset is therefore **imbalanced**.

The class ratio is approximately:

```text
No Churn : Churn = 2.77 : 1
```

This means there are substantially more customers who stayed than customers who churned.

Because of this class imbalance, the model should **not be judged using accuracy alone**. Precision, Recall, F1-score, ROC-AUC and the confusion matrix are also important.

---

## Contract Type vs Churn

![Contract Type vs Churn](images/contract_type_vs_churn.png)


The churn percentage differs strongly by contract type.

| Contract Type | No Churn | Churn |
|---|---:|---:|
| Month-to-month | 57.29% | 42.71% |
| One year | 88.73% | 11.27% |
| Two year | 97.17% | 2.83% |

### Interpretation

Customers with **month-to-month contracts** have the highest churn rate.

Customers with **two-year contracts** have the lowest churn rate.

This indicates that long-term contracts are strongly associated with customer retention.

---

## Internet Service vs Churn

![Internet Service vs Churn](images/internet_service_vs_churn.png)


| Internet Service | No Churn | Churn |
|---|---:|---:|
| DSL | 81.04% | 18.96% |
| Fiber optic | 58.11% | 41.89% |
| No internet | 92.60% | 7.40% |

### Interpretation

Customers using **Fiber optic** service show the highest churn percentage.

Customers with no internet service show the lowest churn percentage.

---

## Payment Method vs Churn

![Payment Method vs Churn](images/payment_method_vs_churn.png)


| Payment Method | No Churn | Churn |
|---|---:|---:|
| Bank transfer (automatic) | 83.29% | 16.71% |
| Credit card (automatic) | 84.76% | 15.24% |
| Electronic check | 54.71% | 45.29% |
| Mailed check | 80.89% | 19.11% |

### Interpretation

Customers paying using **Electronic check** show the highest churn percentage.

Automatic payment methods are associated with lower churn.

---

## Tenure Analysis

![Customer Tenure Distribution by Churn](images/tenure_distribution_by_churn.png)


The tenure distribution shows that churn is more common among customers with **shorter tenure**.

Customers who have stayed for longer periods are generally less likely to churn.

This is also supported later by the Logistic Regression coefficient for `tenure`, which is one of the strongest churn-reducing factors.

---

## Monthly Charges vs Churn

![Monthly Charges vs Churn](images/monthly_charges_vs_churn_1.png)

![Monthly Charges vs Churn - Additional Output](images/monthly_charges_vs_churn_2.png)


The box plot shows that customers who churn generally have **higher monthly charges** than customers who stay.

This suggests that pricing and monthly cost may influence churn behaviour.

---

## Total Charges vs Churn

![Total Charges vs Churn](images/total_charges_vs_churn.png)


Customers who churn generally have lower total charges because many of them have shorter tenure.

Customers who stay longer accumulate higher total charges over time.

---

## Correlation Between Numerical Features

![Correlation Between Numerical Features](images/correlation_heatmap.png)


The strongest numerical correlation observed was:

```text
tenure vs TotalCharges = 0.83
```

This is expected because customers who remain with the company longer naturally accumulate higher total charges.

Other important correlations include:

```text
MonthlyCharges vs TotalCharges = 0.65
tenure vs MonthlyCharges       = 0.25
SeniorCitizen vs MonthlyCharges = 0.22
```

---

# Model Development

## Features and Target

The following columns were removed from the input features:

- `customerID`
- `Churn`

The target variable was:

```python
y = df["Churn"]
```

The remaining 19 columns were used as model input features.

---

## Categorical and Numerical Features

### Categorical Features

The project automatically identified categorical features such as:

- gender
- Partner
- Dependents
- InternetService
- Contract
- PaymentMethod
- TechSupport
- OnlineSecurity
- StreamingTV
- StreamingMovies

Categorical variables were transformed using:

```text
OneHotEncoder
```

### Numerical Features

The numerical variables were:

- SeniorCitizen
- tenure
- MonthlyCharges
- TotalCharges

These numerical columns were standardized using:

```text
StandardScaler
```

---

## Train-Test Split

The dataset was split into:

| Dataset | Records |
|---|---:|
| Training Set | 5,634 |
| Testing Set | 1,409 |

An **80:20 train-test split** was used.

Stratification was applied so that the churn distribution remained similar in both training and testing datasets.

---

## Machine Learning Pipeline

A Scikit-learn Pipeline was used to combine:

```text
Categorical Encoding
        +
Numerical Scaling
        +
Logistic Regression
```

This ensures that the same preprocessing steps are automatically applied during training and prediction.

---

# Model Evaluation

## Overall Evaluation Metrics

![Model Evaluation Metrics](images/model_evaluation_metrics.png)


| Evaluation Metric | Result |
|---|---:|
| Accuracy | **80.55%** |
| Precision | **65.72%** |
| Recall / Sensitivity | **55.88%** |
| F1 Score | **60.40%** |
| Specificity | **89.47%** |
| Balanced Accuracy | **72.68%** |
| ROC-AUC Score | **0.8419** |
| Average Precision | **0.6334** |
| Matthews Correlation Coefficient | **0.4790** |
| Negative Predictive Value | **84.88%** |
| False Positive Rate | **10.53%** |
| False Negative Rate | **44.12%** |

---

## Accuracy

The model achieved:

```text
Accuracy = 80.55%
```

This means approximately 80.55% of the test-set customers were classified correctly.

However, because the dataset is imbalanced, accuracy alone does not fully describe model performance.

---

## Precision

```text
Precision = 65.72%
```

Precision answers:

> Out of all customers predicted as churners, how many actually churned?

A precision of 65.72% means that approximately 65.72% of customers predicted as churn actually belonged to the churn class.

---

## Recall

```text
Recall = 55.88%
```

Recall answers:

> Out of all customers who actually churned, how many did the model correctly identify?

The model correctly identified approximately 55.88% of actual churners.

---

## F1 Score

```text
F1 Score = 60.40%
```

The F1-score is the harmonic mean of Precision and Recall.

It is especially useful when the dataset contains class imbalance.

---

## Specificity

```text
Specificity = 89.47%
```

Specificity measures how well the model correctly identifies customers who **do not churn**.

The model performs strongly when identifying customers who stay.

---

# Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)


The confusion matrix produced:

|  | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| **Actual No Churn** | 926 | 109 |
| **Actual Churn** | 165 | 209 |

Therefore:

```text
True Negative  = 926
False Positive = 109
False Negative = 165
True Positive  = 209
```

### Interpretation

- **926 customers** were correctly identified as non-churners.
- **209 customers** were correctly identified as churners.
- **109 customers** were incorrectly predicted to churn.
- **165 customers** actually churned but were missed by the model.

The relatively high number of false negatives explains why Recall is lower than Specificity.

---

# Classification Report

The classification report produced:

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| No Churn | 0.8488 | 0.8947 | 0.8711 | 1,035 |
| Churn | 0.6572 | 0.5588 | 0.6040 | 374 |

The model performs better at identifying non-churning customers than churners.

This is partly related to the class imbalance in the original dataset.

---

# ROC Curve

![ROC Curve](images/roc_curve.png)


The model achieved:

```text
ROC-AUC = 0.8419
```

An ROC-AUC of approximately **0.84** indicates that the model has good ability to distinguish between churners and non-churners.

A random classifier would have an ROC-AUC close to 0.50.

Therefore, Logistic Regression performs substantially better than random classification.

---

# Precision-Recall Curve

![Precision-Recall Curve](images/precision_recall_curve.png)


The Average Precision score was:

```text
Average Precision = 0.6334
```

The Precision-Recall curve is particularly useful because the churn class is the minority class.

It shows the trade-off between:

- Precision
- Recall

when the classification threshold changes.

---

# 5-Fold Cross-Validation

![5-Fold Cross Validation Accuracy](images/five_fold_cross_validation.png)


The model was also evaluated using 5-fold stratified cross-validation.

| Fold | Accuracy |
|---|---:|
| Fold 1 | 0.8112 |
| Fold 2 | 0.8105 |
| Fold 3 | 0.8197 |
| Fold 4 | 0.7898 |
| Fold 5 | 0.7947 |

The mean cross-validation accuracy was:

```text
80.52%
```

The standard deviation was:

```text
0.0112
```

### Interpretation

The fold accuracies are relatively close to each other.

This indicates that the model performance is reasonably stable across different subsets of the dataset.

The cross-validation accuracy of 80.52% is also very close to the test accuracy of 80.55%, which indicates consistent performance.

---

# Churn Risk Factors

![Important Features Affecting Customer Churn](images/important_features_affecting_churn.png)


Because Logistic Regression is interpretable, its coefficients can be used to understand which features are associated with increased or reduced churn tendency.

## Top Churn-Increasing Factors

| Feature | Coefficient | Odds Ratio |
|---|---:|---:|
| InternetService_Fiber optic | 0.6402 | 1.8968 |
| Contract_Month-to-month | 0.5799 | 1.7858 |
| TotalCharges | 0.5163 | 1.6758 |
| StreamingMovies_Yes | 0.2041 | 1.2265 |
| StreamingTV_Yes | 0.2036 | 1.2258 |
| PaymentMethod_Electronic check | 0.1984 | 1.2194 |
| OnlineSecurity_No | 0.1568 | 1.1698 |
| TechSupport_No | 0.1321 | 1.1412 |
| MultipleLines_Yes | 0.1061 | 1.1119 |
| SeniorCitizen | 0.0541 | 1.0556 |

### Interpretation

The strongest categorical churn-risk indicators include:

- Fiber optic internet
- Month-to-month contract
- Electronic check payment
- No online security
- No technical support

A positive Logistic Regression coefficient indicates an association with increased churn tendency.

---

## Top Churn-Reducing Factors

| Feature | Coefficient | Odds Ratio |
|---|---:|---:|
| tenure | -1.2410 | 0.2891 |
| Contract_Two year | -0.7730 | 0.4616 |
| InternetService_DSL | -0.6526 | 0.5207 |
| MonthlyCharges | -0.5964 | 0.5508 |
| PaperlessBilling_No | -0.3431 | 0.7096 |
| InternetService_No | -0.3011 | 0.7400 |
| StreamingTV_No internet service | -0.3011 | 0.7400 |
| OnlineSecurity_No internet service | -0.3011 | 0.7400 |
| TechSupport_No internet service | -0.3011 | 0.7400 |
| DeviceProtection_No internet service | -0.3011 | 0.7400 |

### Interpretation

The strongest churn-reducing factor is **customer tenure**.

A **two-year contract** is also strongly associated with reduced churn.

A negative Logistic Regression coefficient indicates an association with lower churn tendency.

> Note: Coefficients show model associations and should not automatically be interpreted as proof of real-world causation.

---

# Individual Customer Churn Explanation

A sample customer was tested using the trained model.

The result was:

```text
Predicted Churn     : Yes
Churn Probability   : 76.29%
Risk Level          : HIGH
```

## Main Factors Increasing This Customer's Risk

The strongest contributors included:

- Low tenure
- Fiber optic internet
- Month-to-month contract
- Streaming movies enabled
- Streaming TV enabled
- Electronic check payment
- No online security
- No technical support

The model therefore identified the customer as having a relatively high probability of churn.

---

# Churn Risk Classification

The model itself produces a churn probability.

To make this probability easier to understand in the application, the project uses the following business rules:

| Churn Probability | Risk Level | Meaning |
|---|---|---|
| Below 30% | LOW | Relatively low churn probability |
| 30% to below 70% | MEDIUM | Customer should be monitored |
| 70% and above | HIGH | Priority retention action may be required |

For example:

```text
18% → LOW RISK
52% → MEDIUM RISK
76% → HIGH RISK
```

## Important Note

The **30% and 70% thresholds are not automatically learned by Logistic Regression**.

Logistic Regression produces the churn probability.

The Low, Medium and High categories are **application-level rules** used to make the prediction easier to interpret.

These thresholds can later be adjusted based on business requirements, retention costs or validation studies.

---

# Risk-Level Distribution

![Predicted Customer Churn Risk Levels](images/predicted_churn_risk_levels.png)


The test customers were divided into:

| Risk Level | Customers |
|---|---:|
| LOW | 866 |
| MEDIUM | 451 |
| HIGH | 92 |

Most test customers fall into the Low-risk category, while a smaller group is identified as High risk.

The High-risk group can be prioritized for retention actions.

---

# Actual vs Predicted Classes

![Actual vs Predicted Customer Classes](images/actual_vs_predicted_1.png)

![Actual vs Predicted Customer Classes - Additional Output](images/actual_vs_predicted_2.png)


For the test dataset:

| Churn Status | Actual | Predicted |
|---|---:|---:|
| No | 1,035 | 1,091 |
| Yes | 374 | 318 |

The model predicts slightly more customers as non-churners and fewer customers as churners than actually occur in the test set.

This matches the confusion-matrix observation that some churn customers are missed.

---

# Key Business Insights

The analysis suggests the following customer groups deserve particular attention:

1. **Month-to-month customers** have much higher churn rates than customers with longer contracts.
2. **Fiber optic users** show substantially higher churn rates than DSL users.
3. **Electronic check users** show the highest churn percentage among payment methods.
4. **Low-tenure customers** are more vulnerable to churn.
5. Customers without **Online Security** or **Tech Support** show increased churn tendency.
6. Longer tenure and two-year contracts are strongly associated with customer retention.

---

# Final Model Conclusion

The Logistic Regression model achieved:

```text
Accuracy                  : 80.55%
Precision                 : 65.72%
Recall                    : 55.88%
F1 Score                  : 60.40%
Specificity               : 89.47%
ROC-AUC                   : 0.8419
Balanced Accuracy         : 72.68%
Average Precision         : 0.6334
Cross-Validation Accuracy : 80.52%
```

The model demonstrates **useful predictive performance** for customer churn analysis.

Its major strengths are:

- Good overall accuracy
- Strong specificity
- Good ROC-AUC
- Stable cross-validation performance
- Interpretability through Logistic Regression coefficients
- Ability to calculate churn probabilities
- Ability to identify global and individual churn-risk factors

The main limitation is the lower Recall for churn customers.

This means the model still misses some customers who actually churn.

Future improvements could include:

- Class-weight adjustment
- SMOTE or other imbalance-handling techniques
- Classification-threshold tuning
- Hyperparameter tuning
- Comparing Logistic Regression with Random Forest, XGBoost or other classifiers
- Additional feature engineering

---

# Project Output Files

The notebook generates and saves:

```text
churn_pipeline.pkl
churn_model_metrics.csv
classification_report.csv
churn_test_predictions.csv
```

### `churn_pipeline.pkl`

Contains the complete trained preprocessing and Logistic Regression pipeline.

### `churn_model_metrics.csv`

Contains the model evaluation metrics.

### `classification_report.csv`

Contains precision, recall, F1-score and support for each class.

### `churn_test_predictions.csv`

Contains test-set customer details, actual churn, predicted churn, churn probability and risk level.

---

# How to Run the Project

The project is designed to run in **Google Colab**.

## 1. Upload the Dataset

The expected dataset is:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The notebook currently expects it at:

```text
/content/drive/MyDrive/ML/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

If the file is not found, the notebook allows manual upload.

## 2. Run the Notebook Cells in Order

Run the notebook from the first cell to the final cell.

The notebook will:

```text
Load Dataset
      ↓
Clean Data
      ↓
Perform EDA
      ↓
Preprocess Features
      ↓
Train Logistic Regression
      ↓
Evaluate Model
      ↓
Generate Confusion Matrix
      ↓
Generate Classification Report
      ↓
Calculate Accuracy / Precision / Recall / F1
      ↓
Generate ROC and Precision-Recall Curves
      ↓
Perform Cross-Validation
      ↓
Identify Churn Risk Factors
      ↓
Predict Churn Probability
      ↓
Assign LOW / MEDIUM / HIGH Risk
      ↓
Save Model and Evaluation Results
```

---

# Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

# Project Summary

This project demonstrates a complete **end-to-end machine learning classification workflow** for customer churn prediction.

It does more than simply predict `Yes` or `No`.

The system also provides:

- Churn probability
- Risk level
- Model evaluation metrics
- Confusion matrix
- Classification report
- Cross-validation
- ROC analysis
- Precision-Recall analysis
- Global churn-risk factors
- Individual customer-risk explanations

This makes the project useful not only as a predictive model but also as an interpretable customer-retention analysis system.
