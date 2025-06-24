# 🧬 Breast Cancer Classification using Supervised Learning

This project applies a range of supervised learning techniques to predict whether a breast cancer case is malignant or benign based on digitized image features. The analysis emphasizes **recall** as the primary evaluation metric due to the high cost of false negatives in a medical diagnosis context.

This project was completed as part of the **Predictive Analytics (MSBA 6420)** course at the **Carlson School of Management**, University of Minnesota.

---

## 📦 Dataset

- **Source**: UCI Machine Learning Repository  
- **Name**: Wisconsin Diagnostic Breast Cancer (WDBC)  
- **Records**: 569 samples  
- **Features**: 30 real-valued measurements (radius, texture, area, smoothness, etc.)  
- **Target**: Diagnosis (Malignant = 1, Benign = 0)

[🔗 Dataset Link](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

---

## 🎯 Objective

- Build and evaluate four classification models:
  - **Support Vector Machine (SVM)**
  - **K-Nearest Neighbors (KNN)**
  - **Logistic Regression**
  - **Decision Tree**
- Focus on maximizing **recall** to minimize the risk of missing malignant cases.
- Use **nested cross-validation** to ensure robust performance estimation and prevent overfitting during hyperparameter tuning.

---

## 🧠 Methodology

- Selected 10 relevant features based on domain knowledge.
- Standardized all features using `StandardScaler`.
- Applied **nested cross-validation**:
  - **Inner loop**: hyperparameter tuning with `RandomizedSearchCV`
  - **Outer loop**: evaluation using 5-fold `StratifiedKFold`
- Evaluated models based on:
  - **Recall (primary metric)**
  - Accuracy, Precision, F1-Score
  - ROC Curve (AUC)
  - Lift Curve

---

## 🧪 Results Summary

| Model              | Mean Recall | Std Dev | AUC   |
|--------------------|-------------|---------|-------|
| **SVM**            | **0.9105**  | 0.0521  | **1.00** |
| K-Nearest Neighbors| 0.9152      | 0.0566  | 0.99  |
| Logistic Regression| 0.9055      | 0.0543  | 0.99  |
| Decision Tree      | 0.8732      | 0.0513  | 0.98  |

---

## 🏆 Best Model

**Support Vector Machine (SVM)** demonstrated the most balanced and reliable performance:
- **Perfect AUC (1.00)** indicating flawless separation of malignant vs. benign cases.
- Strong recall and high consistency across folds.
- Lift curves showed SVM effectively prioritizes malignant cases.
