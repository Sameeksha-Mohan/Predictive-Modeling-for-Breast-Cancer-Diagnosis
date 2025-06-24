# 🧠 Breast Cancer Prediction using Supervised Learning | Predictive Analytics

This repository contains a predictive modeling assignment completed as part of the **Predictive Analytics (MSBA 6420)** course at the **Carlson School of Management**. The objective was to apply and compare different classification models to diagnose breast cancer using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

---

## 📦 Dataset

- **Source**: UCI Machine Learning Repository  
- **Link**: [WDBC Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
- **Features**: 30 numerical features related to cell nuclei from digitized images of breast mass  
- **Target**: Diagnosis result — Malignant (M) or Benign (B)

---

## 🎯 Project Goals

- Preprocess and prepare the dataset for modeling.
- Build and evaluate three classifiers:
  - **Decision Tree**
  - **Logistic Regression**
  - **K-Nearest Neighbors (KNN)**
- Perform **hyperparameter tuning** for each model to optimize predictive performance.
- Evaluate models using classification metrics and compare their effectiveness.

---

## ⚙️ Methodology

### 🔹 Data Preprocessing
- Diagnosis column encoded: `M → 1`, `B → 0`
- Selected top 10 relevant features
- Split into training and test sets (80/20)
- Standardized features for KNN and Logistic Regression

---

## 🧪 Models & Tuning

### 🌳 Decision Tree
- **Initial depth**: 5 → overfitting observed
- **Tuned parameters**:
  - `max_depth`: [3, 5, 10, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
  - `criterion`: ['gini', 'entropy']
- **Best params**: `{'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5}`

### 📊 Logistic Regression
- **Tuned parameters**:
  - `C`: [0.1, 1, 10, 100]
  - `penalty`: ['l1', 'l2']
  - `solver`: ['liblinear', 'lbfgs']
- **Best params**: `{'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'}`

### 🧭 K-Nearest Neighbors (KNN)
- Features normalized using `StandardScaler`
- **Tuned parameters**:
  - `n_neighbors`: [3, 5, 7, 9, 11]
  - `metric`: ['euclidean', 'manhattan']
- **Best params**: `k = 5`

---

## 📈 Results & Evaluation

| Metric                  | Decision Tree | KNN     | Logistic Regression |
|-------------------------|---------------|---------|----------------------|
| Training Accuracy       | 0.9802        | 0.9758  | 0.9758               |
| Test Accuracy           | 0.9649        | 0.9737  | **0.9912**           |
| Precision (Malignant)   | 0.95          | 0.98    | **1.00**             |
| Recall (Malignant)      | 0.95          | 0.95    | **0.98**             |
| F1 Score (Malignant)    | 0.95          | 0.96    | **0.99**             |

🔍 **Conclusion**:  
All models performed well, but **Logistic Regression** provided the most reliable and accurate results with a test accuracy of **99.12%** and perfect precision for malignant cases. It was the best balance of generalization and performance.

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn (optional for visualization)
- Jupyter Notebook
