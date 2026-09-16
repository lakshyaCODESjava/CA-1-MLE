# CA-1: Machine Learning Engineering Case Studies

This repository contains implementations and evaluations for two applied machine learning case studies focusing on class imbalance, threshold tuning, and ensemble modeling.

---

## Case Study 1: Hospital Patient Readmission Prediction

### Overview
Predicts whether a patient will be readmitted to the hospital. Due to the high clinical cost of false negatives (discharging a patient who later needs urgent care), the objective prioritizes recall over standard accuracy.

### Key Highlights
- **Dataset:** Synthetic clinical cohort of 5,000 patient records with 15 features and ~15% readmission rate.
- **Preprocessing:** Standardized numerical features using `StandardScaler` to handle scale variance.
- **Model:** Logistic Regression with L2 regularization ($C=1.0$).
- **Cost-Sensitive Decision Threshold:** Lowered classification threshold from `0.5` to `0.25` to capture high-risk readmissions and minimize false negatives.
- **Evaluation:** Evaluated using ROC-AUC and threshold-specific precision/recall trade-offs.

---

## Case Study 2: Credit Card Fraud Detection

### Overview
Detects fraudulent financial transactions in an extreme class imbalance setting (~1% fraud rate). Focuses on synthetic oversampling without data leakage and feature attribution via gradient boosting.

### Key Highlights
- **Dataset:** 10,000 synthetic transaction records with 20 features (15 informative) and a 99:1 class imbalance.
- **Resampling:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) exclusively to the training set to prevent test-set data leakage.
- **Model:** `XGBClassifier` trained with 100 estimators, learning rate of `0.1`, and max depth of `5`.
- **Feature Attribution:** Extracted top 5 features driving decision boundaries using tree-based feature importance.
- **Evaluation:** Evaluated using ROC-AUC and Precision-Recall metrics.

---

## Requirements & Setup

### Dependencies
Ensure Python 3.8+ is installed, then install the required libraries:

```bash
pip install numpy pandas scikit-learn imbalanced-learn xgboost
