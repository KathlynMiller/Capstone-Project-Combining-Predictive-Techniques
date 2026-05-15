**MEMORANDUM**

**TO:** BANA 622 Course Instructor
**FROM:** Kathlyn Miller-Francis
**DATE:** April 27, 2026
**RE:** Lending Club Risk Grade Prediction — Capstone Project Summary

---

**Purpose**

This memo summarizes the approach, findings, and recommendation from a machine learning project developed to assist Lending Club in identifying high-risk loan applicants. The goal was to build a second-opinion model that predicts, using only application-time borrower characteristics, whether Lending Club's internal process will assign a higher-risk grade (D–G) rather than a prime grade (A–C).

**Approach**

The analysis used a 10,000-row sample of Lending Club loan records containing 55 applicant and loan attributes. After removing 12 post-origination leakage columns (including grade, interest rate, and payment outcomes), the working feature set was cleaned in three passes: joint-application fields were filled from individual equivalents; event-based missingness indicators were added for delinquency history columns; and remaining gaps were filled with medians or zeros. Three engineered features were added — credit utilization ratio, loan-to-income ratio, and credit age in years — to capture risk dimensions not directly represented in the raw data. The cleaned dataset was split 80/20 (stratified) before any preprocessing was fit, and four models were trained inside identical sklearn pipelines: Logistic Regression, Decision Tree, Neural Network (MLP), and Random Forest.

**Results**

The target class was moderately imbalanced (81.5% prime, 18.5% risky), making ROC-AUC and F1 score more informative than accuracy alone. Performance on the held-out test set is summarized below:

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|---------------------|----------|-----------|--------|-------|---------|
| Logistic Regression | 0.826    | 0.564     | 0.249  | 0.345 | 0.781   |
| Decision Tree       | 0.814    | 0.491     | 0.216  | 0.300 | 0.722   |
| Neural Network      | 0.783    | 0.408     | 0.389  | 0.398 | 0.723   |
| **Random Forest**   | **0.818**| **0.667** | 0.032  | 0.062 | **0.787** |

**Recommendation**

The Random Forest model is recommended. It achieves the highest ROC-AUC (0.787) and precision (0.667), meaning that when it flags an applicant as risky it is correct two-thirds of the time — the most actionable signal for a lending team conducting secondary reviews. The model's top predictors are loan term, credit utilization ratio, total debit limit, and debt-to-income ratio, all of which have clear underwriting interpretations. At the default threshold of 0.50, recall is low; lowering the threshold to 0.30–0.35 would materially improve the model's ability to catch risky borrowers at an acceptable cost in false positives, and is the recommended next step before production deployment.
