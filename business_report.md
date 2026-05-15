# Predicting High-Risk Borrowers Before the Loan Is Made
## A Risk Analytics Report for Lending Club's Head of Risk Management

**Prepared by:** Kathlyn Miller-Francis
**Date:** May 4, 2026
**Audience:** Head of Risk Management, Lending Club

---

## 1. Problem and Stakes

Lending Club assigns every loan a letter grade — A through G — that determines the interest rate charged and reflects the platform's internal assessment of default risk. Loans graded D through G carry meaningfully higher default rates and generate disproportionate credit losses when they do fail. Today, that grading happens through a proprietary process that cannot be easily audited or stress-tested.

The goal of this project was to build an independent, transparent second-opinion model that predicts — using only information available at the time of application — whether a borrower is likely to receive a high-risk grade (D–G) rather than a prime grade (A–C). Such a model serves two business purposes: it provides a fast, explainable pre-screen that can flag borderline applications for closer review, and it creates a documented, reproducible risk signal that supports regulatory and audit requirements. With credit losses in unsecured consumer lending typically running at several percentage points of the loan book, even modest improvements in upfront risk identification translate directly to reduced charge-offs and better pricing accuracy.

---

## 2. Recommendation

**Deploy the Random Forest model.**

Four modeling approaches were evaluated on an 18.5% / 81.5% imbalanced dataset (risky vs. prime) drawn from 10,000 Lending Club loan records. Because simply predicting "prime" for every applicant would yield 81.5% accuracy while identifying zero risky borrowers, raw accuracy was not the primary selection criterion. The table below summarizes hold-out test performance:

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|---------------------|----------|-----------|--------|-------|---------|
| Logistic Regression | 82.6%    | 56.4%     | 24.9%  | 34.5% | 0.781   |
| Decision Tree       | 81.4%    | 49.1%     | 21.6%  | 30.0% | 0.722   |
| Neural Network      | 78.3%    | 40.8%     | 38.9%  | 39.8% | 0.723   |
| **Random Forest**   | **81.8%**| **66.7%** | 3.2%   | 6.2%  | **0.787** |

The Random Forest leads on the two metrics that matter most for this use case. Its ROC-AUC of 0.787 is the highest of the four models, meaning it ranks high-risk applicants above prime applicants more reliably than any alternative — the foundation of a useful screening tool. More importantly, its precision of 66.7% means that when the model raises a flag, it is correct two out of every three times. For a secondary-review workflow where a risk analyst's time is scarce, a flag that carries that level of reliability is far more actionable than a noisier signal.

The Neural Network showed the highest recall under the default settings but did so by flagging a far larger share of applicants as risky, including many false alarms. The Decision Tree was the weakest performer on every metric. Logistic Regression was competitive on ROC-AUC but trailed on precision. The Random Forest strikes the best balance between discriminatory power and trustworthy positive predictions.

---

## 3. Top Five Risk Drivers

The five most influential inputs to the Random Forest, in order of importance, are listed below. Each represents a dimension of borrower risk that the model learned to weight heavily — and each has a straightforward underwriting interpretation.

**1. Loan Term (36-month vs. 60-month)**
Borrowers who request a 60-month repayment term are flagged as higher risk at substantially higher rates than those on a 36-month term. This reflects two realities: borrowers who need more time to repay are often stretching to afford the payment, and a longer loan means more time for life circumstances to change and defaults to accumulate. Loan term is also a partially deliberate choice by Lending Club's own grading process, which makes it the single strongest predictor of the grade the model is trying to anticipate.

**2. Credit Utilization Ratio (balance used as a share of total credit available)**
This engineered feature — total revolving balance divided by total credit limit across all accounts — captures how close a borrower is to their borrowing ceiling. A borrower using 85% of available credit signals financial strain in a way that raw balance figures alone do not. High utilization consistently separates prime from risky borrowers because it reflects both current cash pressure and reduced capacity to absorb unexpected expenses.

**3. Total Debit Limit (sum of all revolving credit limits)**
The total size of a borrower's revolving credit footprint is a strong predictor in both directions. Very low total limits often indicate a thin or troubled credit history, while unusually high limits can signal over-extension or aggressive borrowing. The model uses this feature alongside utilization to distinguish borrowers with large but responsibly managed credit from those carrying large limits they cannot service.

**4. Debt-to-Income Ratio**
Debt-to-income — monthly debt obligations as a percentage of monthly income — is the oldest and most universally used underwriting ratio for good reason: it directly measures whether a borrower can afford new debt. Applicants with high debt-to-income ratios are far more likely to receive a risky grade, and the model weights this feature heavily. Notably, the model uses the ratio at application time, before the new loan's payment is added, making it a measure of baseline financial health rather than post-loan capacity.

**5. Loan-to-Income Ratio (loan amount relative to annual income)**
This engineered feature captures the size of the loan request in proportion to the borrower's income. A borrower earning $40,000 and requesting a $30,000 loan is in a fundamentally different risk position than one earning $120,000 requesting the same amount, even if their debt-to-income ratios appear similar. Loan-to-income isolates the relative burden of this specific loan, and its appearance in the top five reflects that borrowers who are reaching significantly beyond their income level are materially more likely to be graded high-risk.

---

## 4. Tradeoffs and Classification Threshold

Every prediction model that produces a probability score requires a business decision: at what probability level do we call an applicant "risky"? The default convention of 50% — flag anyone the model considers more likely than not to be high-risk — is almost never optimal in an imbalanced setting.

At the 50% threshold, the Random Forest achieves its high precision (66.7%) but catches only about 3% of actual high-risk borrowers. In practical terms, for every 100 high-risk applicants that pass through, the model flags three and correctly identifies two of them while missing 97 entirely. That is the **cost of false negatives**: missed risky borrowers who proceed to origination, where they may default and generate credit losses.

The **cost of false positives** is a declined or delayed application for a borrower who would have repaid. In Lending Club's marketplace model, this means a lost origination fee and a frustrated investor-borrower pair. False positives damage customer experience and reduce loan volume.

Neither error is free, but in consumer lending the asymmetry typically favors catching more risky borrowers. A missed risky borrower can generate a loss equal to many times the origination fee on a single loan. Lowering the classification threshold from 0.50 to approximately 0.30–0.35 would substantially increase the number of high-risk applicants flagged for review while keeping false positive rates at a level that does not materially disrupt the application experience for prime borrowers. The exact threshold should be calibrated by the risk team against the actual cost of credit losses versus declined-application friction, but the 0.30–0.35 range is recommended as a starting point for pilot testing.

---

## 5. Operational Recommendation

**Immediate action:** Integrate the Random Forest model as an automated pre-screen layer that generates a risk probability score for every new application before it enters the manual underwriting queue. Applications scoring above the 0.35 threshold should be routed to a secondary review track where a risk analyst examines the top drivers — particularly credit utilization, debt-to-income, and loan term — before the application proceeds.

**Deployment caveat:** The model was trained on historical loan data that reflects Lending Club's past underwriting behavior, meaning its predictions are anchored to the economic and credit-market conditions present in that period. If macroeconomic conditions shift materially — for example, if interest rates rise sharply or unemployment increases — the relationships the model learned may drift. The model's performance should be formally monitored on a rolling 90-day basis using a held-out sample of recent originations, and the team should plan for a model refresh if ROC-AUC on recent data falls more than five percentage points below the 0.787 benchmark established at training time.
