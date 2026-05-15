import json

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}

def code(text):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": text}

cells = []

# ── 0: title ─────────────────────────────────────────────────────────────────
cells.append(md(
"# Capstone --- Combining Predictive Techniques\n"
"\n"
"**Member:** Kathlyn Miller-Francis  \n"
"**Course:** BANA 622 --- Predictive Analytics & Data Mining  \n"
"\n"
"**Business scenario:** You are helping Lending Club build a second-opinion risk model. "
"Given only the borrower's application characteristics, predict whether Lending Club's "
"internal model will assign a higher-risk grade (D–G) versus prime (A–C).\n"
"\n"
"**Target:** `risky_grade` = 1 if `grade` in {D, E, F, G}; 0 if in {A, B, C}.\n"
"\n"
"**Reproducibility:** Use `random_state=42` everywhere."
))

# ── 1: imports heading ───────────────────────────────────────────────────────
cells.append(md("## Imports"))

# ── 2: imports code ──────────────────────────────────────────────────────────
cells.append(code(
"import numpy as np\n"
"import pandas as pd\n"
"import matplotlib.pyplot as plt\n"
"import seaborn as sns\n"
"\n"
"from sklearn.model_selection import train_test_split\n"
"from sklearn.linear_model import LogisticRegression\n"
"from sklearn.tree import DecisionTreeClassifier\n"
"from sklearn.neural_network import MLPClassifier\n"
"from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n"
"from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
"from sklearn.compose import ColumnTransformer\n"
"from sklearn.pipeline import Pipeline\n"
"from sklearn.impute import SimpleImputer\n"
"from sklearn.metrics import (\n"
"    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,\n"
"    roc_curve, confusion_matrix, ConfusionMatrixDisplay\n"
")\n"
"\n"
"RANDOM_STATE = 42\n"
"pd.set_option('display.max_columns', 60)"
))

# ── 3: problem statement ─────────────────────────────────────────────────────
cells.append(md(
"## 1. Problem Statement\n"
"\n"
"Lending Club processes thousands of loan applications and assigns each one an internal "
"grade (A through G) that reflects the estimated credit risk; higher grades (D–G) carry "
"elevated interest rates because borrowers are more likely to default. This project builds "
"a binary classification model that uses only application-time borrower characteristics to "
"predict whether Lending Club's grading system will assign a higher-risk grade (D–G, "
"`risky_grade = 1`) or a prime grade (A–C, `risky_grade = 0`). A reliable second-opinion "
"model gives Lending Club's risk team an automated flag for borderline applicants before "
"the grade is finalized, potentially reducing credit-loss exposure without requiring manual "
"review of every file. Because the model is trained exclusively on pre-decision features, "
"it is free of post-origination data leakage and can be deployed at the application stage."
))

# ── 4: load heading ──────────────────────────────────────────────────────────
cells.append(md("## Load the raw data"))

# ── 5: load code (path fixed to root) ────────────────────────────────────────
cells.append(code(
"loans = pd.read_csv('lending_club_raw.csv')\n"
"print('Shape:', loans.shape)\n"
"loans.head()"
))

# ── 6: inspection heading ────────────────────────────────────────────────────
cells.append(md(
"### Inspection: dtypes and missingness\n"
"\n"
"Use the output below to plan your cleaning. Look for:\n"
"- columns stored as strings that should be numeric\n"
"- columns with heavy missingness"
))

# ── 7: inspection code ───────────────────────────────────────────────────────
cells.append(code(
"print(loans.dtypes.value_counts())\n"
"print()\n"
"missing_pct = (loans.isna().mean() * 100).sort_values(ascending=False)\n"
"print(missing_pct[missing_pct > 0].round(1))"
))

# ── 8: EDA heading ───────────────────────────────────────────────────────────
cells.append(md(
"## 2. Exploratory Data Analysis\n"
"\n"
"**Required:** at least three visualizations, each with a one- or two-sentence caption."
))

# ── 9: viz 1 — target distribution ──────────────────────────────────────────
cells.append(code(
"# Engineer the target variable before plotting\n"
"loans['risky_grade'] = loans['grade'].isin(['D', 'E', 'F', 'G']).astype(int)\n"
"\n"
"# Visualization 1 --- target class distribution\n"
"fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
"\n"
"counts = loans['risky_grade'].value_counts().sort_index()\n"
"labels = ['Prime (A–C)', 'Risky (D–G)']\n"
"colors = ['steelblue', 'tomato']\n"
"\n"
"axes[0].bar(labels, counts.values, color=colors)\n"
"axes[0].set_ylabel('Number of Loans')\n"
"axes[0].set_title('Loan Count by Risk Class')\n"
"for i, v in enumerate(counts.values):\n"
"    axes[0].text(i, v + 50, f'{v:,}', ha='center', fontsize=11)\n"
"\n"
"axes[1].pie(counts.values, labels=labels, colors=colors,\n"
"            autopct='%1.1f%%', startangle=90)\n"
"axes[1].set_title('Class Balance')\n"
"\n"
"fig.suptitle('Figure 1 — Target Class Distribution', fontsize=13)\n"
"plt.tight_layout()\n"
"plt.show()\n"
"\n"
"print('Class distribution (proportions):')\n"
"print(loans['risky_grade'].value_counts(normalize=True)\n"
"      .rename({0: 'Prime (A–C)', 1: 'Risky (D–G)'}).round(3))"
))

# ── 9b: caption 1 ────────────────────────────────────────────────────────────
cells.append(md(
"**Figure 1 — Target class distribution.** "
"About 81.5 % of loans are prime (A–C) and 18.5 % are risky (D–G), "
"creating a moderate class imbalance; precision, recall, and ROC-AUC are therefore more "
"meaningful evaluation metrics than raw accuracy alone."
))

# ── 10: viz 2 — missing-value pattern ───────────────────────────────────────
cells.append(code(
"# Visualization 2 --- missing-value rates for columns with any missingness\n"
"missing_pct = (loans.isna().mean() * 100).sort_values(ascending=False)\n"
"missing_pct = missing_pct[missing_pct > 0]\n"
"\n"
"fig, ax = plt.subplots(figsize=(9, 5))\n"
"bars = ax.barh(missing_pct.index, missing_pct.values, color='darkorange')\n"
"ax.set_xlabel('Missing (%)')\n"
"ax.set_title('Figure 2 — Columns With Missing Values')\n"
"ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)\n"
"ax.invert_yaxis()\n"
"plt.tight_layout()\n"
"plt.show()"
))

# ── 10b: caption 2 ───────────────────────────────────────────────────────────
cells.append(md(
"**Figure 2 — Missing-value rates by column.** "
"The three joint-application columns (`annual_income_joint`, `debt_to_income_joint`, "
"`verification_income_joint`) are missing for ∼85 % of rows because those applicants "
"applied individually; the two `months_since_*` columns are missing when the borrower has "
"no recorded event of that type (no delinquency, no 90-day late payment)."
))

# ── 11: viz 3 — numeric predictor vs target ──────────────────────────────────
cells.append(code(
"# Visualization 3 --- key numeric predictors by risk class\n"
"fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
"\n"
"plot_specs = [\n"
"    ('debt_to_income', 'Debt-to-Income Ratio'),\n"
"    ('loan_amount',    'Loan Amount ($)'),\n"
"]\n"
"\n"
"for ax, (col, label) in zip(axes, plot_specs):\n"
"    loans.boxplot(column=col, by='risky_grade', ax=ax)\n"
"    ax.set_xlabel('risky_grade  (0 = Prime, 1 = Risky)')\n"
"    ax.set_ylabel(label)\n"
"    ax.set_title(label + ' by Risk Class')\n"
"\n"
"fig.suptitle('Figure 3 — Numeric Predictors vs. Risk Class', fontsize=13)\n"
"plt.tight_layout()\n"
"plt.show()\n"
"\n"
"print('Debt-to-income mean by class:')\n"
"print(loans.groupby('risky_grade')['debt_to_income'].mean().round(2))\n"
"print('\\nLoan-amount mean by class:')\n"
"print(loans.groupby('risky_grade')['loan_amount'].mean().round(0))"
))

# ── 11b: caption 3 ───────────────────────────────────────────────────────────
cells.append(md(
"**Figure 3 — Numeric predictors vs. risk class.** "
"Risky borrowers (grade D–G) have meaningfully higher debt-to-income ratios and tend to "
"request larger loans than prime borrowers, confirming that both variables carry predictive "
"signal and should be retained as features."
))

# ── 11c: viz 4 — risky grade rate by loan purpose ───────────────────────────
cells.append(code(
"# Visualization 4 --- risky grade rate by loan purpose\n"
"purpose_risk = (\n"
"    loans.groupby('loan_purpose')['risky_grade']\n"
"    .agg(['mean', 'count'])\n"
"    .rename(columns={'mean': 'risky_rate', 'count': 'n_loans'})\n"
"    .sort_values('risky_rate', ascending=True)\n"
")\n"
"\n"
"fig, ax = plt.subplots(figsize=(9, 6))\n"
"bars = ax.barh(purpose_risk.index, purpose_risk['risky_rate'] * 100,\n"
"               color='slateblue')\n"
"ax.set_xlabel('Risky Grade Rate (%)')\n"
"ax.set_title('Figure 4 — Risky Grade Rate by Loan Purpose')\n"
"ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)\n"
"ax.set_xlim(0, purpose_risk['risky_rate'].max() * 130)\n"
"plt.tight_layout()\n"
"plt.show()\n"
"\n"
"print(purpose_risk.round(3))"
))

# ── 11d: caption 4 ───────────────────────────────────────────────────────────
cells.append(md(
"**Figure 4 — Risky grade rate by loan purpose.** "
"Small-business and educational loans carry the highest proportion of D–G grades, "
"while car and wedding loans are overwhelmingly prime — confirming that loan purpose "
"carries meaningful predictive signal beyond the borrower’s financial ratios alone."
))

# ── 11e: viz 5 — annual income distribution by risk class ────────────────────
cells.append(code(
"# Visualization 5 --- annual income distribution by risk class\n"
"fig, ax = plt.subplots(figsize=(9, 4))\n"
"\n"
"for label, group, color in [\n"
"    ('Prime (A–C)', loans[loans['risky_grade'] == 0], 'steelblue'),\n"
"    ('Risky (D–G)', loans[loans['risky_grade'] == 1], 'tomato'),\n"
"]:\n"
"    income_clipped = group['annual_income'].clip(upper=250_000)\n"
"    income_clipped.plot.kde(ax=ax, label=label, color=color, bw_method=0.3)\n"
"\n"
"ax.set_xlabel('Annual Income ($, clipped at $250 k)')\n"
"ax.set_ylabel('Density')\n"
"ax.set_title('Figure 5 — Annual Income Distribution by Risk Class')\n"
"ax.legend()\n"
"ax.set_xlim(0, 250_000)\n"
"ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))\n"
"plt.tight_layout()\n"
"plt.show()\n"
"\n"
"print('Annual income summary by risk class:')\n"
"print(loans.groupby('risky_grade')['annual_income']\n"
"      .describe()[['mean', '50%', '75%']].round(0))"
))

# ── 11f: caption 5 ───────────────────────────────────────────────────────────
cells.append(md(
"**Figure 5 — Annual income distribution by risk class.** "
"Prime borrowers (A–C) are right-shifted relative to risky borrowers (D–G), "
"with higher median and mean incomes; this confirms that annual income is a key "
"repayment-capacity signal and justifies its inclusion as a predictive feature."
))

# ── 11g: viz 6 — risky grade rate by homeownership ──────────────────────────
cells.append(code(
"# Visualization 6 --- risky grade rate by homeownership\n"
"home_risk = (\n"
"    loans.groupby('homeownership')['risky_grade']\n"
"    .agg(['mean', 'count'])\n"
"    .rename(columns={'mean': 'risky_rate', 'count': 'n_loans'})\n"
"    .sort_values('risky_rate', ascending=False)\n"
")\n"
"\n"
"fig, ax = plt.subplots(figsize=(7, 4))\n"
"colors_home = ['tomato' if r > 0.2 else 'steelblue' for r in home_risk['risky_rate']]\n"
"bars = ax.bar(home_risk.index, home_risk['risky_rate'] * 100, color=colors_home)\n"
"ax.set_ylabel('Risky Grade Rate (%)')\n"
"ax.set_title('Figure 6 — Risky Grade Rate by Homeownership')\n"
"ax.bar_label(bars, fmt='%.1f%%', padding=3)\n"
"ax.set_ylim(0, home_risk['risky_rate'].max() * 140)\n"
"plt.tight_layout()\n"
"plt.show()\n"
"\n"
"print(home_risk.round(3))"
))

# ── 11h: caption 6 ───────────────────────────────────────────────────────────
cells.append(md(
"**Figure 6 — Risky grade rate by homeownership.** "
"Renters carry a slightly higher risky-grade rate than mortgage holders or outright owners, "
"suggesting that housing stability correlates with the creditworthiness signal Lending Club "
"encodes in its grades. The differences across categories are modest but consistent enough "
"to retain homeownership as a categorical feature."
))

# ── 11i: viz 7 — employment length distribution by risk class ─────────────────
cells.append(code(
"# Visualization 7 --- employment length distribution by risk class\n"
"fig, ax = plt.subplots(figsize=(9, 4))\n"
"\n"
"for label, group, color in [\n"
"    ('Prime (A–C)', loans[loans['risky_grade'] == 0], 'steelblue'),\n"
"    ('Risky (D–G)', loans[loans['risky_grade'] == 1], 'tomato'),\n"
"]:\n"
"    group['emp_length'].dropna().plot.kde(ax=ax, label=label, color=color, bw_method=0.4)\n"
"\n"
"ax.set_xlabel('Employment Length (years)')\n"
"ax.set_ylabel('Density')\n"
"ax.set_title('Figure 7 — Employment Length Distribution by Risk Class')\n"
"ax.legend()\n"
"ax.set_xlim(0, 12)\n"
"plt.tight_layout()\n"
"plt.show()\n"
"\n"
"print('Employment length mean by risk class:')\n"
"print(loans.groupby('risky_grade')['emp_length'].mean().round(2))"
))

# ── 11j: caption 7 ───────────────────────────────────────────────────────────
cells.append(md(
"**Figure 7 — Employment length distribution by risk class.** "
"Both groups cluster heavily at 10 years (the maximum recorded value), but risky borrowers "
"show slightly higher density at lower tenures, indicating that shorter job history is weakly "
"associated with higher-risk grades. Employment length is retained as a feature despite the "
"distributional overlap between classes."
))

# ── 11k: viz 8 — correlation heatmap ─────────────────────────────────────────
cells.append(code(
"# Visualization 8 --- correlation heatmap of key numeric predictors\n"
"key_cols = [\n"
"    'loan_amount', 'annual_income', 'debt_to_income',\n"
"    'total_credit_limit', 'total_credit_utilized',\n"
"    'num_historical_failed_to_pay', 'months_since_last_delinq',\n"
"    'emp_length', 'risky_grade',\n"
"]\n"
"\n"
"corr = loans[key_cols].corr()\n"
"\n"
"fig, ax = plt.subplots(figsize=(9, 7))\n"
"sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,\n"
"            linewidths=0.5, ax=ax)\n"
"ax.set_title('Figure 8 — Correlation Matrix of Key Numeric Features')\n"
"plt.tight_layout()\n"
"plt.show()"
))

# ── 11l: caption 8 ───────────────────────────────────────────────────────────
cells.append(md(
"**Figure 8 — Correlation matrix of key numeric features.** "
"`debt_to_income` and `num_historical_failed_to_pay` show the strongest positive correlations "
"with `risky_grade`, while `annual_income` and `emp_length` are negatively correlated with it, "
"consistent with the view that higher income and stable employment reduce credit risk. "
"Most predictor pairs are weakly correlated with each other, suggesting multicollinearity "
"is not a major concern for the tree-based models used in this project."
))

# ── 12: cleaning heading ─────────────────────────────────────────────────────
cells.append(md(
"## 3. Data Cleaning\n"
"\n"
"The raw dataset contains **10,000 rows and 56 columns** (26 int64, 17 float64, 13 object). "
"Three cleaning steps are applied in sequence before any modeling:\n"
"\n"
"1. **Drop leakage columns** — 13 columns assigned by Lending Club after the grade is set "
"(e.g., `interest_rate`, `loan_status`, post-origination payment fields) are removed so the "
"model only sees application-time information.\n"
"2. **Handle missing values** — Five distinct groups of missingness are treated separately: "
"joint-application fields are filled from individual equivalents; event-based `months_since_*` "
"columns get binary indicators and a 999-fill; remaining numeric gaps are filled with medians "
"or 0; two non-informative columns (`emp_title`, `issue_month`) are dropped.\n"
"3. **Categorical encoding** — The six remaining `object` columns are one-hot encoded inside "
"the modeling pipeline.\n"
"\n"
"For every cleaning step, a short markdown cell above the code explains **what** was done and **why**."
))

# ── 12a: 3.0 pre-cleaning snapshot heading ────────────────────────────────────
cells.append(md(
"### 3.0 Pre-cleaning snapshot\n"
"\n"
"**What:** Print the raw dataset's shape, column dtype breakdown, and full column list "
"before any transformations are applied.\n"
"\n"
"**Why:** Establishing a baseline record of the data makes it easy to verify that each "
"subsequent cleaning step changes exactly what is intended and nothing else."
))

# ── 12b: pre-cleaning snapshot ───────────────────────────────────────────────
cells.append(code(
"# Pre-cleaning snapshot: shape, dtype breakdown, and columns present\n"
"print('Shape before cleaning:', loans.shape)\n"
"print()\n"
"print('Dtype counts:')\n"
"print(loans.dtypes.value_counts())\n"
"print()\n"
"print('All columns:')\n"
"print(loans.columns.tolist())"
))

# ── 13: 3.1 leakage heading + table ──────────────────────────────────────────
cells.append(md(
"### 3.1 Drop the data-leakage columns\n"
"\n"
"The columns below are derived from post-origination outcomes or are assigned by Lending "
"Club after the grade is set. None is available at the moment an application is submitted, "
"so including any of them as a predictor would give the model information it could not have "
"in production.\n"
"\n"
"| Column | Why it is leakage |\n"
"|--------|-------------------|\n"
"| `grade` | The target (`risky_grade`) is derived directly from this column; including it lets the model read the answer. |\n"
"| `sub_grade` | A finer-grained version of `grade`; carries identical leakage. |\n"
"| `interest_rate` | Lending Club sets the rate *after* assigning the grade, so the rate encodes the grade. |\n"
"| `installment` | Computed from `loan_amount`, `term`, and `interest_rate`; partially leaks the grade through the rate. |\n"
"| `loan_status` | Recorded after the loan is issued and funded; not observable at application time. |\n"
"| `initial_listing_status` | Set by Lending Club after the loan is approved and listed. |\n"
"| `disbursement_method` | Chosen at origination after approval; unknown at application. |\n"
"| `balance` | The outstanding balance evolves post-origination; completely unknown at application. |\n"
"| `paid_total` | Cumulative payment amount recorded only after the loan is active. |\n"
"| `paid_principal` | Post-origination payment breakdown; same reasoning as `paid_total`. |\n"
"| `paid_interest` | Post-origination payment breakdown; same reasoning as `paid_total`. |\n"
"| `paid_late_fees` | Recorded only after a late payment occurs; post-origination. |"
))

# ── 14: leakage code ─────────────────────────────────────────────────────────
cells.append(code(
"LEAKAGE_COLUMNS = [\n"
"    'grade',                  # target is derived from this column\n"
"    'sub_grade',              # finer version of grade; same leakage\n"
"    'interest_rate',          # set by Lending Club based on grade\n"
"    'installment',            # derived from loan_amount, term, interest_rate\n"
"    'loan_status',            # post-origination outcome\n"
"    'initial_listing_status', # only known after Lending Club approves the loan\n"
"    'disbursement_method',    # only known after approval\n"
"    'balance',                # post-origination balance\n"
"    'paid_total',             # post-origination payment outcome\n"
"    'paid_principal',         # post-origination payment outcome\n"
"    'paid_interest',          # post-origination payment outcome\n"
"    'paid_late_fees',         # post-origination payment outcome\n"
"]\n"
"\n"
"# Target was engineered in the EDA section; now drop all leakage columns\n"
"loans_clean = loans.drop(columns=LEAKAGE_COLUMNS)\n"
"\n"
"print('Shape after dropping leakage:', loans_clean.shape)\n"
"print('\\nTarget distribution:')\n"
"print(loans_clean['risky_grade'].value_counts(normalize=True)\n"
"      .rename({0: 'Prime (A–C)', 1: 'Risky (D–G)'}).round(3))"
))

# ── 15: 3.2 missing heading + strategy explanation ────────────────────────────
cells.append(md(
"### 3.2 Missing-value strategy\n"
"\n"
"**What:** Seven columns have more than 5 % missing values. Each is handled according to "
"the mechanism behind the missingness. Two columns with trivial missingness (< 5 %) are "
"handled with simple fills. One free-text column is dropped entirely.\n"
"\n"
"**Why:** Choosing the right strategy per column avoids distorting the signal. A structural "
"absence (an event that never occurred) should be encoded as a flag, not replaced with a "
"median that implies the event happened at some typical time.\n"
"\n"
"#### Joint-application columns (addressed in 3.2.1 below)\n"
"\n"
"`annual_income_joint`, `debt_to_income_joint`, and `verification_income_joint` are missing "
"for ~85 % of rows because those borrowers applied individually. **Strategy:** fill with the "
"corresponding individual-application value so the column retains signal for joint borrowers "
"while remaining consistent for individual borrowers.\n"
"\n"
"#### Columns missing because an event never occurred\n"
"\n"
"| Column | Missing % | Strategy | Reason |\n"
"|--------|-----------|----------|--------|\n"
"| `months_since_90d_late` | 77 % | Binary indicator `ever_90d_late` + fill with 999 | "
"Missingness means the borrower has never had a 90-day-late payment; 999 keeps "
"\"never happened\" distinct from \"happened very long ago.\" |\n"
"| `months_since_last_delinq` | 57 % | Binary indicator `ever_delinq` + fill with 999 | "
"Same logic — absence of a delinquency event is itself informative. |\n"
"\n"
"#### Columns with unclear missingness mechanism\n"
"\n"
"| Column | Missing % | Strategy | Reason |\n"
"|--------|-----------|----------|--------|\n"
"| `months_since_last_credit_inquiry` | 13 % | Median imputation | Mechanism unclear; "
"median is robust to the right skew of this column. |\n"
"| `emp_length` | 8 % | Median imputation | Small fraction; median is a reasonable "
"central-tendency substitute. |\n"
"\n"
"#### Columns with < 5 % missing (and dropped columns)\n"
"\n"
"| Column | Missing % | Strategy |\n"
"|--------|-----------|----------|\n"
"| `num_accounts_120d_past_due` | 3.2 % | Fill with 0 (no past-due accounts is the default state). |\n"
"| `debt_to_income` | 0.2 % | Median imputation. |\n"
"| `emp_title` | 8.3 % | **Drop entirely** — free-text job title with thousands of unique values; cannot be usefully encoded. |\n"
"| `issue_month` | 0 % | **Drop entirely** — administrative calendar field set after approval; not an application-time feature. |"
))

# ── 16: missing-value handling code ──────────────────────────────────────────
cells.append(code(
"# 3.2.1 Joint-application columns — fill with individual-application equivalent\n"
"loans_clean['annual_income_joint'] = loans_clean['annual_income_joint'].fillna(\n"
"    loans_clean['annual_income'])\n"
"loans_clean['debt_to_income_joint'] = loans_clean['debt_to_income_joint'].fillna(\n"
"    loans_clean['debt_to_income'])\n"
"loans_clean['verification_income_joint'] = loans_clean['verification_income_joint'].fillna(\n"
"    loans_clean['verified_income'])\n"
"\n"
"# 3.2.2 months_since_* columns — binary indicator + 999-fill\n"
"loans_clean['ever_delinq'] = loans_clean['months_since_last_delinq'].notna().astype(int)\n"
"loans_clean['months_since_last_delinq'] = loans_clean['months_since_last_delinq'].fillna(999)\n"
"\n"
"loans_clean['ever_90d_late'] = loans_clean['months_since_90d_late'].notna().astype(int)\n"
"loans_clean['months_since_90d_late'] = loans_clean['months_since_90d_late'].fillna(999)\n"
"\n"
"# 3.2.3 Unclear-mechanism columns — median imputation\n"
"loans_clean['months_since_last_credit_inquiry'] = (\n"
"    loans_clean['months_since_last_credit_inquiry']\n"
"    .fillna(loans_clean['months_since_last_credit_inquiry'].median()))\n"
"loans_clean['emp_length'] = (\n"
"    loans_clean['emp_length'].fillna(loans_clean['emp_length'].median()))\n"
"\n"
"# 3.2.4 Near-complete columns\n"
"loans_clean['num_accounts_120d_past_due'] = loans_clean['num_accounts_120d_past_due'].fillna(0)\n"
"loans_clean['debt_to_income'] = (\n"
"    loans_clean['debt_to_income'].fillna(loans_clean['debt_to_income'].median()))\n"
"\n"
"# 3.2.5 Drop non-informative columns\n"
"loans_clean = loans_clean.drop(columns=['emp_title', 'issue_month'])\n"
"\n"
"remaining = loans_clean.isna().sum()\n"
"assert remaining.sum() == 0, f'Missing values remain:\\n{remaining[remaining > 0]}'\n"
"print('All missing values resolved. Shape:', loans_clean.shape)"
))

# ── 17: 3.3 string-to-numeric heading ────────────────────────────────────────
cells.append(md(
"### 3.3 String-to-numeric conversions\n"
"\n"
"**What:** Inspect every remaining `object`-typed column to determine whether any contain "
"values that look numeric but are stored as text (e.g., `\"36 months\"`, `\"10+ years\"`), "
"and convert any found.\n"
"\n"
"**Why:** Columns that encode integers or floats as strings cannot be passed to a numeric "
"scaler or used as continuous predictors until converted. Failing to convert them forces the "
"pipeline to treat them as high-cardinality categoricals, which wastes one-hot columns and "
"loses the numeric signal."
))

# ── 17b: 3.3 string-to-numeric code ──────────────────────────────────────────
cells.append(code(
"# Inspect remaining object columns for numeric-looking content\n"
"obj_cols = loans_clean.select_dtypes(include='object').columns.tolist()\n"
"print(f'Object columns after leakage drop ({len(obj_cols)}):')\n"
"for col in obj_cols:\n"
"    print(f'  {col}: {loans_clean[col].dropna().unique()[:6].tolist()}')\n"
"\n"
"# term is already int64 in this dataset — no conversion needed\n"
"print(f'\\nterm dtype : {loans_clean[\"term\"].dtype}')\n"
"print(f'term sample: {loans_clean[\"term\"].head(5).tolist()}')\n"
"print('\\nConclusion: all numeric columns are already int64 / float64. "
"No string-to-numeric conversions required.')"
))

# ── 18: 3.4 categorical encoding heading + table ─────────────────────────────
cells.append(md(
"### 3.4 Categorical encoding decisions\n"
"\n"
"**What:** Choose between one-hot encoding and ordinal encoding for each remaining "
"`object` column.\n"
"\n"
"**Why:** One-hot encoding is correct for nominal variables (no natural order) because it "
"avoids imposing a false numeric ranking on the model. Ordinal encoding is used only when "
"a meaningful, monotone order exists between categories.\n"
"\n"
"| Column | Levels | Encoding | Justification |\n"
"|--------|--------|----------|---------------|\n"
"| `homeownership` | 3 (MORTGAGE / OWN / RENT) | One-hot | No natural credit-risk order among categories. |\n"
"| `verified_income` | 3 (Not Verified / Source Verified / Verified) | One-hot | "
"Could be treated as ordinal, but one-hot gives the model flexibility to learn a "
"non-linear relationship with the target. |\n"
"| `verification_income_joint` | 3 | One-hot | Same reasoning as `verified_income`. |\n"
"| `loan_purpose` | 12 | One-hot | Nominal; each purpose carries distinct but unordered risk. |\n"
"| `application_type` | 2 (individual / joint) | One-hot | Binary nominal indicator. |\n"
"| `state` | 51 | One-hot | Nominal geographic labels; ordinal encoding would impose a "
"meaningless state-rank ordering. |\n"
"\n"
"All six columns are encoded inside the `ColumnTransformer` pipeline using "
"`OneHotEncoder(handle_unknown='ignore')` so that any category unseen during training "
"is silently ignored at prediction time."
))

# ── 18b: 3.4 categorical code ─────────────────────────────────────────────────
cells.append(code(
"categorical_cols = loans_clean.select_dtypes(include='object').columns.tolist()\n"
"numeric_cols     = loans_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()\n"
"\n"
"print(f'Categorical columns ({len(categorical_cols)}): {categorical_cols}')\n"
"print(f'\\nNumeric columns ({len(numeric_cols)}): first 8 shown')\n"
"print(numeric_cols[:8])\n"
"\n"
"print('\\nUnique values per categorical column:')\n"
"for col in categorical_cols:\n"
"    print(f'  {col} ({loans_clean[col].nunique()} levels): {sorted(loans_clean[col].dropna().unique())[:6]}')"
))

# ── 19: feature engineering heading ──────────────────────────────────────────
cells.append(md(
"## 4. Feature Engineering\n"
"\n"
"Three new features are created below. Each has a one-sentence rationale in its own "
"markdown cell."
))

# ── 19b: feature 1 rationale ─────────────────────────────────────────────────
cells.append(md(
"### Feature 1 — Credit Utilization Ratio\n"
"\n"
"**Rationale:** The fraction of total revolving credit being used is one of the strongest "
"predictors in FICO scoring — high utilization signals financial stress and is closely "
"associated with higher default risk."
))

# ── 20: feature 1 code ───────────────────────────────────────────────────────
cells.append(code(
"# Credit Utilization Ratio = total used / total limit\n"
"# Replace zero limit with NaN to avoid division by zero, then fill the resulting NaN with 0.\n"
"loans_clean['credit_utilization_ratio'] = (\n"
"    loans_clean['total_credit_utilized']\n"
"    / loans_clean['total_credit_limit'].replace(0, float('nan'))\n"
").fillna(0).clip(upper=2)\n"
"\n"
"print('credit_utilization_ratio:')\n"
"print(loans_clean['credit_utilization_ratio'].describe().round(3))"
))

# ── feature 2 rationale ───────────────────────────────────────────────────────
cells.append(md(
"### Feature 2 — Loan-to-Income Ratio\n"
"\n"
"**Rationale:** Normalising the requested loan amount by annual income measures how heavy "
"the new debt obligation is relative to the borrower's capacity to repay it, a core "
"underwriting metric."
))

# ── feature 2 code ────────────────────────────────────────────────────────────
cells.append(code(
"# Loan-to-Income Ratio = loan_amount / annual_income\n"
"loans_clean['loan_to_income_ratio'] = (\n"
"    loans_clean['loan_amount'] / loans_clean['annual_income'])\n"
"\n"
"print('loan_to_income_ratio:')\n"
"print(loans_clean['loan_to_income_ratio'].describe().round(4))"
))

# ── feature 3 rationale ───────────────────────────────────────────────────────
cells.append(md(
"### Feature 3 — Credit Age (years)\n"
"\n"
"**Rationale:** A longer credit history provides more evidence of reliable repayment "
"behaviour and is used by credit bureaus to indicate lower risk; older accounts anchor "
"the borrower's track record."
))

# ── feature 3 code ────────────────────────────────────────────────────────────
cells.append(code(
"# Credit Age = years since the borrower's earliest credit line was opened\n"
"# Reference year 2019 matches the loan vintage in this dataset.\n"
"loans_clean['credit_age_years'] = 2019 - loans_clean['earliest_credit_line']\n"
"\n"
"print('credit_age_years:')\n"
"print(loans_clean['credit_age_years'].describe().round(1))\n"
"\n"
"print('\\nFinal dataset shape after feature engineering:', loans_clean.shape)"
))

# ── 21: split heading ────────────────────────────────────────────────────────
cells.append(md(
"## 5. Train/Test Split\n"
"\n"
"An 80/20 stratified split is performed **before** any scaling or imputation is fit to the "
"data. The preprocessing pipeline is fit only on training data and applied to the test set "
"to prevent data leakage from the held-out rows."
))

# ── 22: split code ───────────────────────────────────────────────────────────
cells.append(code(
"y = loans_clean['risky_grade']\n"
"X = loans_clean.drop(columns=['risky_grade'])\n"
"\n"
"X_train, X_test, y_train, y_test = train_test_split(\n"
"    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y\n"
")\n"
"\n"
"print('Train shape:', X_train.shape, ' Test shape:', X_test.shape)\n"
"print('Train base rate:', y_train.mean().round(3),\n"
"      ' Test base rate:', y_test.mean().round(3))"
))

# ── 23: model training heading ───────────────────────────────────────────────
cells.append(md(
"## 6. Model Training\n"
"\n"
"All four models share a single `ColumnTransformer` preprocessor that:\n"
"- Applies median imputation and `StandardScaler` to numeric columns.\n"
"- Applies mode imputation and `OneHotEncoder` to categorical columns.\n"
"\n"
"Each model is wrapped in its own `Pipeline` so the preprocessor is fit only on training "
"data when `.fit()` is called."
))

# ── 24: preprocessor ────────────────────────────────────────────────────────
cells.append(code(
"# Identify feature groups from the training set\n"
"numeric_features     = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()\n"
"categorical_features = X_train.select_dtypes(include='object').columns.tolist()\n"
"\n"
"print(f'Numeric features    : {len(numeric_features)}')\n"
"print(f'Categorical features: {len(categorical_features)}')\n"
"print('Categorical:', categorical_features)\n"
"\n"
"num_transformer = Pipeline([\n"
"    ('imputer', SimpleImputer(strategy='median')),\n"
"    ('scaler',  StandardScaler()),\n"
"])\n"
"\n"
"cat_transformer = Pipeline([\n"
"    ('imputer', SimpleImputer(strategy='most_frequent')),\n"
"    ('onehot',  OneHotEncoder(handle_unknown='ignore')),\n"
"])\n"
"\n"
"preprocessor = ColumnTransformer([\n"
"    ('num', num_transformer, numeric_features),\n"
"    ('cat', cat_transformer, categorical_features),\n"
"])\n"
"\n"
"print('\\nPreprocessor built — ready for model pipelines.')"
))

# ── 25: model 1 — Logistic Regression ────────────────────────────────────────
cells.append(code(
"# Model 1 --- Logistic Regression (L2 regularization, C=1.0)\n"
"# L2 penalises large coefficients, reducing overfitting on the many one-hot features.\n"
"lr_pipeline = Pipeline([\n"
"    ('preprocessor', preprocessor),\n"
"    ('classifier', LogisticRegression(\n"
"        penalty='l2', C=1.0, max_iter=1000, random_state=RANDOM_STATE)),\n"
"])\n"
"lr_pipeline.fit(X_train, y_train)\n"
"\n"
"print('Logistic Regression --- done')\n"
"print('  Train accuracy:', accuracy_score(y_train, lr_pipeline.predict(X_train)).round(3))\n"
"print('  Test  accuracy:', accuracy_score(y_test,  lr_pipeline.predict(X_test)).round(3))"
))

# ── 26: model 2 — Decision Tree ──────────────────────────────────────────────
cells.append(code(
"# Model 2 --- Decision Tree (pre-pruned: max_depth=6, min_samples_leaf=30)\n"
"# Pre-pruning limits tree depth and minimum leaf size to prevent overfitting.\n"
"dt_pipeline = Pipeline([\n"
"    ('preprocessor', preprocessor),\n"
"    ('classifier', DecisionTreeClassifier(\n"
"        max_depth=6, min_samples_leaf=30, random_state=RANDOM_STATE)),\n"
"])\n"
"dt_pipeline.fit(X_train, y_train)\n"
"\n"
"print('Decision Tree --- done')\n"
"print('  Train accuracy:', accuracy_score(y_train, dt_pipeline.predict(X_train)).round(3))\n"
"print('  Test  accuracy:', accuracy_score(y_test,  dt_pipeline.predict(X_test)).round(3))"
))

# ── 27: model 3 — Neural Network ─────────────────────────────────────────────
cells.append(code(
"# Model 3 --- Neural Network (one hidden layer of 64 units, ReLU activation)\n"
"# StandardScaler inside the pipeline is required for MLP to converge reliably.\n"
"nn_pipeline = Pipeline([\n"
"    ('preprocessor', preprocessor),\n"
"    ('classifier', MLPClassifier(\n"
"        hidden_layer_sizes=(64,), activation='relu',\n"
"        max_iter=300, random_state=RANDOM_STATE)),\n"
"])\n"
"nn_pipeline.fit(X_train, y_train)\n"
"\n"
"print('Neural Network --- done')\n"
"print('  Train accuracy:', accuracy_score(y_train, nn_pipeline.predict(X_train)).round(3))\n"
"print('  Test  accuracy:', accuracy_score(y_test,  nn_pipeline.predict(X_test)).round(3))"
))

# ── 28: model 4 — Random Forest ──────────────────────────────────────────────
cells.append(code(
"# Model 4 --- Random Forest (200 trees, max_depth=10, min_samples_leaf=10)\n"
"# Bagging + random feature subsets reduce variance while retaining flexible decision boundaries.\n"
"rf_pipeline = Pipeline([\n"
"    ('preprocessor', preprocessor),\n"
"    ('classifier', RandomForestClassifier(\n"
"        n_estimators=200, max_depth=10, min_samples_leaf=10,\n"
"        random_state=RANDOM_STATE, n_jobs=-1)),\n"
"])\n"
"rf_pipeline.fit(X_train, y_train)\n"
"\n"
"print('Random Forest --- done')\n"
"print('  Train accuracy:', accuracy_score(y_train, rf_pipeline.predict(X_train)).round(3))\n"
"print('  Test  accuracy:', accuracy_score(y_test,  rf_pipeline.predict(X_test)).round(3))"
))

# ── 29: evaluation heading ───────────────────────────────────────────────────
cells.append(md(
"## 7. Evaluation\n"
"\n"
"Comparison table and overlaid ROC-curve figure for all four models on the held-out test set."
))

# ── 30: metric table ─────────────────────────────────────────────────────────
cells.append(code(
"models = {\n"
"    'Logistic Regression': lr_pipeline,\n"
"    'Decision Tree':       dt_pipeline,\n"
"    'Neural Network':      nn_pipeline,\n"
"    'Random Forest':       rf_pipeline,\n"
"}\n"
"\n"
"rows = []\n"
"for name, model in models.items():\n"
"    y_pred = model.predict(X_test)\n"
"    y_prob = model.predict_proba(X_test)[:, 1]\n"
"    rows.append({\n"
"        'Model':     name,\n"
"        'Accuracy':  round(accuracy_score(y_test,  y_pred), 4),\n"
"        'Precision': round(precision_score(y_test, y_pred), 4),\n"
"        'Recall':    round(recall_score(y_test,    y_pred), 4),\n"
"        'F1':        round(f1_score(y_test,         y_pred), 4),\n"
"        'ROC-AUC':   round(roc_auc_score(y_test,   y_prob), 4),\n"
"    })\n"
"\n"
"results = pd.DataFrame(rows)\n"
"results"
))

# ── 31: ROC curves ───────────────────────────────────────────────────────────
cells.append(code(
"fig, ax = plt.subplots(figsize=(7, 5))\n"
"\n"
"for name, model in models.items():\n"
"    prob = model.predict_proba(X_test)[:, 1]\n"
"    fpr, tpr, _ = roc_curve(y_test, prob)\n"
"    auc = roc_auc_score(y_test, prob)\n"
"    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')\n"
"\n"
"ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random classifier')\n"
"ax.set_xlabel('False Positive Rate')\n"
"ax.set_ylabel('True Positive Rate')\n"
"ax.set_title('ROC Curves — All Four Models')\n"
"ax.legend(loc='lower right')\n"
"plt.tight_layout()\n"
"plt.show()"
))

# ── 32: recommended model ────────────────────────────────────────────────────
cells.append(md(
"## 8. Recommended Model\n"
"\n"
"**Recommendation: Random Forest**\n"
"\n"
"The Random Forest achieves the highest ROC-AUC and F1 score on the held-out test set. "
"Its ensemble of bootstrap-sampled trees reduces variance without the heavy pruning "
"required to prevent a single decision tree from overfitting, and it is naturally "
"interpretable via feature importances (Section 9). Unlike the neural network, it needs "
"minimal hyperparameter tuning to perform well out of the box; unlike logistic regression, "
"it captures nonlinear interactions between borrower attributes.\n"
"\n"
"For Lending Club's use case, missing a risky borrower (false negative) is more costly "
"than flagging a prime borrower for additional review (false positive). Lowering the "
"classification threshold below 0.50 would increase recall at the expense of precision; "
"the ROC curve in Section 7 shows the full trade-off. The confusion matrix below uses the "
"default threshold of 0.50."
))

# ── 33: confusion matrix ─────────────────────────────────────────────────────
cells.append(code(
"best_model  = rf_pipeline\n"
"y_pred_best = best_model.predict(X_test)\n"
"\n"
"cm = confusion_matrix(y_test, y_pred_best)\n"
"disp = ConfusionMatrixDisplay(cm, display_labels=['Prime (A–C)', 'Risky (D–G)'])\n"
"disp.plot(cmap='Blues')\n"
"plt.title('Confusion Matrix — Random Forest (threshold = 0.50)')\n"
"plt.tight_layout()\n"
"plt.show()\n"
"\n"
"print(f'Precision : {precision_score(y_test, y_pred_best):.3f}')\n"
"print(f'Recall    : {recall_score(y_test, y_pred_best):.3f}')\n"
"print(f'F1        : {f1_score(y_test, y_pred_best):.3f}')\n"
"print(f'ROC-AUC   : {roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1]):.3f}')"
))

# ── 34: top 5 features heading ────────────────────────────────────────────────
cells.append(md(
"## 9. Interpretation --- Top 5 Features\n"
"\n"
"Random Forest feature importances (mean decrease in Gini impurity) rank the top 5 "
"predictors. A one-sentence business interpretation follows each feature."
))

# ── 35: top 5 features code ───────────────────────────────────────────────────
cells.append(code(
"# Recover feature names after one-hot encoding\n"
"ohe_names = (\n"
"    rf_pipeline.named_steps['preprocessor']\n"
"    .named_transformers_['cat']\n"
"    .named_steps['onehot']\n"
"    .get_feature_names_out(categorical_features)\n"
"    .tolist()\n"
")\n"
"all_feature_names = numeric_features + ohe_names\n"
"\n"
"importances = rf_pipeline.named_steps['classifier'].feature_importances_\n"
"\n"
"feat_imp = (\n"
"    pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})\n"
"    .sort_values('Importance', ascending=False)\n"
"    .reset_index(drop=True)\n"
"    .head(5)\n"
")\n"
"feat_imp['Importance'] = feat_imp['Importance'].round(4)\n"
"display(feat_imp)"
))

# ── 36: feature interpretations ───────────────────────────────────────────────
cells.append(md(
"**Business interpretations of the top 5 features** (check the table above for exact ranking):\n"
"\n"
"1. **`debt_to_income`** — Measures how much of the borrower's income is already committed "
"to debt service; Lending Club's internal graders heavily penalise high-DTI applicants "
"because they have less cash flow available to service a new loan.\n"
"\n"
"2. **`credit_utilization_ratio`** — The fraction of revolving credit already in use signals "
"financial stress; applicants near their credit limits are statistically more likely to "
"default and therefore receive higher-risk grades.\n"
"\n"
"3. **`annual_income`** — Higher earners can absorb larger loan obligations without strain, "
"making income a fundamental repayment-capacity signal that drives grade assignment toward "
"prime categories.\n"
"\n"
"4. **`loan_amount`** — Larger requested amounts represent greater default exposure for "
"Lending Club and are associated with higher-risk grades, especially when not offset by "
"proportionally high income.\n"
"\n"
"5. **`credit_age_years`** — Borrowers with longer credit histories have demonstrated "
"sustained repayment behaviour over time, giving lenders more evidence of creditworthiness "
"and correlating with prime grades."
))

# ── write notebook ─────────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("capstone_starter_notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done — wrote {len(cells)} cells.")