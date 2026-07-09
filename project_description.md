# Group 3 - SME Financial Health Prediction (Southern Africa)

## The Business Problem

Across Southern Africa, small and medium-sized enterprises (SMEs) are the
backbone of employment and economic growth. Yet many remain financially
fragile — excluded from formal credit, struggling with cash flow, and
vulnerable to shocks like illness, drought, or theft.

Traditional measures like revenue or profit do not capture how truly
financially healthy a business is. A business can be profitable today
but have no savings, no insurance, and no ability to survive a crisis next month.

FinMark Trust and data.org introduced a Financial Health Index (FHI) —
a composite measure that classifies businesses as Low, Medium, or High
financial health across four dimensions: savings and assets, debt and
repayment ability, resilience to shocks, and access to financial services.

## The Solution

Build a multi-class classification model that predicts whether a small
business falls into Low, Medium, or High financial health based on
survey data about the business owner, business characteristics,
and financial behaviour.

## The Business Value

- Financial institutions can identify creditworthy SMEs more accurately
- Governments can target support and subsidies to the most vulnerable businesses
- Development organisations can measure the impact of financial inclusion programmes
- Business owners can understand their financial health and where to improve

## The Data

Survey data collected from small business owners in four Southern African
countries: Eswatini, Lesotho, Zimbabwe, and Malawi. Data collected by
FinMark Trust through their FinScope SME survey programme.


## Target Variable

| Value | Meaning |
|---|---|
| Low | Business has poor financial health — high risk, limited resilience |
| Medium | Business has moderate financial health — some stability but vulnerable |
| High | Business has strong financial health — resilient and financially included |

Class distribution: Low 65.3%, Medium 29.8%, High 4.9%
This is an imbalanced multi-class problem.

## Column Descriptions

### Business Owner Profile

| Column | Description |
|---|---|
| `ID` | Unique business identifier |
| `country` | Country (eswatini, lesotho, zimbabwe, malawi) |
| `owner_age` | Age of the business owner in years |
| `owner_sex` | Gender of the business owner |

### Business Characteristics

| Column | Description |
|---|---|
| `business_age_years` | How long the business has been operating in years |
| `business_age_months` | Business age in months (more precise) |
| `business_turnover` | Monthly or annual revenue of the business |
| `business_expenses` | Approximate monthly or annual expenses |
| `personal_income` | Owner's total monthly personal income |

### Financial Products and Access

| Column | Description |
|---|---|
| `has_mobile_money` | Whether the owner uses mobile money (Yes/No) |
| `has_cellphone` | Whether the owner has a cellphone (Yes/No) |
| `has_credit_card` | Whether the owner has a credit card (Yes/No) |
| `has_loan_account` | Whether the business has a loan account (Yes/No) |
| `has_internet_banking` | Whether the owner uses internet banking (Yes/No) |
| `has_debit_card` | Whether the owner has a debit card (Yes/No) |
| `has_insurance` | Whether the business has any insurance (Yes/No) |
| `motor_vehicle_insurance` | Whether the business has vehicle insurance |
| `medical_insurance` | Whether the owner has medical insurance |
| `funeral_insurance` | Whether the owner has funeral insurance |

### Business Behaviour and Attitudes

| Column | Description |
|---|---|
| `keeps_financial_records` | Whether the business keeps financial records |
| `offers_credit_to_customers` | Whether the business gives credit to customers |
| `compliance_income_tax` | Whether the business pays income tax |
| `uses_friends_family_savings` | Whether owner borrows from friends or family |
| `uses_informal_lender` | Whether owner uses an informal lender (e.g. loan shark) |
| `marketing_word_of_mouth` | Primary marketing method is word of mouth |
| `problem_sourcing_money` | Whether accessing finance is a current problem |
| `current_problem_cash_flow` | Whether cash flow is a current problem |

### Owner Perceptions and Attitudes

| Column | Description |
|---|---|
| `attitude_stable_business_environment` | Owner believes the business environment will be stable |
| `attitude_worried_shutdown` | Owner is worried the business will shut down |
| `attitude_satisfied_with_achievement` | Owner is satisfied with business achievements |
| `attitude_more_successful_next_year` | Owner expects to be more successful next year |
| `perception_insurance_doesnt_cover_losses` | Owner believes insurance does not cover their losses |
| `perception_cannot_afford_insurance` | Owner believes they cannot afford insurance |
| `perception_insurance_important` | Owner believes insurance is important |
| `perception_insurance_companies_dont_insure_businesses_like_yours` | Owner believes insurers do not cover businesses like theirs |

### Risk Exposure

| Column | Description |
|---|---|
| `future_risk_theft_stock` | Owner is worried about theft of stock in future |
| `covid_essential_service` | Business was classified as essential during COVID-19 |
| `motivation_make_more_money` | Primary motivation is making more money |

## Key Challenges

- Multi-class target with imbalance: Low dominates at 65.3%
- Many binary Yes/No columns that need encoding
- Missing values across multiple columns
- 4 different countries — country may be an important feature

## Evaluation Metric

F1 Score (macro) — averages F1 equally across all three classes.
This prevents the model from ignoring the rare High class.

## Suggested Approach

1. Encode binary Yes/No columns to 1/0
2. Encode country and other categorical columns
3. Fill missing values — mode for categorical, median for numerical
4. Use class_weight='balanced' to handle imbalance
5. Train and compare: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM
6. Tune with RandomizedSearchCV scoring='f1_macro'
7. Check per-class F1 to ensure the model is not ignoring the High class
