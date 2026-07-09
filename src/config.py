import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'sme_financial_health_train.csv')
ENCODERS_PATH = os.path.join(BASE_DIR, 'models', 'encoders.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')

TARGET_COL   = 'Target'
TEST_SIZE    = 0.2
RANDOM_STATE = 42

DROP_COLS = ['ID']

CATEGORICAL_COLS = [
    'country', 'owner_sex',
    'perception_insurance_doesnt_cover_losses',
    'perception_cannot_afford_insurance',
    'has_mobile_money', 'offers_credit_to_customers', 
    'perception_insurance_companies_dont_insure_businesses_like_yours',
    'perception_insurance_important', 'covid_essential_service',  
    'uses_friends_family_savings', 'uses_informal_lender','funeral_insurance', 'has_insurance',
    'medical_insurance', 'motor_vehicle_insurance',
    'has_loan_account', 'has_credit_card',
    'has_debit_card', 'has_internet_banking',
    'keeps_financial_records', 'compliance_income_tax','problem_sourcing_money', 'attitude_stable_business_environment', 'motivation_make_more_money',
    'attitude_satisfied_with_achievement', 'marketing_word_of_mouth',
    'attitude_worried_shutdown', 'future_risk_theft_stock',
    'current_problem_cash_flow', 'has_cellphone', 'attitude_more_successful_next_year'
]

NUMERICAL_COLS = [
    'owner_age', 'personal_income', 'business_expenses',
    'business_turnover', 'business_age_years', 'business_age_months',
    'insurance_score',       
    'credit_access_score',   
    'profit_ratio',          
    'financial_formality'   
]

TARGET_CLASSES  = ['Low', 'Medium', 'High']
TARGET_MAPPING  = {'Low': 0, 'Medium': 1, 'High': 2}

MODEL_PARAMS = {
    'n_estimators':  200,
    'learning_rate': 0.05,
    'num_leaves':    31,
    'random_state':  RANDOM_STATE,
    'verbose':       -1
}