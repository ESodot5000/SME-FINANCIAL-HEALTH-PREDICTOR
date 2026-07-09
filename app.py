import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predict import load_model, load_encoders,  predict_single, predict_batch
from config import MODEL_PATH, TARGET_CLASSES, ENCODERS_PATH

st.set_page_config(page_title='SME Financial Health Predictor', layout='wide')
st.title('SME Financial Health Predictor')
st.markdown('Predict the financial health category of a small business across Southern Africa.')

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error('Model not found. Run python src/train.py first.')
        st.stop()
    return load_model()

@st.cache_resource                          # <-- NEW function, placed right after get_model()
def get_encoders():
    if not os.path.exists(ENCODERS_PATH):
        st.error('Encoders not found. Run python src/train.py first.')
        st.stop()
        return load_encoders()

model = get_model()
encoders = get_encoders() 

model = get_model()
tab1, tab2 = st.tabs(['Single Business', 'Batch Prediction'])

with tab1:
    st.header('Single Business Assessment')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('Business Profile')
        country        = st.selectbox('Country', ['eswatini', 'lesotho', 'zimbabwe', 'malawi'])
        owner_age      = st.number_input('Owner Age', 18, 80, 35)
        owner_sex      = st.selectbox('Owner Gender', ['Male', 'Female'])
        business_age_years = st.number_input('Business Age (Years)', 0, 50, 5)
        business_turnover  = st.number_input('Monthly Turnover', 0.0, 1000000.0, 5000.0)
        business_expenses  = st.number_input('Monthly Expenses', 0.0, 500000.0, 3000.0)
        personal_income    = st.number_input('Personal Income', 0.0, 500000.0, 4000.0)

with col2:
    st.subheader('Financial Access')
    has_mobile_money = st.selectbox('Has Mobile Money?', ['Have now', 'Never had', "Used to have but don't have now"])
    has_internet_banking = st.selectbox('Has Internet Banking?', ['Have now', 'Never had', "Used to have but don't have now"])
    has_credit_card  = st.selectbox('Has Credit Card?', ['Have now', 'Never had', "Used to have but don't have now"])
    has_debit_card   = st.selectbox('Has Debit Card?', ['Have now', 'Never had', "Used to have but don't have now"])
    has_loan_account = st.selectbox('Has Loan Account?', ['Have now', 'Never had', "Used to have but don't have now"])
    keeps_financial_records = st.selectbox('Keeps Financial Records?', ['Yes, always', 'Yes, sometimes', 'Yes', 'No'])
    compliance_income_tax = st.selectbox('Pays Income Tax?', ['Yes', 'No'])

with col3:
    st.subheader('Insurance and Risk')
    has_insurance  = st.selectbox('Has Any Insurance?', ['Yes', 'No'])
    funeral_insurance = st.selectbox('Funeral Insurance?', ['Have now', 'Never had', "Used to have but don't have now"])
    medical_insurance = st.selectbox('Medical Insurance?', ['Have now', 'Never had', "Used to have but don't have now"])
    motor_vehicle_insurance = st.selectbox('Motor Vehicle Insurance?', ['Have now', 'Never had', "Used to have but don't have now"])
    uses_informal_lender = st.selectbox('Uses Informal Lender?', ['Have now', 'Never had', "Used to have but don't have now"])
    uses_friends_family_savings = st.selectbox('Uses Friends/Family Savings?', ['Have now', 'Never had', "Used to have but don't have now"])
    offers_credit_to_customers = st.selectbox('Offers Credit to Customers?', ['Yes, always', 'Yes, sometimes', 'No'])
        

if st.button('Assess Financial Health', type='primary'):
        input_data = {
        # === VISIBLE FIELDS — filled in by user ===
        'country':               country,
        'owner_age':             owner_age,
        'owner_sex':             owner_sex,
        'business_age_years':    business_age_years,
        'business_age_months':   business_age_years * 12,
        'business_turnover':     business_turnover,
        'business_expenses':     business_expenses,
        'personal_income':       personal_income,
        'has_mobile_money':      has_mobile_money,
        'has_internet_banking':  has_internet_banking,
        'has_credit_card':       has_credit_card,
        'has_debit_card':        has_debit_card,
        'has_loan_account':      has_loan_account,
        'keeps_financial_records':      keeps_financial_records,
        'compliance_income_tax':        compliance_income_tax,
        'has_insurance':                has_insurance,
        'funeral_insurance':            funeral_insurance,
        'medical_insurance':            medical_insurance,
        'motor_vehicle_insurance':      motor_vehicle_insurance,
        'uses_informal_lender':         uses_informal_lender,
        'uses_friends_family_savings':  uses_friends_family_savings,
        'offers_credit_to_customers':   offers_credit_to_customers,

        # === HIDDEN FIELDS — set to neutral defaults ===
        # Perception columns
        'perception_insurance_doesnt_cover_losses':                      'Unknown',
        'perception_cannot_afford_insurance':                            'Unknown',
        'perception_insurance_companies_dont_insure_businesses_like_yours': 'Unknown',
        'perception_insurance_important':                                'Unknown',
        # Attitude columns
        'attitude_stable_business_environment':                          'Unknown',
        'attitude_satisfied_with_achievement':                           'Unknown',  
        'attitude_worried_shutdown':                                     'Unknown',
        'attitude_more_successful_next_year':                            'Unknown',
        # Other missing columns
        'has_cellphone':               'Unknown',
        'covid_essential_service':     'No',
        'future_risk_theft_stock':     'Unknown',
        'motivation_make_more_money':  'Unknown',
        'marketing_word_of_mouth':     'Unknown',
        'problem_sourcing_money':      'Unknown',
        'current_problem_cash_flow':   'Unknown',
    }    
        result = predict_single(model, input_data, encoders)
        st.divider()
        st.metric('Financial Health Category', result['label'])
        color_map = {'Low': 'error', 'Medium': 'warning', 'High': 'success'}
        if result['label'] == 'High':
            st.success('This business has strong financial health.')
        elif result['label'] == 'Medium':
            st.warning('This business has moderate financial health. Some areas need improvement.')
        else:
            st.error('This business has low financial health. Significant support is needed.')

with tab2:
    st.header('Batch Prediction from CSV')
    uploaded = st.file_uploader('Upload CSV', type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(10))
        if st.button('Run Predictions', type='primary'):
            results = predict_batch(model, df, encoders)
            st.dataframe(results[['Prediction', 'Label']].head(10))
            c1, c2, c3 = st.columns(3)
            c1.metric('Total Businesses', len(results))
            c2.metric('High Financial Health', int((results['Label'] == 'High').sum()))
            c3.metric('Low Financial Health', int((results['Label'] == 'Low').sum()))
            fig, ax = plt.subplots(figsize=(5, 4))
            results['Label'].value_counts().plot(kind='bar', ax=ax,
                color=['tomato', 'steelblue', 'seagreen'], edgecolor='black')
            ax.set_title('Financial Health Distribution')
            ax.tick_params(axis='x', rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.download_button('Download Predictions', results.to_csv(index=False), 'predictions.csv')
