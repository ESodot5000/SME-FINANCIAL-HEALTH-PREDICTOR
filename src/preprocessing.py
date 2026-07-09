import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from config import (
    DATA_PATH, TARGET_COL, DROP_COLS,
    CATEGORICAL_COLS, NUMERICAL_COLS,
    TARGET_MAPPING, TEST_SIZE, RANDOM_STATE
)


def load_data(path=DATA_PATH):
    """Load the SME financial health dataset."""
    # TODO: Read the CSV file and return a DataFrame
    df = pd.read_csv(path)
    return df

def engineer_features(df):
    """
    Creates 4 new features.
    Must be called BEFORE clean_data() and encode_features().
    """
    df = df.copy()

    def has_now(col):
        return df[col].astype(str).str.strip().str.lower() == 'have now'

    def is_yes(col):
        return df[col].astype(str).str.strip().str.lower().str.startswith('yes')

    # FEATURE 1 - INSURANCE SCORE (0-4)
    df['insurance_score'] = (
        is_yes('has_insurance').astype(int) +
        has_now('funeral_insurance').astype(int) +
        has_now('medical_insurance').astype(int) +
        has_now('motor_vehicle_insurance').astype(int)
    )

    # FEATURE 2 - CREDIT ACCESS SCORE (0-4)
    df['credit_access_score'] = (
        has_now('has_loan_account').astype(int) +
        has_now('has_credit_card').astype(int) +
        has_now('has_debit_card').astype(int) +
        has_now('has_internet_banking').astype(int)
    )

    # FEATURE 3 - PROFIT RATIO (capped at 99th percentile to remove extreme outliers)
    df['profit_ratio'] = (
        df['business_turnover'] / (df['business_expenses'] + 1)
    ).round(4)
    df['profit_ratio'] = df['profit_ratio'].clip(upper=266.5)

    # FEATURE 4 - FINANCIAL FORMALITY SCORE (0-2)
    df['financial_formality'] = (
        is_yes('keeps_financial_records').astype(int) +
        is_yes('compliance_income_tax').astype(int)
    )
    return df


def clean_data(df):
    """
    Clean the dataset.
    Steps:
    1. Drop columns in DROP_COLS
    2. Fill missing numerical values with median
    3. Fill missing categorical values with 'Unknown'
    """
    df = df.copy()

    # TODO: Step 1 — drop ID column
    df = df.drop(columns=DROP_COLS, errors='ignore')
    # TODO: Step 2 — fill missing numerical with median
    df[NUMERICAL_COLS] = df[NUMERICAL_COLS].fillna(df[NUMERICAL_COLS].median())
    # TODO: Step 3 — fill missing categorical with 'Unknown'
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna('Unknown')
    return df


def encode_features(df, encoders=None):
    """
    Encode categorical columns.
    - encoders=None -> FIT new encoders (training time), returns (df, encoders)
    - encoders={...} -> REUSE existing encoders (prediction time), returns (df, encoders)
    """
    df = df.copy()
    fit_mode = encoders is None
    if fit_mode:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if fit_mode:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            values = df[col].astype(str)
            known = set(le.classes_)
            values = values.apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(values)

    return df, encoders



def prepare_features(df):
    """
    Full pipeline — returns X_train, X_test, y_train, y_test, encoders
    Encode target: Low=0, Medium=1, High=2
    """
    # TODO: Step 1 — Feature Engineering
    df = engineer_features(df)
    # TODO: Step 2 - clean
    df = clean_data(df)
    # TODO: Step 3 — encode target using TARGET_MAPPING
    df[TARGET_COL]= df[TARGET_COL].map(TARGET_MAPPING) 
    # Hint: df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAPPING)
    # TODO: Step 4 — encode features
    df, encoders = encode_features(df)
    # TODO: Step 5 — split X and y, then train_test_split with stratify=y
    X = df.drop(columns= [TARGET_COL])
    y = df[TARGET_COL]

    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    return X_train, X_test, y_train, y_test, encoders


