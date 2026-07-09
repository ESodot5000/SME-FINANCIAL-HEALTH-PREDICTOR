import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
import lightgbm as lgb
from scipy.stats import randint, loguniform, uniform
from imblearn.over_sampling import SMOTE
from preprocessing import load_data, prepare_features
from config import MODEL_PATH, RANDOM_STATE, TARGET_CLASSES, TARGET_MAPPING, ENCODERS_PATH


def get_models_and_params():
    """
    Define all models. Use class_weight='balanced' for all
    to handle the imbalance (Low=65%, Medium=30%, High=5%).
    """
    models = {
        'Decision Tree': (
            DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE),
            {
                'max_depth':        randint(3, 15),
                'min_samples_leaf': randint(1, 20),
                'max_features': ['sqrt','log2']
            }
        ),
        # TODO: Add Random Forest with class_weight='balanced'
        'Random Forest':(
            RandomForestClassifier( class_weight='balanced', random_state=RANDOM_STATE),
            {
                'n_estimators': randint(50,500),
                'max_depth':        randint(3, 15),
                'min_samples_leaf': randint(1, 20),
                'max_features': ['sqrt','log2'],
                'min_samples_split': randint(2,20)

            }
        ),
        # TODO: Add XGBoost
        'XgBoost': (
            XGBClassifier ( class_weight='balanced', random_state=RANDOM_STATE),
            {
                'n_estimators':     randint(100, 500),
                'max_depth':        randint(3, 12),
                'learning_rate':    loguniform(0.005, 0.3),
                'subsample':        uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4) 
            }
        ),
        # TODO: Add LightGBM
        'LightGBM': (
            lgb.LGBMClassifier( class_weight='balanced',random_state=RANDOM_STATE,verbose=-1),
            {
                'n_estimators':      randint(50, 300),
                'max_depth':         randint(3, 12),
                'learning_rate':     loguniform(0.005, 0.2),
                'num_leaves':        randint(20,150),
                'min_child_samples': randint(5,50),
                'subsample':         uniform(0.6, 0.4),
                'colsample_bytree':  uniform(0.6, 0.4) 
            }
        ),
        # TODO: Add Logistic Regression
        'Logistic Regression': (
            LogisticRegression (class_weight='balanced', max_iter =2000, random_state=RANDOM_STATE),
            {
                'C': loguniform(0.01, 0.3), 
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['lbfgs','saga'],
                'l1_ratio': [0.0, 0.15, 0.5, 0.7, 1.0]
            }
        ),
    }

    return models


def tune_and_compare(models, X_train, y_train, X_test, y_test, n_iter=50, cv=5):
    """
    Run RandomizedSearchCV. Use scoring='f1_macro'.
    """
    results = []
    best_models = {}

    #Applying SMOTE
    #smote = SMOTE(random_state = RANDOM_STATE)
    #X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    #print(f'Before SMOTE: {X_train.shape}')
    #print(f'After SMOTE: {X_train_balanced.shape}')
          
    for name, (model, params) in models.items():
        print(f" \nTuning {name}...")

        # TODO: RandomizedSearchCV with scoring='f1_macro'
        search=RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=n_iter,
            scoring='f1_macro',
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        # TODO: Fit, predict, compute f1_score with average='macro'
          #Fit
        search.fit(X_train, y_train)

          #Predict
        best_model = search.best_estimator_
        best_models[name] = best_model

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:,1]

          #Compute f1_score
        cv_score = search.best_score_
        test_score = f1_score(y_test, y_pred, average = 'macro')
        # TODO: Print classification_report with target_names=TARGET_CLASSES
        print(f'{name} Results')
        print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))
        # TODO: Append results and store best model
        results.append({
            'Model': name,
            'CV F1 Macro': round(cv_score,4),
            'Test F1 Macro': round(test_score,4)
        })
        pass

    results_df = pd.DataFrame(results).sort_values('Test F1 Macro', ascending=False)
    return results_df, best_models


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved: {path}')


if __name__ == '__main__':
    df = load_data()
    print(f'Data loaded: {df.shape}')

    X_train, X_test, y_train, y_test, encoders = prepare_features(df)
    print(f'Train: {X_train.shape}  |  Test: {X_test.shape}')

    models = get_models_and_params()

    print('\nRunning RandomizedSearchCV...\n')
    results_df, best_models = tune_and_compare(models, X_train, y_train, X_test, y_test)

    print('\n--- Model Comparison ---')
    print(results_df[['Model', 'CV F1 Macro', 'Test F1 Macro']].to_string(index=False))

    best_name  = results_df.iloc[0]['Model']
    best_model = best_models[best_name]
    print(f'\nBest model: {best_name}')

    save_model(best_model)

    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump(encoders, f)
    print(f'Encoders saved: {ENCODERS_PATH}')

    print('Training complete.')
   


