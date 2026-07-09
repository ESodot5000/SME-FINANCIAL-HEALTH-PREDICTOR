import pickle
import pandas as pd
from preprocessing import clean_data, encode_features, engineer_features
from config import MODEL_PATH, TARGET_COL, TARGET_CLASSES, ENCODERS_PATH


def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_encoders(path=ENCODERS_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)
    

def align_columns(df, model):
    if hasattr(model, 'feature_names_in_'):
        cols = [c for c in model.feature_names_in_ if c in df.columns]
        return df[cols]
    return df


def predict_single(model, input_dict, encoders):
    """
    Predict financial health category for a single business.
    Returns dict with prediction (int), label (Low/Medium/High), probabilities.
    """
    # TODO: DataFrame, engineer features, clean, encode, align, predict
    #DATAFRAME
    df = pd.DataFrame([input_dict])

    #ENGINEER
    df = engineer_features(df)

    #CLEAN
    df = clean_data(df)

    #ENCODE
    df, _ = encode_features(df, encoders=encoders)

    #ALIGN
    df = align_columns(df, model)

    #PREDICT
    prediction = model.predict(df)[0]

    probabilities = model.predict_proba(df)[0]

   
    # TODO: return {'prediction': int, 'label': TARGET_CLASSES[prediction],
    #               'probabilities': dict of class probabilities}
    return{
        'prediction': int(prediction),
        'label': TARGET_CLASSES[prediction],
        'probabilities': {
            cls: round(float(prob), 4)
            for cls, prob in zip(TARGET_CLASSES, probabilities)
        }
    }
print('predict_single function defined')    


def predict_batch(model, df, encoders):
    """
    Predict financial health for a batch of businesses.
    Returns df with Prediction, Label columns added.
    """
    # TODO: copy, engineer features, clean, encode, align, predict
    df = df.copy()

    #ENGINEER
    df_clean = engineer_features(df)

    #CLEAN
    df_clean= clean_data(df_clean)

    #ENCODE
    df_clean, _ = encode_features(df_clean, encoders=encoders)

    #ALIGN
    df_clean = align_columns(df_clean, model)

    #PREDICT
    predictions = model.predict(df_clean)

    df['Prediction'] = predictions

    df['Label'] = df['Prediction']. apply(lambda x: TARGET_CLASSES[x])

    return df

print('predict_batch function defined')
    
