# Group 3 — SME Financial Health Predictor

## Setup
```bash
conda create -n sme_health python=3.10
conda activate sme_health
pip install -r requirements.txt
```

## Data
Place `sme_financial_health_train_csv.csv` in the `data/` folder.

## Run
```bash
python src/train.py
python src/evaluate.py
streamlit run app.py
```

## Files to Complete
| File | What to do |
|---|---|
| src/preprocessing.py | Complete all TODO sections |
| src/train.py | Add all 5 models and complete tune_and_compare() |
| src/evaluate.py | Complete evaluate_model() and plot functions |
| src/predict.py | Complete predict_single() and predict_batch() |
