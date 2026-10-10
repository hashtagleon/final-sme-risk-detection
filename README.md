# SME Loan Risk Predictor (Flask)

Predicts LOW / MEDIUM / HIGH risk for a small business loan applicant.
The model is the best model from `Final_SME_Risk_Analysis.ipynb`: Gradient Boosting (CV 0.8262, test 0.8075).

## Run locally
```
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000

## Rebuild the model (optional)
`python train_model.py` repeats the notebook's preprocessing, split and Gradient Boosting fit,
and saves `model/best_model.pkl`. `sme_risk_dataset.csv` must sit next to it.

## API
`POST /predict` with JSON:
```
{"district":"Dhaka","location_type":"urban","sector":"retail",
 "business_age_years":5,"owner_experience_years":4,"banking_years":3,
 "annual_revenue":3400000,"loan_amount":800000,"debt_to_asset_ratio":0.45,
 "num_employees":3,"collateral_value":500000,"past_defaults":0}
```
Returns `risk_level` and class `probabilities`. `GET /health` checks the server.

## Deploy on Render
1. Push the contents of this folder to a GitHub repo (app.py at the top level).
2. On Render choose New > Web Service and connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Add an environment variable `PYTHON_VERSION` = `3.13.5`

`render.yaml` and `Procfile` hold the same settings.

## Notes
- `preprocessing.py` must stay next to `app.py`. The saved model needs it.
- Package versions in `requirements.txt` match the ones used to build the model.
  If you rebuild the model on other versions, run `train_model.py` there.
