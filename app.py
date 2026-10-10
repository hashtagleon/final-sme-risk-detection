import json, os, pickle
import pandas as pd
from flask import Flask, jsonify, render_template, request
import preprocessing  # noqa: F401  (the pickled pipeline needs ToDense from this module)

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

with open(os.path.join(BASE, "model", "best_model.pkl"), "rb") as f:
    MODEL = pickle.load(f)
FEATURES = json.load(open(os.path.join(BASE, "model", "feature_columns.json")))
CLASSES = list(MODEL.classes_)

CHOICES = {
    "district": ["Barishal", "Chattogram", "Dhaka", "Khulna", "Rajshahi", "Rangpur", "Sylhet"],
    "location_type": ["urban", "semi_urban", "rural"],
    "sector": ["agriculture", "food_processing", "manufacturing", "retail", "services", "textiles", "trading"],
}
INT_FIELDS = ["business_age_years", "owner_experience_years", "banking_years", "num_employees", "past_defaults"]
FLOAT_FIELDS = ["annual_revenue", "loan_amount", "debt_to_asset_ratio", "collateral_value"]
NON_NEGATIVE = INT_FIELDS + FLOAT_FIELDS


def parse(payload):
    """Validate input and return (row_dict, errors)."""
    row, errors = {}, []
    for k, opts in CHOICES.items():
        v = str(payload.get(k, "")).strip()
        if v not in opts:
            errors.append(f"{k} must be one of {opts}")
        row[k] = v
    for k in INT_FIELDS + FLOAT_FIELDS:
        raw = payload.get(k)
        try:
            val = int(float(raw)) if k in INT_FIELDS else float(raw)
        except (TypeError, ValueError):
            errors.append(f"{k} must be a number")
            continue
        if val < 0:
            errors.append(f"{k} cannot be negative")
        row[k] = val
    return row, errors


def score(row):
    df = pd.DataFrame([row], columns=FEATURES)
    proba = MODEL.predict_proba(df)[0]
    probs = {c: round(float(p), 4) for c, p in zip(CLASSES, proba)}
    label = max(probs, key=probs.get)
    return label, probs


@app.get("/")
def index():
    return render_template("index.html", choices=CHOICES)


@app.get("/health")
def health():
    return jsonify(status="ok", classes=CLASSES)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    row, errors = parse(payload)
    if errors:
        return jsonify(error="Invalid input", details=errors), 400
    label, probs = score(row)
    return jsonify(risk_level=label, probabilities=probs, input=row)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
