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


DEMO_PRESETS = {
    "high": {
        "risk_level": "HIGH",
        "probabilities": {"HIGH": 0.785, "MEDIUM": 0.163, "LOW": 0.052},
        "input": {
            "district": "Barishal",
            "location_type": "rural",
            "sector": "agriculture",
            "business_age_years": 2,
            "owner_experience_years": 1,
            "banking_years": 1,
            "annual_revenue": 1400000.0,
            "loan_amount": 3000000.0,
            "debt_to_asset_ratio": 0.88,
            "num_employees": 3,
            "collateral_value": 0.0,
            "past_defaults": 2,
        },
    },
    "medium": {
        "risk_level": "MEDIUM",
        "probabilities": {"HIGH": 0.212, "MEDIUM": 0.624, "LOW": 0.164},
        "input": {
            "district": "Dhaka",
            "location_type": "semi_urban",
            "sector": "trading",
            "business_age_years": 5,
            "owner_experience_years": 6,
            "banking_years": 3,
            "annual_revenue": 4500000.0,
            "loan_amount": 1500000.0,
            "debt_to_asset_ratio": 0.55,
            "num_employees": 8,
            "collateral_value": 1200000.0,
            "past_defaults": 0,
        },
    },
    "low": {
        "risk_level": "LOW",
        "probabilities": {"HIGH": 0.048, "MEDIUM": 0.134, "LOW": 0.818},
        "input": {
            "district": "Dhaka",
            "location_type": "urban",
            "sector": "manufacturing",
            "business_age_years": 10,
            "owner_experience_years": 12,
            "banking_years": 8,
            "annual_revenue": 12000000.0,
            "loan_amount": 2000000.0,
            "debt_to_asset_ratio": 0.28,
            "num_employees": 25,
            "collateral_value": 3500000.0,
            "past_defaults": 0,
        },
    },
}


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
    return render_template("index.html", choices=CHOICES, demo_presets=DEMO_PRESETS)


@app.get("/health")
def health():
    return jsonify(status="ok", classes=CLASSES)


@app.get("/api/demo")
def get_demo():
    preset_name = request.args.get("preset", "high").lower()
    preset = DEMO_PRESETS.get(preset_name, DEMO_PRESETS["high"])
    return jsonify(
        risk_level=preset["risk_level"],
        probabilities=preset["probabilities"],
        input=preset["input"],
        presets=DEMO_PRESETS,
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    if payload.get("demo_preset"):
        preset = DEMO_PRESETS.get(payload["demo_preset"], DEMO_PRESETS["high"])
        return jsonify(risk_level=preset["risk_level"], probabilities=preset["probabilities"], input=preset["input"])
    row, errors = parse(payload)
    if errors:
        return jsonify(error="Invalid input", details=errors), 400
    label, probs = score(row)
    return jsonify(risk_level=label, probabilities=probs, input=row)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
