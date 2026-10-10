"""Rebuilds the model exactly as in Final_SME_Risk_Analysis.ipynb
(best model by CV accuracy = Gradient Boosting) and saves it for the Flask app.

Run:  python train_model.py
"""
import json, os, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from preprocessing import ToDense

np.random.seed(42)
BASE = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE, "sme_risk_dataset.csv"))
df = df.drop(columns=[c for c in df.columns if c.lower() in {"id", "applicant_id"}])
df.columns = [c.strip() for c in df.columns]
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].astype(str).str.strip()
df = df.drop_duplicates().reset_index(drop=True)

X, y = df.drop(columns=["risk_level"]), df["risk_level"]
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler(with_mean=False))]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
pipe = Pipeline([("pre", pre), ("dense", ToDense()),
                 ("clf", GradientBoostingClassifier(random_state=42))]).fit(X_train, y_train)

acc = accuracy_score(y_test, pipe.predict(X_test))
print(f"Test accuracy: {acc:.4f}")
assert acc >= 0.80

os.makedirs(os.path.join(BASE, "model"), exist_ok=True)
pickle.dump(pipe, open(os.path.join(BASE, "model", "best_model.pkl"), "wb"))
json.dump(X.columns.tolist(), open(os.path.join(BASE, "model", "feature_columns.json"), "w"), indent=2)
json.dump(sorted(y.unique().tolist()), open(os.path.join(BASE, "model", "class_labels.json"), "w"))
print("Saved model/best_model.pkl")
