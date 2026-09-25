"""
Generates the bundled sample model + dataset used by the "Use sample
model" toggle on the New Analysis page. Run once:

    python sample_data/generate_sample.py
"""

import pickle

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer(as_frame=True)
df = data.frame  # features + 'target' column

# Deliberately under-fit a bit (shallow trees, few estimators) so the
# sample analysis actually has interesting failures to show off.
model = RandomForestClassifier(n_estimators=15, max_depth=3, random_state=42)
X = df.drop(columns=["target"])
y = df["target"]
model.fit(X, y)

with open("sample_data/sample_model.pkl", "wb") as f:
    pickle.dump(model, f)

df.to_csv("sample_data/sample_dataset.csv", index=False)

print("Wrote sample_data/sample_model.pkl and sample_data/sample_dataset.csv")
print(f"Sample accuracy on training data: {model.score(X, y):.2%}")
