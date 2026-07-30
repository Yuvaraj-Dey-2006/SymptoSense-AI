"""
train_model.py
Trains a TF-IDF + Logistic Regression intent classifier for the
AI Medical Symptom Assistant, and saves the model artifacts for
use in the Streamlit app.

Run: python train_model.py
Requires: pandas, scikit-learn, joblib
"""

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("processed_data.csv")

with open("intents.json", "r") as f:
    intents_data = json.load(f)

print(f"Loaded {len(df)} rows across {df['tag'].nunique()} tags")

# -----------------------------
# Build tag -> list of responses (supports multiple responses per tag)
# -----------------------------
tag_to_responses = {
    intent["tag"]: intent["responses"] for intent in intents_data["intents"]
}

# -----------------------------
# Vectorize patterns
# -----------------------------
vectorizer = TfidfVectorizer(ngram_range=(1, 1), lowercase=True, stop_words="english")
X = vectorizer.fit_transform(df["pattern"])
y = df["tag"]

# -----------------------------
# Train classifier
# -----------------------------
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# Quick sanity check on training data itself (not a real eval, just a smoke test)
train_acc = clf.score(X, y)
print(f"Training accuracy (smoke test, not held-out): {train_acc:.3f}")

# -----------------------------
# Save artifacts
# -----------------------------
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(clf, "classifier.pkl")
joblib.dump(tag_to_responses, "tag_to_responses.pkl")

print("Saved: vectorizer.pkl, classifier.pkl, tag_to_responses.pkl")
