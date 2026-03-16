"""
train.py  –  Train and save the Fake News Detection model.
Run once:
    python train.py
"""

import pickle
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── Sample dataset ─────────────────────────────────────────────────────────────
# Replace with a larger real dataset for better accuracy.
# Recommended: https://www.kaggle.com/c/fake-news/data

real_headlines = [
    "Scientists discover new vaccine effective against multiple virus strains",
    "Government announces new infrastructure investment plan worth billions",
    "Stock markets reach record highs amid strong economic data",
    "Climate summit concludes with historic agreement on carbon emissions",
    "Tech giant releases quarterly earnings above analyst expectations",
    "New study links exercise to improved mental health outcomes",
    "Central bank raises interest rates to curb inflation",
    "Space agency successfully launches satellite into orbit",
    "International trade deal signed between major economies",
    "Medical researchers develop breakthrough treatment for rare disease",
    "Parliament passes new data protection legislation",
    "Olympic committee announces host city for upcoming games",
    "University research finds link between diet and longevity",
    "Major automaker unveils fully electric vehicle lineup",
    "Global food prices stabilise after months of volatility",
    "Human rights report highlights progress in developing nations",
    "Central bank keeps interest rates unchanged at latest meeting",
    "New public transport line opens in major city centre",
    "Agriculture ministry reports record grain harvest this year",
    "Health authority approves new treatment for chronic illness",
]

fake_headlines = [
    "Government secretly putting mind control chemicals in water supply",
    "Alien spacecraft lands in capital city, officials cover it up",
    "Eating chocolate cures cancer, big pharma hiding the truth",
    "Famous celebrity secretly runs underground criminal empire",
    "Moon landing was filmed in Hollywood studio, documents prove it",
    "5G towers causing mass bird deaths, whistleblower reveals",
    "World leaders are reptilian shapeshifters controlling the economy",
    "Miracle herb banned by government because it cures everything",
    "Vaccines contain microchips to track and monitor the population",
    "Secret cabal of billionaires planning to reduce global population",
    "Doctor reveals truth about sunscreen causing more cancer",
    "Winning lottery numbers leaked by insider every week",
    "Scientists admit global warming is a complete hoax",
    "Underground city of giants discovered beneath the Sahara Desert",
    "Famous dead singer spotted alive at shopping mall in Asia",
    "New law will allow government to confiscate all private property",
    "Drinking bleach solution cures all known viruses says expert",
    "Ancient pyramid discovered on Mars proves advanced alien life",
    "Bank is secretly stealing money from dormant accounts",
    "Free energy device suppressed by oil companies for decades",
]

texts  = real_headlines + fake_headlines
labels = ["REAL"] * len(real_headlines) + ["FAKE"] * len(fake_headlines)

df = pd.DataFrame({"text": texts, "label": labels})
print("Dataset shape:", df.shape)
print(df["label"].value_counts())

# ── Preprocessing ──────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean"] = df["text"].apply(clean_text)

X = df["clean"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Compare models ─────────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True,
    stop_words="english",
)

model_candidates = {
    "Logistic Regression":          LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes":                  MultinomialNB(),
    "Passive Aggressive Classifier":PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    "Linear SVC":                   LinearSVC(max_iter=2000, random_state=42),
}

print("\n=== Model Comparison ===")
best_name, best_pipe, best_acc = None, None, 0

for name, clf in model_candidates.items():
    pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name:35s}  Accuracy = {acc*100:.1f}%")
    if acc > best_acc:
        best_acc, best_name, best_pipe = acc, name, pipe

print(f"\nBest: {best_name}  ({best_acc*100:.1f}%)")
print("\n=== Classification Report ===")
print(classification_report(y_test, best_pipe.predict(X_test)))

# ── Save ───────────────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(best_pipe, f)

print("Saved → model.pkl")
print("Run:  streamlit run app.py")
