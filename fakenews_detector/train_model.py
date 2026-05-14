"""
Fake News Detection — ML Training Pipeline
==========================================
Dataset:  Synthetically generated news samples modeled after WELFake/LIAR structure
          (label: 0 = real, 1 = fake)
Features: TF-IDF unigrams + bigrams with sublinear TF scaling
Models:   Logistic Regression, Naive Bayes, Random Forest, Gradient Boosting
          → Best model persisted via joblib for serving
"""

import pandas as pd
import numpy as np
import re
import joblib
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

DATA_PATH  = 'data/news_dataset.csv'
MODEL_DIR  = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Text preprocessing ───────────────────────────────────────────────────────
ENGLISH_STOP_WORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'by','from','up','about','into','through','during','is','are','was',
    'were','be','been','being','have','has','had','do','does','did','will',
    'would','could','should','may','might','shall','can','need','dare',
    'ought','used','this','that','these','those','i','you','he','she','it',
    'we','they','who','which','what','there','here','when','where','why',
    'how','all','each','every','both','few','more','most','other','some',
    'such','no','not','only','same','so','than','too','very','just','also',
}

SENSATIONALIST = {
    'BREAKING', 'URGENT', 'EXPOSED', 'SHOCKING', 'BANNED', 'PROOF',
    'WARNING', 'LEAKED', 'BOMBSHELL', 'TRUTH', 'WAKE UP', 'SHARE',
    'MIRACLE', 'DELETED', 'SILENCED', 'FURIOUS', 'SUPPRESSED',
}

def preprocess(text: str) -> str:
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\b[A-Z]{2,}\b', lambda m: m.group().lower(), text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

def sensationalism_score(text: str) -> float:
    words = set(text.upper().split())
    return sum(1 for w in SENSATIONALIST if w in words) / len(SENSATIONALIST)

def exclamation_ratio(text: str) -> float:
    return text.count('!') / max(len(text.split()), 1)

def caps_ratio(text: str) -> float:
    words = text.split()
    if not words: return 0.0
    return sum(1 for w in words if w.isupper() and len(w) > 2) / len(words)

# ─── Load & prepare data ──────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)
df['processed'] = df['text'].apply(preprocess)
df['sensationalism'] = df['text'].apply(sensationalism_score)
df['exclamation']    = df['text'].apply(exclamation_ratio)
df['caps_ratio']     = df['text'].apply(caps_ratio)

X_text = df['processed']
y      = df['label']

X_train_t, X_test_t, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train_t)}  |  Test: {len(X_test_t)}")
print(f"Class balance — 0 (real): {(y==0).sum()}  1 (fake): {(y==1).sum()}\n")

# ─── Define model pipelines ───────────────────────────────────────────────────
tfidf_params = dict(
    ngram_range=(1, 2),
    max_features=25000,
    sublinear_tf=True,
    min_df=2,
    stop_words=list(ENGLISH_STOP_WORDS),
)

models = {
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')),
    ]),
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   MultinomialNB(alpha=0.1)),
    ]),
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ]),
    'Gradient Boosting': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000, sublinear_tf=True, min_df=2, stop_words=list(ENGLISH_STOP_WORDS))),
        ('clf',   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                              max_depth=4, random_state=42)),
    ]),
}

# ─── Train & evaluate ─────────────────────────────────────────────────────────
results = {}
best_model_name = None
best_accuracy   = 0.0

for name, pipe in models.items():
    print(f"Training {name} …")
    pipe.fit(X_train_t, y_train)
    y_pred = pipe.predict(X_test_t)
    y_prob = pipe.predict_proba(X_test_t)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cv   = cross_val_score(pipe, X_text, y, cv=5, scoring='accuracy')
    cm   = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'], output_dict=True)

    results[name] = {
        'accuracy': round(acc, 4),
        'roc_auc':  round(auc, 4),
        'cv_mean':  round(cv.mean(), 4),
        'cv_std':   round(cv.std(),  4),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }

    print(f"  Accuracy : {acc:.4f}  |  AUC: {auc:.4f}  |  CV: {cv.mean():.4f} ± {cv.std():.4f}")

    if acc > best_accuracy:
        best_accuracy   = acc
        best_model_name = name

print(f"\n✓ Best model: {best_model_name}  (accuracy={best_accuracy:.4f})")

# ─── Persist best model ───────────────────────────────────────────────────────
best_pipe = models[best_model_name]
joblib.dump(best_pipe, f'{MODEL_DIR}/best_model.pkl')
print(f"  → Saved to {MODEL_DIR}/best_model.pkl")

# Save all pipelines too
for name, pipe in models.items():
    fname = name.lower().replace(' ', '_')
    joblib.dump(pipe, f'{MODEL_DIR}/{fname}.pkl')

# Save results metadata
meta = {
    'best_model': best_model_name,
    'best_accuracy': best_accuracy,
    'model_results': results,
    'features': {
        'tfidf_ngram_range': list(tfidf_params['ngram_range']),
        'tfidf_max_features': tfidf_params['max_features'],
        'extra_features': ['sensationalism_score', 'exclamation_ratio', 'caps_ratio'],
    }
}
with open(f'{MODEL_DIR}/metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\n=== Final Results ===")
for name, r in results.items():
    print(f"  {name:<22} acc={r['accuracy']:.4f}  auc={r['roc_auc']:.4f}  cv={r['cv_mean']:.4f}±{r['cv_std']:.4f}")
print("\nTraining complete!")
