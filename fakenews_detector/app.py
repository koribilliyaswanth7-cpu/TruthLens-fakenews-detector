"""
Fake News Detector — Flask API Server
"""

from flask import Flask, request, jsonify, send_from_directory
import joblib
import json
import re
import os
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='templates')

# ─── Load artifacts ───────────────────────────────────────────────────────────
MODEL_DIR = 'models'
models    = {}
metadata  = {}

for name in ['logistic_regression', 'naive_bayes', 'random_forest', 'gradient_boosting']:
    path = f'{MODEL_DIR}/{name}.pkl'
    if os.path.exists(path):
        models[name] = joblib.load(path)

with open(f'{MODEL_DIR}/metadata.json') as f:
    metadata = json.load(f)

SENSATIONALIST = {
    'BREAKING','URGENT','EXPOSED','SHOCKING','BANNED','PROOF',
    'WARNING','LEAKED','BOMBSHELL','TRUTH','WAKE UP','SHARE',
    'MIRACLE','DELETED','SILENCED','FURIOUS','SUPPRESSED',
}

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

# ─── Feature helpers ──────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\b[A-Z]{2,}\b', lambda m: m.group().lower(), text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

def get_signals(text: str) -> dict:
    words = text.split()
    upper_words = [w for w in words if w.isupper() and len(w) > 2]
    exclamations = text.count('!')
    questions    = text.count('?')
    sensational  = [w for w in SENSATIONALIST if w in text.upper()]
    has_sources  = bool(re.search(r'\b(according to|study|report|researchers|officials|data|survey)\b', text.lower()))
    has_numbers  = bool(re.search(r'\d+', text))
    all_caps_pct = round(len(upper_words) / max(len(words), 1) * 100, 1)

    red_flags = []
    if exclamations >= 2:
        red_flags.append(f"{exclamations} exclamation marks")
    if all_caps_pct > 15:
        red_flags.append(f"{all_caps_pct}% ALL-CAPS words")
    if sensational:
        red_flags.append(f"sensationalist words: {', '.join(sensational[:3])}")
    if questions >= 2:
        red_flags.append("multiple rhetorical questions")

    green_flags = []
    if has_sources:
        green_flags.append("cites sources or data")
    if has_numbers:
        green_flags.append("includes specific numbers/statistics")
    if exclamations == 0:
        green_flags.append("no excessive punctuation")
    if all_caps_pct == 0:
        green_flags.append("no ALL-CAPS sensationalism")

    return {
        'exclamations': exclamations,
        'questions': questions,
        'all_caps_pct': all_caps_pct,
        'sensational_words': sensational,
        'has_sources': has_sources,
        'has_numbers': has_numbers,
        'red_flags': red_flags,
        'green_flags': green_flags,
        'word_count': len(words),
    }

MODEL_LABELS = {
    'logistic_regression': 'Logistic Regression',
    'naive_bayes':         'Naive Bayes',
    'random_forest':       'Random Forest',
    'gradient_boosting':   'Gradient Boosting',
}

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if len(text) < 10:
        return jsonify({'error': 'Text too short — please enter a headline or article excerpt.'}), 400

    processed = preprocess(text)
    signals   = get_signals(text)

    model_predictions = {}
    votes_fake = 0
    votes_real = 0
    ensemble_fake_prob = 0.0

    for key, pipe in models.items():
        pred  = pipe.predict([processed])[0]
        proba = pipe.predict_proba([processed])[0]
        fake_prob = float(proba[1])
        model_predictions[key] = {
            'label':     'Fake' if pred == 1 else 'Real',
            'fake_prob': round(fake_prob * 100, 1),
            'real_prob': round((1 - fake_prob) * 100, 1),
        }
        ensemble_fake_prob += fake_prob
        if pred == 1: votes_fake += 1
        else: votes_real += 1

    n = len(models)
    ensemble_fake_prob = ensemble_fake_prob / n if n else 0.0
    ensemble_label = 'Fake' if ensemble_fake_prob >= 0.5 else 'Real'

    if ensemble_fake_prob >= 0.8:
        confidence_label = 'High confidence'
    elif ensemble_fake_prob >= 0.6 or ensemble_fake_prob <= 0.4:
        confidence_label = 'Moderate confidence'
    else:
        confidence_label = 'Low confidence (borderline)'

    return jsonify({
        'verdict':           ensemble_label,
        'fake_probability':  round(ensemble_fake_prob * 100, 1),
        'real_probability':  round((1 - ensemble_fake_prob) * 100, 1),
        'confidence_label':  confidence_label,
        'votes_fake':        votes_fake,
        'votes_real':        votes_real,
        'model_predictions': model_predictions,
        'signals':           signals,
        'text_preview':      text[:120] + ('…' if len(text) > 120 else ''),
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    result = {}
    for key in models:
        r = metadata.get('model_results', {}).get(MODEL_LABELS[key], {})
        result[key] = {
            'label':    MODEL_LABELS[key],
            'accuracy': r.get('accuracy', 0),
            'roc_auc':  r.get('roc_auc', 0),
            'cv_mean':  r.get('cv_mean', 0),
        }
    return jsonify({'models': result, 'best': metadata.get('best_model', '')})

if __name__ == '__main__':
    print(f"Loaded {len(models)} models: {list(models.keys())}")
    app.run(host='0.0.0.0', port=5000, debug=False)
