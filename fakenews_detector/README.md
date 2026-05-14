# TruthLens — Fake News Detector

A machine learning project that detects fake news using NLP and ensemble classification.

## Project Structure

```
fakenews_detector/
├── data/
│   └── news_dataset.csv       # 1,620 labeled samples (balanced 50/50)
├── models/
│   ├── best_model.pkl          # Best performing model (Logistic Regression)
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   └── metadata.json           # Training metrics & config
├── templates/
│   └── index.html              # Frontend UI
├── generate_dataset.py         # Dataset generation script
├── train_model.py              # ML training pipeline
├── app.py                      # Flask API server
└── README.md
```

## Dataset

Modeled after the **WELFake** and **LIAR** dataset structures.
- **1,620 samples** total (810 real, 810 fake) — perfectly balanced
- **Label encoding:** `0 = Real`, `1 = Fake`
- **Features used:** raw text → TF-IDF unigrams + bigrams

Real news patterns follow journalistic conventions (citations, statistics, measured tone).  
Fake news patterns include sensationalism, ALL-CAPS, excessive punctuation, conspiracy language.

## ML Pipeline

### Features
- **TF-IDF** vectorization (unigrams + bigrams, 25,000 features, sublinear TF scaling)
- **Linguistic signals** (post-prediction): exclamation count, ALL-CAPS ratio, sensationalism score, source citation detection

### Models Trained
| Model               | Accuracy | ROC-AUC | 5-Fold CV    |
|---------------------|----------|---------|--------------|
| Logistic Regression | 100.0%   | 100.0%  | 99.8% ± 0.15%|
| Naive Bayes         | 100.0%   | 100.0%  | 99.8% ± 0.15%|
| Random Forest       | 100.0%   | 100.0%  | 99.75% ± 0.12%|
| Gradient Boosting   | 99.7%    | 99.8%   | 99.6% ± 0.31%|

### Ensemble Strategy
All 4 models vote. Final verdict uses the **average fake probability** across all models.

## Running the App

```bash
# 1. Install dependencies
pip install flask scikit-learn pandas numpy joblib

# 2. Generate dataset (already done)
python generate_dataset.py

# 3. Train models (already done — models/ folder exists)
python train_model.py

# 4. Start the web server
python app.py
# → Open http://localhost:5000
```

## API Endpoints

### `POST /api/analyze`
```json
{ "text": "Your headline or article text here" }
```
Returns verdict, probabilities, individual model predictions, and linguistic signals.

### `GET /api/models`
Returns model names, accuracy, ROC-AUC scores, and the best model name.

## How It Works

1. Input text is preprocessed (URLs removed, lowercased, stop words filtered)
2. TF-IDF transforms text to a feature vector
3. All 4 models independently predict real/fake + probability
4. Ensemble averages fake probability; final verdict if ≥ 50% → Fake
5. Linguistic signals (flags) are computed separately as an explainability layer

## Notes

- High accuracy is expected on synthetic data; real-world performance on unseen news datasets (LIAR, FakeNewsNet) would be lower (~70–85%) due to distribution shift.
- To improve real-world performance: swap in the actual WELFake CSV (kaggle.com/saurabhshahane/fake-news-classification) and retrain with `train_model.py`.
