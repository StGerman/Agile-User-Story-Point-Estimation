# AGENTS.md - AI Agent Guidelines

This document provides instructions for AI agents (GitHub Copilot, Claude, GPT, Codex, etc.) working on this codebase.

---

## Project Summary

**Agile User Story Point Estimation** is a machine learning research project that predicts story points from user story text using NLP embeddings and regression models.

| Aspect         | Details                                                               |
| -------------- | --------------------------------------------------------------------- |
| **Input**      | User story text (title + description)                                 |
| **Output**     | Story point estimate (numeric)                                        |
| **Dataset**    | 23,313 issues from 16 open-source projects                            |
| **Tech Stack** | Python 3.7, TensorFlow 1.15, scikit-learn, LightGBM, CatBoost, Gensim |

---

## Repository Structure

```text
├── preprocessing.py        # Data preprocessing and TF-IDF vectorization
├── helper.py               # split_data() - project-stratified splitting
├── prepareData.py          # LSTM data preparation
├── mongodbConnector.py     # MongoDB data access
├── Word2VecFeature.py      # Word2Vec feature extraction
├── Doc2VecFeature.py       # Doc2Vec feature extraction
├── RandomForest.py         # Random Forest regressor
├── lightGBM.py             # LightGBM regressor
├── catb.py                 # CatBoost regressor
├── LSTM_regression.py      # End-to-end LSTM model
├── data_csv/               # Preprocessed data
├── dataset/                # Raw CSV files for MongoDB import
├── features/               # Generated embeddings
├── helper/                 # Pickled artifacts
├── models/                 # Trained models
├── docker-compose.yml      # Docker services (app + MongoDB)
├── Dockerfile              # Python 3.7 environment
└── requirements.txt        # Pinned dependencies
```

---

## Critical Rules

### 🔴 MUST Follow

1. **Use project-stratified splits**

   ```python
   from helper import split_data
   x_train, x_test, y_train, y_test = split_data(x, y, ratio=0.2)
   ```

   **NEVER** use `sklearn.model_selection.train_test_split` directly.

2. **Run preprocessing first**

   ```bash
   python preprocessing.py
   ```

   This creates required artifacts in `helper/` before any feature extraction.

3. **Exclude 'project' column from training**

   ```python
   model.fit(x_train.iloc[:, :-1], y_train)  # Last column is 'project'
   ```

4. **Report all three metrics**: MAE, MdAE, MSE

5. **Use Docker for consistent environment**

   ```bash
   docker compose up -d --build
   docker compose exec app python <script.py>
   ```

### 🟡 Should Follow

- Use `--proc` as even number for multiprocessing
- Delete `models/` folder to retrain from scratch
- Use Word2Vec/Doc2Vec features with CatBoost (not TF-IDF)

### 🟢 Conventions

- Text concatenation: `title + ". " + description`
- Feature file naming: `features/{feature_name}_{size}.csv`
- Sparse matrices: `.npz` format (not pickle)

---

## Execution Order

```mermaid
graph TD
    A[1. Start MongoDB] --> B[2. Import CSVs]
    B --> C[3. preprocessing.py]
    C --> D[4. Word2VecFeature.py / Doc2VecFeature.py]
    D --> E[5. Model Training]
    E --> F[6. Evaluation]
```

### Docker Commands (Recommended)

```bash
# Setup
docker compose up -d --build
docker compose exec mongo mongo mydb --eval 'db.createCollection("storypoint")'
docker compose exec mongo bash -lc 'for f in /dataset/*.csv; do mongoimport -d mydb -c storypoint --type CSV --file "$f" --headerline; done'

# Run pipeline
docker compose exec app python preprocessing.py
docker compose exec app python Word2VecFeature.py --proc 8
docker compose exec app python RandomForest.py --size 100 --feature_name word2vec_ave
```

---

## Code Patterns

### Adding a New Model

```python
#!/usr/bin/env python3
"""New regressor model template."""

import pandas as pd
import argparse
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from helper import split_data

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--size', default='100', type=str, help='Embedding size')
parser.add_argument('--feature_name', default='word2vec_ave', type=str, help='Feature name')
args = parser.parse_args()

# Load data
x = pd.read_csv(f"features/{args.feature_name}_{args.size}.csv", index_col=0)
data_csv = pd.read_csv("data_csv/data")
y = data_csv.point

# Split (project-stratified)
x_train, x_test, y_train, y_test = split_data(x, y, ratio=0.2)

# Train (exclude 'project' column)
# model = YourRegressor()
# model.fit(x_train.iloc[:, :-1], y_train)

# Predict
# y_pred = model.predict(x_test.iloc[:, :-1])

# Evaluate
# print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
# print(f"MdAE: {median_absolute_error(y_test, y_pred):.2f}")
# print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
```

### Per-Project Evaluation

```python
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

def evaluate_per_project(x_test, y_pred, y_test):
    """Evaluate model performance per project."""
    results = pd.DataFrame({
        'project': x_test['project'],
        'pred': y_pred,
        'truth': y_test
    })

    return results.groupby('project').apply(
        lambda df: pd.Series({
            'MAE': mean_absolute_error(df.truth, df.pred),
            'MdAE': median_absolute_error(df.truth, df.pred),
            'MSE': mean_squared_error(df.truth, df.pred),
            'count': len(df)
        })
    )
```

---

## Troubleshooting Guide

| Error                                         | Cause                         | Solution                                                 |
| --------------------------------------------- | ----------------------------- | -------------------------------------------------------- |
| `FileNotFoundError: helper/dictionary.pickle` | Preprocessing not run         | Run `python preprocessing.py`                            |
| `std::bad_alloc`                              | Insufficient RAM              | Reduce `--batch_size`, `--embedding_size`, `--rnn_units` |
| CatBoost crashes with TF-IDF                  | Sparse matrix incompatibility | Use Word2Vec/Doc2Vec features                            |
| `ModuleNotFoundError`                         | Missing dependencies          | `pip install -r requirements.txt` or use Docker          |
| MongoDB connection refused                    | MongoDB not running           | `docker compose up -d mongo`                             |
| TensorFlow deprecation warnings               | TF 1.x API                    | Expected, can be ignored                                 |

---

## Environment Configuration

### MongoDB Connection

```python
# Environment variables (set in docker-compose.yml)
MONGO_HOST = "mongo"  # or "localhost" for local
MONGO_PORT = 27017
MONGO_DB = "mydb"
```

### Feature Generation Options

| Feature          | Embedding Sizes | Output File                        |
| ---------------- | --------------- | ---------------------------------- |
| Word2Vec Average | 100, 300        | `features/word2vec_ave_{size}.csv` |
| Doc2Vec          | 100, 300        | `features/doc2vec_{size}.csv`      |
| TF-IDF           | N/A             | `features/tf_idf_matrix.npz`       |

---

## Benchmark Results

| Model                   | MAE      | MdAE | MSE       |
| ----------------------- | -------- | ---- | --------- |
| TF-IDF + Random Forest  | **3.96** | 1.90 | 82.64     |
| TF-IDF + LightGBM       | 4.41     | 2.43 | 84.59     |
| Word2Vec 100 + LightGBM | 4.42     | 2.54 | 82.25     |
| Word2Vec 300 + LightGBM | 4.35     | 2.50 | 80.21     |
| Word2Vec 400 + LightGBM | 4.31     | 2.46 | **78.66** |
| Doc2Vec 100 + LightGBM  | 4.84     | 3.05 | 97.44     |
| LSTM (50 units)         | 3.97     | N/A  | 92.07     |
| LSTM (100 embedding)    | 3.98     | N/A  | 90.51     |

**Reference**: MAE ≈ 4.0 is a competitive baseline.

---

## Do's and Don'ts

### ✅ Do

- Use Docker for reproducible results
- Check `helper/` artifacts exist before feature extraction
- Include `project` column in feature CSVs for stratified splitting
- Report MAE, MdAE, and MSE together
- Use `.iloc[:, :-1]` to exclude project column when training

### ❌ Don't

- Use random train/test splits (causes data leakage)
- Use TF-IDF features with CatBoost
- Modify `helper.split_data()` logic
- Skip `preprocessing.py` before feature extraction
- Assume TensorFlow 2.x compatibility

---

## File Dependencies

```text
preprocessing.py
├── Creates: helper/WordNetLemmatizer.pickle
├── Creates: helper/contraction_map.pickle
├── Creates: helper/dictionary.pickle
├── Creates: helper/corpus_hdf
├── Creates: features/tf_idf_vectorizer.pickle
├── Creates: features/tf_idf_matrix.npz
└── Creates: data_csv/data

Word2VecFeature.py
├── Requires: helper/corpus_hdf
├── Creates: helper/word2vec_{size}.model
└── Creates: features/word2vec_ave_{size}.csv

Doc2VecFeature.py
├── Requires: helper/corpus_hdf
└── Creates: features/doc2vec_{size}.csv

RandomForest.py / lightGBM.py / catb.py
├── Requires: features/{feature_name}_{size}.csv
└── Requires: data_csv/data

LSTM_regression.py
├── Requires: helper/dictionary.pickle
├── Requires: features/tf_idf_vectorizer.pickle
├── Requires: data_csv/data
├── Creates: models/ckpt/
└── Creates: models/pb/
```

---

## Contact & Resources

- **Original Author**: @bking (2018)
- **README**: [README.md](README.md)
- **References**: See README.md for academic references
