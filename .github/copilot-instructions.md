# Agile User Story Point Estimation - Copilot Instructions

## Project Overview

Machine learning research project for estimating Agile story points from user story text using NLP and regression models. The system predicts effort (story points) from issue title and description text.

**Dataset**: 23,313 issues from 16 open-source projects across 9 repositories (Apache, Atlassian, Moodle, Spring, etc.)

**Goal**: End-to-end prediction: `User Story Text → Story Point Estimate`

---

## Quick Reference

### Environment Setup (Docker - Recommended)
```bash
docker compose up -d --build
docker compose exec mongo mongo mydb --eval 'db.createCollection("storypoint")'
docker compose exec mongo bash -lc 'for f in /dataset/*.csv; do mongoimport -d mydb -c storypoint --type CSV --file "$f" --headerline; done'
```

### Full Pipeline Execution
```bash
docker compose exec app python preprocessing.py
docker compose exec app python Word2VecFeature.py --proc 8
docker compose exec app python RandomForest.py --size 100 --feature_name word2vec_ave
```

---

## Architecture

### Data Pipeline Flow
```
MongoDB (mydb.storypoint)
    ↓
preprocessing.py → helper/*.pickle, features/tf_idf_matrix.npz, data_csv/data
    ↓
Word2VecFeature.py / Doc2VecFeature.py → features/{name}_{size}.csv
    ↓
RandomForest.py / lightGBM.py / catb.py → Evaluation metrics
    ↓
LSTM_regression.py → models/ckpt/, models/pb/
```

### Directory Structure
| Directory | Purpose |
|-----------|---------|
| `data_csv/` | Preprocessed CSV data (`data`) |
| `dataset/` | Raw CSV files for MongoDB import |
| `features/` | Generated embeddings and TF-IDF matrix |
| `helper/` | Pickled artifacts (dictionary, vectorizer, corpus_hdf) |
| `models/` | Trained model checkpoints and SavedModel |

### Core Files
| File | Purpose |
|------|---------|
| `preprocessing.py` | Text cleaning, TF-IDF vectorization, creates all `helper/` artifacts |
| `helper.py` | `split_data()` - project-stratified train/test split (80/20) |
| `prepareData.py` | LSTM data preparation with sequence padding |
| `mongodbConnector.py` | MongoDB interface for data retrieval |
| `Word2VecFeature.py` | Word2Vec embedding extraction (parallel) |
| `Doc2VecFeature.py` | Doc2Vec embedding extraction (parallel) |
| `RandomForest.py` | Random Forest regressor training |
| `lightGBM.py` | LightGBM regressor training |
| `catb.py` | CatBoost regressor training |
| `LSTM_regression.py` | End-to-end LSTM model |

---

## Critical Conventions

### Data Handling Rules
1. **ALWAYS use project-stratified splits** via `helper.split_data()` - NEVER use random `train_test_split`
2. **Text concatenation pattern**: `title + ". " + description`
3. **Feature files format**: `features/{feature_name}_{size}.csv` with `project` column preserved
4. **Sparse matrix format**: TF-IDF saved as `.npz` (scipy sparse), not pickle

### Model Training Rules
1. **Multiprocessing**: `--proc` must be an **even number**
2. **Retraining**: Delete `models/` folder entirely (no incremental training)
3. **CatBoost limitation**: Cannot handle sparse matrices - use only Word2Vec/Doc2Vec features

### Evaluation Metrics
Always report all three metrics:
- **MAE** (Mean Absolute Error) - primary metric
- **MdAE** (Median Absolute Error)
- **MSE** (Mean Squared Error)

**Baseline**: TF-IDF + Random Forest achieves MAE=3.96

---

## Command Reference

### Feature Extraction
```bash
# Word2Vec (generates features/word2vec_ave_{size}.csv)
python Word2VecFeature.py --proc 8

# Doc2Vec (generates features/doc2vec_{size}.csv)
python Doc2VecFeature.py --proc 8
```

### Model Training
```bash
# Tree-based models
python RandomForest.py --size 100 --feature_name word2vec_ave
python lightGBM.py --size 100 --feature_name word2vec_ave
python catb.py --size 100 --feature_name word2vec_ave

# LSTM end-to-end
python LSTM_regression.py --rnn_layers 2 --rnn_units 50 --embedding_size 100 --batch_size 100
```

### Available Arguments
| Script | Arguments |
|--------|-----------|
| Feature extraction | `--proc <int>` (even number, default: 8) |
| Tree models | `--size <int>`, `--feature_name <word2vec_ave\|doc2vec>` |
| LSTM | `--batch_size`, `--embedding_size`, `--rnn_layers`, `--rnn_units` |

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `std::bad_alloc` in LSTM | Insufficient RAM | Reduce `batch_size`, `embedding_size`, or `rnn_units` |
| CatBoost fails with TF-IDF | Sparse matrix incompatibility | Use only Word2Vec/Doc2Vec features |
| Feature extraction fails | Missing `preprocessing.py` artifacts | Run `preprocessing.py` first |
| Import errors | Wrong Python environment | Use Docker or install from `requirements.txt` |
| TensorFlow warnings | TF 1.x deprecated APIs | Expected behavior, ignore |

---

## Code Patterns

### Adding a New Regressor
```python
from helper import split_data
import pandas as pd

# Load features
x = pd.read_csv("features/word2vec_ave_100.csv", index_col=0)
data_csv = pd.read_csv("data_csv/data")
y = data_csv.point

# Project-stratified split (REQUIRED)
x_train, x_test, y_train, y_test = split_data(x, y, ratio=0.2)

# Train (exclude 'project' column)
model.fit(x_train.iloc[:, :-1], y_train)
y_pred = model.predict(x_test.iloc[:, :-1])

# Evaluate
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MdAE: {median_absolute_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
```

### Per-Project Evaluation
```python
tmp = pd.DataFrame({'project': x_test['project'], 'pred': y_pred, 'truth': y_test})
result = tmp.groupby('project').apply(
    lambda df: pd.Series({
        'MAE': mean_absolute_error(df.truth, df.pred),
        'MdAE': median_absolute_error(df.truth, df.pred),
        'MSE': mean_squared_error(df.truth, df.pred)
    })
)
```

---

## Technical Notes

- **Python version**: 3.7 (TensorFlow 1.15 compatibility)
- **TensorFlow**: Uses 1.x eager execution API (`tf.enable_eager_execution()`)
- **No Fibonacci rounding**: Dataset uses mixed scales across projects
- **HDF5 storage**: Corpus stored in `helper/corpus_hdf` for Word2Vec/Doc2Vec
