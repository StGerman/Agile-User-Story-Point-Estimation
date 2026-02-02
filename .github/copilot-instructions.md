# Agile User Story Point Estimation - AI Agent Instructions

## Project Overview
Machine learning research project for estimating Agile story points from user story text using NLP and regression models. Combines text embedding techniques (Word2Vec, Doc2Vec, TF-IDF, LSTM) with tree-based regressors (Random Forest, LightGBM, CatBoost).

## Architecture

### Data Pipeline Flow
1. **MongoDB → CSV**: Raw data (23,313 issues from 16 projects) stored in MongoDB `mydb.storypoint`
2. **Preprocessing** ([preprocessing.py](../preprocessing.py)): Text cleaning, TF-IDF vectorization, creates pickled artifacts in `helper/`
3. **Feature Extraction**: Parallel processing to generate embeddings in `features/`
4. **Model Training**: Separate scripts for each model type
5. **Evaluation**: Metrics (MAE, MdAE, MSE) compared across models

### Critical Files
- [helper.py](../helper.py): `split_data()` - project-stratified train/test split (80/20) - DO NOT use random splits
- [prepareData.py](../prepareData.py): LSTM data preparation, sequence padding
- [mongodbConnector.py](../mongodbConnector.py): MongoDB interface, retrieves title + description concatenated text

## Running Experiments

### Sequential Workflow (MUST follow this order)
```bash
# 1. Setup MongoDB (passwordless localhost:27017)
mongo mydb --eval 'db.createCollection("storypoint")'
bash importCSV.sh  # Imports all CSV files from dataset/

# 2. Install dependencies
bash install_library.sh  # Only NLTK, LightGBM, CatBoost listed
# Missing from script: gensim, tensorflow, pandas, pymongo, scikit-learn

# 3. Preprocessing (generates helper/ and features/ artifacts)
python preprocessing.py

# 4. Feature extraction (parallel processing)
python Word2VecFeature.py --proc 8  # Multi-embedding sizes: [10,50,100,300]
python Doc2VecFeature.py --proc 8

# 5. Model training
python RandomForest.py --size 100 --feature_name word2vec_ave
python lightGBM.py --size 100 --feature_name word2vec_ave
python catb.py --size 100 --feature_name word2vec_ave

# 6. LSTM end-to-end
python LSTM_regression.py --rnn_layers 2 --rnn_units 50 --embedding_size 100
```

## Key Conventions

### Data Handling
- **Stratified split**: Always split by `project` column to prevent data leakage between projects
- **Feature files**: CSV format `features/{feature_name}_{size}.csv` with `project` column preserved
- **Text concatenation**: `title + ". " + description` pattern throughout
- **Sparse matrix**: TF-IDF saved as `.npz`, not pickle (memory efficiency)

### Multiprocessing Pattern
Feature extraction scripts use `Pool(proc)` where `proc` must be **even number**. Scripts check for existing feature files before regenerating.

### Model Persistence
- Models saved to `models/` directory
- LSTM checkpoints: `models/ckpt/`, serving format: `models/pb/`
- To retrain: **delete `models/` folder** (no incremental training)

### Evaluation Metrics
Report all three: Mean Absolute Error (MAE), Median Absolute Error (MdAE), Mean Squared Error (MSE). Reference baseline: TF-IDF + Random Forest achieves MAE=3.96.

## Common Pitfalls

1. **CatBoost + sparse data**: CatBoost models cannot handle scipy sparse matrices from TF-IDF - only dense features (Word2Vec/Doc2Vec)
2. **LSTM memory**: `std::bad_alloc` errors indicate insufficient RAM - reduce `batch_size`, `embedding_size`, or `rnn_units`
3. **Dependency order**: `preprocessing.py` must run before any feature extraction (creates dictionaries/vectorizers)
4. **TensorFlow version**: Uses deprecated eager execution API (`tf.enable_eager_execution()`) - assumes TensorFlow 1.x patterns

## Project-Specific Details
- No Fibonacci rounding on predictions (dataset projects use mixed scales)
- HDF5 storage for corpus (`helper/corpus_hdf`) used by Word2Vec/Doc2Vec
- Pickle files store: lemmatizer, contraction map, dictionary, TF-IDF vectorizer
