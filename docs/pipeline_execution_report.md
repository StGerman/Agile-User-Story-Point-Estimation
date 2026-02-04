# Agile User Story Point Estimation - Pipeline Execution Report

**Generated:** 2026-02-03 00:33:04

## Executive Summary

Successfully executed the core ML pipeline with the following achievements:
- MongoDB setup and data import (23,313 records)
- Text preprocessing and TF-IDF vectorization
- Generated comprehensive feature matrices
- Pipeline ready for model training (blocked by dependency issues)

## Pipeline Status

### ✅ Completed Steps

1. **Environment Setup**
   - Created Python virtual environment (.venv)
   - Installed core dependencies: nltk, pandas, pymongo, scikit-learn
   - Set up Docker Compose for MongoDB service

2. **Data Import & Processing**
   - MongoDB service: Successfully started via Docker
   - Data import: 23,313 documents imported to mydb.storypoint
   - Field mapping: project, title, user_story, point → MongoDB structure

3. **Preprocessing Pipeline**
   - Text cleaning and concatenation (title + user_story)
   - TF-IDF vectorization: 7.3MB sparse matrix
   - Dictionary creation: 89232 vocabulary terms
   - Corpus serialization: Converted from HDF5 to pickle format
   - Feature artifacts saved to features/ and helper/ directories

### ❌ Blocked Steps

1. **Advanced Feature Extraction**
   - Word2Vec/Doc2Vec: gensim build failed (missing native dependencies)
   - CatBoost: C++ compilation failed
   - TensorFlow/LSTM: No compatible wheels for Python 3.14.2

2. **Model Training**
   - RandomForest: Training timeout on large TF-IDF matrix (>23313 samples)
   - LightGBM: Missing OpenMP library (libomp.dylib)

## Data Overview

**Dataset Characteristics:**
- Total samples: 23313
- Projects: 17
- Story point range: 1 - 100
- Average story points: 6.22

**Project Distribution:**

           Count  Mean_Points  Std_Points
project
APSTUD       829         8.02        5.95
BAM          521         2.42        2.14
CLOV         384         4.59        6.55
DM          4667         9.57       16.60
DURACLOUD    666         2.13        2.03
GHS           68         4.88        3.52
JSW          284         4.32        3.51
MDL         1166        15.54       21.65
MESOS       1680         3.09        2.42
MULE         889         5.08        3.50

(Showing top 10 projects)

## Technical Artifacts Generated

**Helper Files:**
- dictionary.pickle: 1.4MB
- corpus.pickle: 16.1MB
- contraction_map.pickle: 0.0MB
- corpus_hdf: 16.9MB
- WordNetLemmatizer.pickle: 0.0MB

**Feature Files:**
- tf_idf_matrix.npz: 7.3MB
- tf_idf_vectorizer.pickle: 2.1MB
- word2vec_ave.csv: 45.3MB
- preprocessing_text.csv: 17.3MB

## Dependency Issues Encountered

### 1. Native Compilation Failures
- **gensim**: Cython/C++ compilation errors
- **catboost**: Missing C++ build tools
- **tensorflow**: No Python 3.14.2 wheels available
- **pytables**: Missing HDF5 development headers

### 2. Runtime Library Issues
- **lightgbm**: Missing libomp.dylib (OpenMP runtime)

### 3. Workarounds Applied
- Replaced HDF5 format with pickle serialization
- Updated mongodbConnector for CSV field structure
- Created manual project-stratified data splitting

## Recommendations

### Immediate Actions
1. **Install OpenMP**: ==> Fetching downloads for: libomp
==> Pouring libomp--21.1.8.arm64_tahoe.bottle.tar.gz
==> Caveats
libomp is keg-only, which means it was not symlinked into /opt/homebrew,
because it can override GCC headers and result in broken builds.

For compilers to find libomp you may need to set:
  export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
  export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
==> Summary
🍺  /opt/homebrew/Cellar/libomp/21.1.8: 9 files, 1.8MB
==> Running `brew cleanup libomp`...
Disable this behaviour by setting `HOMEBREW_NO_INSTALL_CLEANUP=1`.
Hide these hints with `HOMEBREW_NO_ENV_HINTS=1` (see `man brew`). to fix LightGBM
2. **Use older Python**: Python 3.9-3.11 for better package compatibility
3. **Alternative approach**: Use Docker container with pre-built environment

### Alternative Pipeline

1. Use existing TF-IDF features with simplified models
2. Focus on statistical analysis rather than complex ML models
3. Generate comparative reports using available baselines

### Model Baseline Available

The README indicates TF-IDF + Random Forest achieves **MAE=3.96** as baseline.
Current pipeline is positioned to reproduce/validate these results.

---
*Report generated from partial pipeline execution*
