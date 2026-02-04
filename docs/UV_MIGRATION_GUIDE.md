# UV Migration Guide

## Overview

This document describes the migration from `requirements.txt` to modern `pyproject.toml` + `uv` package management for the Agile User Story Point Estimation project.

## Changes Made

### 1. Created `pyproject.toml`
- **Modern PEP 621 compliant project configuration**
- **Python constraint**: `>=3.9,<3.10` (upgraded from 3.7 due to Apple Silicon compatibility)
- **TensorFlow upgrade**: `2.8.4` (from 1.15.5) for Python 3.9 compatibility
- **All dependencies migrated** from requirements.txt with exact version pins
- **Build system configuration** using setuptools backend

### 2. Created `.python-version`
- Automatic Python 3.9 detection for uv and other tools
- Ensures consistent Python version across environments

### 3. Version Updates Required

| Package | Old Version | New Version | Reason |
|---------|-------------|-------------|---------|
| Python | >=3.7,<3.8 | >=3.9,<3.10 | Apple Silicon availability |
| TensorFlow | 1.15.5 | 2.8.4 | Python 3.9 compatibility |

## Known Issues

### Certificate/Network Issues
During testing, encountered PyPI certificate validation errors:
```
error: invalid peer certificate: UnknownIssuer
```

**Workarounds:**
1. Use `--native-tls` flag: `uv sync --native-tls`
2. Configure corporate proxy/certificates if in enterprise environment
3. Fallback to pip install from pyproject.toml: `pip install -e .`

### TensorFlow Migration Impact
**IMPORTANT**: Upgrading from TensorFlow 1.15 → 2.8 requires code changes:

#### Required Code Updates:
```python
# OLD (TensorFlow 1.x)
import tensorflow as tf
tf.enable_eager_execution()
session = tf.Session()

# NEW (TensorFlow 2.x)  
import tensorflow as tf
# Eager execution is default in TF 2.x
# No sessions needed - functions run eagerly
```

#### Model Loading/Saving:
```python
# OLD
tf.saved_model.simple_save(session, ...)
saver = tf.train.Saver()

# NEW  
model.save('path/to/model')
tf.saved_model.save(model, 'path/to/model')
```

## Usage Instructions

### With uv (Recommended)
```bash
# Install dependencies
uv sync --native-tls

# Run scripts in uv environment
uv run python preprocessing.py
uv run python RandomForest.py --feature_name tf_idf --size 100

# Add new dependencies
uv add "new-package==1.0.0"

# Export for legacy tools
uv export --format requirements-txt > requirements_new.txt
```

### Fallback with pip
```bash
# Create virtual environment
python3.9 -m venv .venv
source .venv/bin/activate

# Install from pyproject.toml
pip install -e .

# Or from exported requirements
uv export --format requirements-txt | pip install -r /dev/stdin
```

### Docker Integration
Update `Dockerfile` to use uv:
```dockerfile
FROM python:3.9-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies  
RUN uv sync --frozen --no-cache

# Set uv environment as default
ENV PATH="/app/.venv/bin:$PATH"
```

## Benefits of Migration

1. **Faster installs**: uv is ~10x faster than pip
2. **Better resolver**: Handles complex dependency conflicts 
3. **Reproducible builds**: uv.lock ensures exact versions
4. **Modern tooling**: PEP 621 compliant, future-proof
5. **Built-in venv management**: No manual environment handling

## Validation Steps

1. **Test core pipeline**:
   ```bash
   uv run python preprocessing.py
   uv run python RandomForest.py --feature_name tf_idf --size 100
   ```

2. **Validate TensorFlow 2.x compatibility** in `LSTM_regression.py`
3. **Run full evaluation pipeline** to ensure no regressions
4. **Generate new performance baseline** with TF 2.x

## Rollback Plan

If issues arise, rollback is simple:
1. Keep existing `requirements.txt` (unchanged)
2. Use original Python 3.7 venv: `python3.7 -m venv .venv_old`  
3. Install old dependencies: `pip install -r requirements.txt`
4. Revert to TensorFlow 1.15 workflow

## Next Steps

1. **Resolve certificate issues** for full uv adoption
2. **Update LSTM code** for TensorFlow 2.x compatibility  
3. **Generate uv.lock** once network issues resolved
4. **Update CI/CD** to use uv for faster builds
5. **Document TF 2.x migration** for other team members

---

*Migration completed: 2026-02-03*  
*Status: Partial - pyproject.toml created, network issues prevent full uv adoption*