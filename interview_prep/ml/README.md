# ML Interview Prep Kit

## 1. Purpose

This prep kit is designed for **live coding interviews** focused on applied machine learning. It's optimized for:

- **Quick reference** during warm-up before interviews
- **Pattern recall** for common ML tasks
- **Confidence building** through runnable examples

### How to Use Before an Interview

1. **30 min before**: Skim the reference sheets in `reference/`
2. **20 min before**: Run one PyTorch and one HuggingFace example
3. **10 min before**: Review `reference/ml_interview_reference.md` for mental frameworks

---

## 2. Tooling Overview

### PyTorch vs scikit-learn

| Use Case | PyTorch | scikit-learn |
|----------|---------|--------------|
| Deep learning, neural networks | ✅ Primary choice | ❌ |
| Custom architectures | ✅ Full flexibility | ❌ |
| GPU training | ✅ Native support | ❌ |
| Classical ML (trees, SVM, etc.) | ❌ | ✅ Primary choice |
| Quick baselines | ❌ | ✅ Fast iteration |
| Preprocessing pipelines | Use for tensors | ✅ Sklearn pipelines |
| Metrics & evaluation | Basic support | ✅ Comprehensive |

**Interview insight**: Start with sklearn for baselines, move to PyTorch when you need deep learning or custom loss functions.

### Where Hugging Face Fits

- **Pretrained models**: NLP, vision, audio - don't train from scratch
- **Pipelines**: Zero-code inference for common tasks
- **Tokenizers**: Fast, battle-tested text preprocessing
- **Datasets**: Standardized loading for common benchmarks
- **Fine-tuning**: When you need to adapt a foundation model

**Interview insight**: HuggingFace is your shortcut for transformer-based tasks. Know when it's overkill (simple tabular data, classical ML).

### Why Pandas + Quick EDA Matters

- **First 5 minutes matter**: Understanding data shape and quality
- **Catch issues early**: Missing values, type mismatches, class imbalance
- **Build trust**: Show interviewer you're methodical before modeling

---

## 3. How to Run

### Python Version

- Python 3.9+ recommended
- Tested with Python 3.10

### Minimal Setup

```bash
# Option 1: Quick start (install as needed)
pip install torch pandas numpy matplotlib scikit-learn transformers pytest

# Option 2: Virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install torch pandas numpy matplotlib scikit-learn transformers pytest
```

### Running Example Scripts

```bash
# PyTorch examples
python interview_prep/ml/pytorch/tensors_autograd.py
python interview_prep/ml/pytorch/model_training_loop.py

# HuggingFace examples
python interview_prep/ml/huggingface/pipelines_and_inference.py

# Visualization
python interview_prep/ml/visualization/matplotlib_quick_patterns.py
```

### Running Tests

```bash
# Run all tests
pytest interview_prep/ml/ -v

# Run specific test files
pytest interview_prep/ml/pytorch/test_training_loop.py -v
pytest interview_prep/ml/huggingface/test_hf_pipeline.py -v
pytest interview_prep/ml/testing/pytest_examples.py -v
```

---

## 4. Suggested Prep Order (30-90 minutes)

### Quick Warm-up (30 min)

1. **Skim** `reference/ml_interview_reference.md` (10 min)
2. **Run** `pytorch/model_training_loop.py` and trace the flow (10 min)
3. **Run** `huggingface/pipelines_and_inference.py` (5 min)
4. **Review** `pytorch/debugging_and_shapes.md` (5 min)

### Standard Prep (60 min)

1. Everything in Quick Warm-up (30 min)
2. **Read** `reference/python_idioms.md` (10 min)
3. **Run** `pytest interview_prep/ml/` and understand assertions (10 min)
4. **Skim** `reference/pandas_patterns.md` (10 min)

### Deep Prep (90 min)

1. Everything in Standard Prep (60 min)
2. **Read** `huggingface/tokenization_notes.md` (10 min)
3. **Read** `sklearn_notes/when_to_use_sklearn.md` (10 min)
4. **Practice**: Modify one training loop without looking at reference (10 min)

---

## 5. Live Coding Tips

### Narrating Decisions

- **Say why before what**: "I'll use a DataLoader here because we need batching and shuffling..."
- **Acknowledge tradeoffs**: "I'm using CrossEntropyLoss which handles softmax internally, so I won't add softmax to my model output..."
- **Verbalize uncertainty**: "I'm not 100% sure if this is the most efficient approach, but it's correct and clear..."

### Debugging Under Pressure

1. **Print shapes first**: `print(x.shape, y.shape)` before any operation
2. **Check dtypes**: `print(x.dtype)` when you hit type errors
3. **Isolate the problem**: Comment out code until error disappears, then add back
4. **Use assertions**: `assert x.shape[0] == batch_size, f"Expected {batch_size}, got {x.shape[0]}"`

### If Time Runs Short

**Prioritize in order:**

1. **Working code** over perfect code
2. **Correct logic** over optimization
3. **Clear structure** over clever tricks
4. **Verbal explanation** of what you'd improve

**What to skip:**
- Extensive error handling
- Perfect variable names (good is fine)
- Visualization (describe it verbally)
- Hyperparameter tuning

**What to never skip:**
- Data shape verification
- Loss sanity check (is it decreasing?)
- Clear explanation of your approach

---

## File Reference

| File | Purpose |
|------|---------|
| `reference/ml_interview_reference.md` | Core ML workflow and patterns |
| `reference/python_idioms.md` | Python patterns for clean code |
| `reference/pandas_patterns.md` | Quick EDA patterns |
| `pytorch/tensors_autograd.py` | Tensor basics and autograd |
| `pytorch/model_training_loop.py` | Complete training loop |
| `pytorch/debugging_and_shapes.md` | Common debugging patterns |
| `huggingface/pipelines_and_inference.py` | Quick inference examples |
| `huggingface/tokenization_notes.md` | Tokenizer patterns |
| `huggingface/fine_tuning_outline.md` | Fine-tuning workflow |
| `sklearn_notes/when_to_use_sklearn.md` | When sklearn beats PyTorch |
| `testing/testing_strategy.md` | What to test in interviews |
| `testing/pytest_examples.py` | Practical test patterns |
