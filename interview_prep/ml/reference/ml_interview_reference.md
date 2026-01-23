# ML Interview Reference Sheet

## End-to-End Applied ML Flow

### 1. Data Ingestion
```python
import pandas as pd

# Common patterns
df = pd.read_csv("data.csv")
df = pd.read_json("data.json")
df = pd.read_parquet("data.parquet")  # Preferred for large data

# Quick inspection
df.shape          # (rows, cols)
df.head()         # First 5 rows
df.info()         # Types and nulls
df.describe()     # Statistics
```

### 2. Data Cleaning
```python
# Missing values
df.isnull().sum()                    # Count per column
df.dropna(subset=['critical_col'])   # Drop rows with nulls in specific cols
df['col'].fillna(df['col'].median()) # Fill with median

# Type conversion
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')

# Duplicates
df.drop_duplicates(subset=['id'], keep='first')
```

### 3. Feature Engineering
```python
# Numerical
df['log_value'] = np.log1p(df['value'])  # log(1+x) handles zeros
df['normalized'] = (df['col'] - df['col'].mean()) / df['col'].std()

# Categorical
pd.get_dummies(df, columns=['category'])  # One-hot encoding
df['category'].map({'a': 0, 'b': 1})      # Label encoding

# Text (basic)
df['text_len'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
```

### 4. Train/Val/Test Split
```python
from sklearn.model_selection import train_test_split

# Standard split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify for classification
)

# With validation set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
```

### 5. Modeling

**Classification baseline:**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**PyTorch pattern:**
```python
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch['x'])
        loss = criterion(outputs, batch['y'])
        loss.backward()
        optimizer.step()
```

### 6. Evaluation

**Classification metrics:**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_score(y_true, y_pred)
precision_score(y_true, y_pred, average='weighted')
recall_score(y_true, y_pred, average='weighted')
f1_score(y_true, y_pred, average='weighted')
```

**Regression metrics:**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

---

## Common Interview Task Patterns

### Pattern 1: "Build a classifier for X"

1. Load and inspect data (2 min)
2. Basic cleaning (2 min)
3. Train/test split (1 min)
4. Baseline model (3 min)
5. Evaluate and discuss (2 min)

### Pattern 2: "Preprocess this text data"

1. Check text column quality (1 min)
2. Basic cleaning (lowercase, remove punctuation) (2 min)
3. Tokenization choice discussion (1 min)
4. Implement tokenization (3 min)
5. Show sample output (1 min)

### Pattern 3: "Debug this training loop"

1. Check data shapes at each step
2. Verify loss is decreasing
3. Check for NaN in outputs
4. Verify gradient flow (are gradients non-zero?)

### Pattern 4: "Use a pretrained model for X"

1. Choose appropriate HuggingFace model
2. Use pipeline for quick inference
3. Discuss fine-tuning if more accuracy needed

---

## Tradeoffs and "Why" Explanations

### Why use CrossEntropyLoss over manual softmax + NLLLoss?
- More numerically stable
- Fewer operations = less error accumulation
- Standard practice everyone recognizes

### Why Adam over SGD?
- Adam adapts learning rate per parameter
- Less sensitive to initial learning rate choice
- Good default for most problems
- SGD + momentum can generalize better but needs tuning

### Why batch normalization?
- Stabilizes training
- Allows higher learning rates
- Acts as regularization
- Standard in modern architectures

### Why dropout?
- Prevents overfitting
- Forces redundant representations
- Easy to implement
- Can hurt training time

### Why pretrained models?
- Leverage patterns from massive datasets
- Faster convergence
- Better generalization on small datasets
- State-of-the-art results with less compute

---

## "What Would You Improve With More Time?"

### Quick wins (5-15 min more):
- Add learning rate scheduler
- Implement early stopping
- Add data augmentation
- Try different optimizer

### Medium investments (30-60 min more):
- Hyperparameter search
- Cross-validation
- Ensemble methods
- Feature engineering iteration

### Larger improvements (hours+):
- Architecture search
- Custom loss functions
- More sophisticated preprocessing
- Collect more data

---

## Key Numbers to Remember

| Metric | Good Starting Point |
|--------|---------------------|
| Learning rate | 1e-3 (Adam), 1e-2 (SGD) |
| Batch size | 32 (small GPU), 64-128 (typical) |
| Epochs | 10-20 for quick experiments |
| Dropout rate | 0.1-0.5 |
| Train/test split | 80/20 or 70/15/15 |

---

## Red Flags to Catch Early

1. **Loss not decreasing**: Check learning rate, data loading, model output
2. **Loss goes to NaN**: Learning rate too high, numerical instability
3. **100% train accuracy**: Overfitting, need regularization
4. **0% improvement**: Model not learning, check gradients
5. **Validation worse than random**: Data leakage or label issues
