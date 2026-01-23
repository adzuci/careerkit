# When to Use scikit-learn

## The Rule of Thumb

**Start with scikit-learn, move to PyTorch when you need:**
- Deep learning / neural networks
- GPU training
- Custom architectures or loss functions
- Pretrained models (especially for NLP/vision)

## When scikit-learn is the Right Choice

### 1. Classical ML Algorithms

| Task | Algorithm | sklearn Code |
|------|-----------|--------------|
| Classification | Logistic Regression | `LogisticRegression()` |
| Classification | Random Forest | `RandomForestClassifier()` |
| Classification | Gradient Boosting | `GradientBoostingClassifier()` |
| Classification | SVM | `SVC()` |
| Regression | Linear Regression | `LinearRegression()` |
| Regression | Random Forest | `RandomForestRegressor()` |
| Clustering | K-Means | `KMeans()` |
| Dimensionality | PCA | `PCA()` |

### 2. Quick Baselines

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 5-line baseline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

**Interview value:** Shows you can establish baselines quickly before jumping to complex solutions.

### 3. Preprocessing Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Combine preprocessing + model
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

**Interview value:** Shows production-ready thinking with reproducible preprocessing.

### 4. Model Selection and Validation

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# Hyperparameter search
param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

### 5. Metrics and Evaluation

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# Classification
print(classification_report(y_true, y_pred))

# Regression
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}")
print(f"R²: {r2_score(y_true, y_pred):.3f}")
```

## When to NOT Use scikit-learn

### 1. Deep Learning Tasks
- Image classification → PyTorch + pretrained CNN
- NLP tasks → HuggingFace transformers
- Custom neural network architectures

### 2. Large-Scale Data
- Data doesn't fit in memory → Use Spark MLlib or Dask-ML
- Need GPU acceleration → PyTorch

### 3. Production Inference at Scale
- Need optimized inference → ONNX, TensorRT
- Need model serving → TorchServe, TF Serving

## sklearn in a PyTorch Workflow

scikit-learn complements PyTorch well:

```python
# sklearn for preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# sklearn for train/test split
from sklearn.model_selection import train_test_split
X_train, X_val = train_test_split(X_scaled, test_size=0.2)

# sklearn for metrics
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred, average='weighted')

# sklearn for baseline comparison
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
baseline_acc = rf.score(X_test, y_test)
# Compare: "My neural net got 0.92, baseline RF got 0.85"
```

## Common sklearn Patterns

### Pattern 1: Quick EDA to Modeling

```python
# 1. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Baseline
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
print(f"Baseline: {dummy.score(X_test, y_test):.3f}")

# 3. Simple model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f"LogReg: {lr.score(X_test, y_test):.3f}")

# 4. Better model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
print(f"RF: {rf.score(X_test, y_test):.3f}")
```

### Pattern 2: Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: zero mean, unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!

# MinMaxScaler: scale to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
```

### Pattern 3: Handling Imbalanced Classes

```python
# Option 1: Class weights
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')

# Option 2: SMOTE (requires imbalanced-learn)
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

## Interview Talking Points

### "Why would you use sklearn over PyTorch?"

> "For tabular data and classical ML tasks, sklearn is faster to implement, doesn't require GPU, and often performs just as well. I'd use PyTorch for deep learning tasks where I need custom architectures or pretrained models."

### "How do you avoid data leakage?"

> "Always fit preprocessing on training data only, then transform both train and test. That's why sklearn's Pipeline is useful - it handles this automatically."

### "How do you select which algorithm to use?"

> "I start with a simple baseline like Logistic Regression, then try Random Forest for non-linear patterns. If performance isn't good enough, I'll try gradient boosting or neural networks. The key is to start simple and add complexity only when needed."

## Quick sklearn Imports Cheatsheet

```python
# Data splitting
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report

# Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
```
