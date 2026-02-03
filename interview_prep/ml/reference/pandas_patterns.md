# Pandas Patterns for Quick EDA

## First 2 Minutes: Data Overview

```python
import pandas as pd
import numpy as np

# Load and inspect
df = pd.read_csv("data.csv")

# Essential checks
print(f"Shape: {df.shape}")           # (rows, columns)
print(f"Columns: {df.columns.tolist()}")
df.head()                              # First 5 rows
df.info()                              # Types, non-null counts
df.describe()                          # Numeric statistics
```

## Missing Values

```python
# Count missing per column
df.isnull().sum()

# Percentage missing
(df.isnull().sum() / len(df) * 100).round(2)

# Visualize missing pattern
df.isnull().sum().plot(kind='bar')

# Handle missing
df.dropna()                           # Drop rows with any null
df.dropna(subset=['col1', 'col2'])    # Drop only if these cols null
df['col'].fillna(df['col'].mean())    # Fill with mean
df['col'].fillna(method='ffill')      # Forward fill
df.fillna({'col1': 0, 'col2': 'unknown'})  # Different fills per column
```

## Column Selection and Filtering

```python
# Select columns
df['column']                     # Single column (Series)
df[['col1', 'col2']]            # Multiple columns (DataFrame)
df.select_dtypes(include=['number'])  # Only numeric
df.select_dtypes(include=['object'])  # Only string/object

# Filter rows
df[df['col'] > 0]                         # Condition
df[(df['a'] > 0) & (df['b'] < 10)]       # Multiple conditions (AND)
df[(df['a'] > 0) | (df['b'] < 10)]       # Multiple conditions (OR)
df[df['col'].isin(['a', 'b', 'c'])]      # In list
df[df['col'].str.contains('pattern')]    # String contains
df.query("age > 25 and city == 'NYC'")   # SQL-like syntax
```

## Aggregation and Grouping

```python
# Basic aggregation
df['col'].mean()
df['col'].value_counts()            # Frequency counts
df['col'].nunique()                 # Number of unique values

# GroupBy
df.groupby('category')['value'].mean()
df.groupby('category').agg({
    'value': ['mean', 'std', 'count'],
    'other': 'sum'
})

# Multiple groupby columns
df.groupby(['cat1', 'cat2']).size()

# Pivot table
pd.pivot_table(df, values='value', index='row_cat', 
               columns='col_cat', aggfunc='mean')
```

## Data Transformation

```python
# Apply function
df['new'] = df['col'].apply(lambda x: x * 2)
df['new'] = df.apply(lambda row: row['a'] + row['b'], axis=1)

# Map values
df['category'] = df['category'].map({'old': 'new', 'a': 'b'})

# Replace values
df['col'].replace({'old': 'new'})
df['col'].replace(-1, np.nan)

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], 
                          labels=['child', 'young', 'middle', 'senior'])

# Quantile-based binning
df['quartile'] = pd.qcut(df['value'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

## String Operations

```python
# Access string methods
df['text'].str.lower()
df['text'].str.upper()
df['text'].str.strip()
df['text'].str.replace('old', 'new')
df['text'].str.split(' ')
df['text'].str.len()
df['text'].str.contains('pattern')
df['text'].str.extract(r'(\d+)')  # Regex extract
```

## Date/Time

```python
# Parse dates
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.dayofweek
df['hour'] = df['date'].dt.hour

# Date arithmetic
df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days

# Resample time series
df.set_index('date').resample('D').mean()  # Daily
df.set_index('date').resample('W').sum()   # Weekly
```

## Merging and Joining

```python
# Merge (SQL-like join)
pd.merge(df1, df2, on='key')                    # Inner join
pd.merge(df1, df2, on='key', how='left')        # Left join
pd.merge(df1, df2, left_on='a', right_on='b')   # Different column names

# Concat (stack)
pd.concat([df1, df2])                           # Stack vertically
pd.concat([df1, df2], axis=1)                   # Stack horizontally

# Join on index
df1.join(df2, how='left')
```

## Quick Visualization (for EDA)

```python
import matplotlib.pyplot as plt

# Distribution
df['col'].hist(bins=30)

# Value counts bar chart
df['category'].value_counts().plot(kind='bar')

# Scatter
df.plot.scatter(x='col1', y='col2')

# Correlation heatmap
import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.tight_layout()
```

## Interview-Ready EDA Template

```python
def quick_eda(df: pd.DataFrame) -> None:
    """Run quick EDA on a DataFrame. Call this first in any interview."""
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print("=" * 50)
    
    print("\n--- Column Types ---")
    print(df.dtypes)
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values")
    
    print("\n--- Numeric Summary ---")
    print(df.describe())
    
    print("\n--- Categorical Columns ---")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts().head())

# Usage in interview:
# quick_eda(df)
```

## Common Gotchas

```python
# SettingWithCopyWarning - use .loc
df.loc[df['a'] > 0, 'b'] = 1  # Correct
# df[df['a'] > 0]['b'] = 1    # Warning!

# Chained indexing - avoid
df.loc[0, 'col']              # Get specific cell
df.at[0, 'col']               # Faster for single cell

# Reset index after filtering
df_filtered = df[df['col'] > 0].reset_index(drop=True)

# Copy to avoid modifying original
df_copy = df.copy()
```

## Performance Tips

```python
# Use category dtype for repeated strings
df['category'] = df['category'].astype('category')

# Use query() for complex filters (often faster)
df.query("age > 25 and city == 'NYC'")

# Avoid iterrows - use vectorized operations
# Bad: for idx, row in df.iterrows(): ...
# Good: df['new'] = df['a'] + df['b']

# Read only needed columns
df = pd.read_csv("file.csv", usecols=['col1', 'col2'])
```
