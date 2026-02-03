# Python Idioms for ML Interviews

## List Comprehensions

```python
# Basic
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(10) if x % 2 == 0]

# Nested (flattening)
flat = [item for sublist in nested for item in sublist]

# Dict comprehension
word_lengths = {word: len(word) for word in words}
```

## Unpacking

```python
# Tuple unpacking
a, b = 1, 2
a, b = b, a  # Swap

# Extended unpacking
first, *rest = [1, 2, 3, 4]  # first=1, rest=[2,3,4]
*start, last = [1, 2, 3, 4]  # start=[1,2,3], last=4

# In function calls
def func(a, b, c):
    return a + b + c

args = [1, 2, 3]
func(*args)

kwargs = {'a': 1, 'b': 2, 'c': 3}
func(**kwargs)
```

## Enumerate and Zip

```python
# Enumerate for index + value
for i, value in enumerate(items):
    print(f"{i}: {value}")

# Zip for parallel iteration
for x, y in zip(list1, list2):
    print(x, y)

# Zip with unequal lengths
from itertools import zip_longest
for x, y in zip_longest(short, long, fillvalue=0):
    print(x, y)
```

## Dictionary Patterns

```python
# Get with default
value = d.get('key', default_value)

# setdefault for accumulating
d.setdefault('key', []).append(item)

# defaultdict alternative
from collections import defaultdict
d = defaultdict(list)
d['key'].append(item)

# Counter for frequencies
from collections import Counter
counts = Counter(items)
most_common = counts.most_common(5)

# Dictionary merging (3.9+)
merged = dict1 | dict2
```

## F-strings

```python
# Basic
name = "tensor"
print(f"Shape of {name}: {tensor.shape}")

# With formatting
pi = 3.14159
print(f"Pi: {pi:.2f}")  # Pi: 3.14

# With expressions
print(f"Sum: {a + b}")

# Debug format (3.8+)
x = 42
print(f"{x=}")  # x=42
```

## Context Managers

```python
# File handling
with open('file.txt', 'r') as f:
    content = f.read()

# Multiple contexts
with open('in.txt') as fin, open('out.txt', 'w') as fout:
    fout.write(fin.read())

# PyTorch no_grad
with torch.no_grad():
    predictions = model(inputs)
```

## Lambda and Functional

```python
# Lambda
squared = lambda x: x**2

# Map
results = list(map(lambda x: x**2, items))

# Filter
evens = list(filter(lambda x: x % 2 == 0, items))

# Sorted with key
sorted_items = sorted(items, key=lambda x: x['score'], reverse=True)

# Reduce (use sparingly)
from functools import reduce
total = reduce(lambda a, b: a + b, items)
```

## Type Hints (Signal Professionalism)

```python
from typing import List, Dict, Optional, Tuple, Union

def process_batch(
    data: List[Dict[str, float]],
    batch_size: int = 32,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Process a batch of data."""
    pass

# Optional = Union[X, None]
def find_item(items: List[str], target: str) -> Optional[int]:
    """Return index or None if not found."""
    pass
```

## Exception Handling

```python
# Basic try/except
try:
    result = risky_operation()
except ValueError as e:
    print(f"Value error: {e}")
    result = default_value

# Multiple exceptions
try:
    result = operation()
except (ValueError, TypeError) as e:
    handle_error(e)

# Finally for cleanup
try:
    resource = acquire()
    use(resource)
finally:
    release(resource)

# Raise with context
try:
    operation()
except Exception as e:
    raise RuntimeError("Operation failed") from e
```

## Class Patterns

```python
# Dataclass (clean data containers)
from dataclasses import dataclass

@dataclass
class Config:
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 10

config = Config(learning_rate=1e-4)

# Property decorator
class Model:
    def __init__(self):
        self._parameters = []

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self._parameters)
```

## Useful One-Liners

```python
# Check all/any
all(x > 0 for x in values)  # All positive?
any(x < 0 for x in values)  # Any negative?

# Flatten list
flat = [item for sublist in nested for item in sublist]
# Or: sum(nested, [])

# Get unique while preserving order
unique = list(dict.fromkeys(items))

# Chunk a list
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Transpose list of lists
transposed = list(zip(*matrix))
```

## NumPy-Specific Idioms

```python
import numpy as np

# Vectorized operations (avoid loops)
result = np.sqrt(arr**2 + 1)  # Not: [np.sqrt(x**2 + 1) for x in arr]

# Boolean indexing
mask = arr > 0
positive = arr[mask]

# Where for conditional
result = np.where(condition, true_val, false_val)

# Broadcasting
a = np.array([[1], [2], [3]])  # (3, 1)
b = np.array([1, 2, 3])         # (3,)
c = a + b                        # (3, 3) - broadcasts

# Axis operations
arr.sum(axis=0)  # Sum over rows (collapse first dimension)
arr.sum(axis=1)  # Sum over columns
```

## Interview-Friendly Patterns

```python
# Early return for clarity
def process(x):
    if x is None:
        return default
    if not is_valid(x):
        return error_value
    # Main logic here
    return result

# Guard clauses
def calculate(data):
    assert len(data) > 0, "Data cannot be empty"
    assert all(x >= 0 for x in data), "All values must be non-negative"
    # Proceed with calculation

# Descriptive variable names
# Bad: x, temp, data2
# Good: batch_predictions, normalized_features, validation_loss
```
