# Testing Strategy for ML Interviews

## The Interview Testing Philosophy

In a live coding interview, testing serves a different purpose than in production:

| Production Testing | Interview Testing |
|-------------------|-------------------|
| Comprehensive coverage | Sanity checks only |
| Edge cases | Happy path |
| CI/CD integration | Quick feedback |
| Prevent regressions | Demonstrate rigor |

**Goal:** Show engineering maturity without slowing yourself down.

## What to Test in an Interview

### 1. Shape Checks (Always)

Shape mismatches are the #1 bug in ML code. Always verify.

```python
# Before any operation
assert x.shape == (batch_size, features), f"Expected {(batch_size, features)}, got {x.shape}"

# After model forward pass
output = model(x)
assert output.shape == (batch_size, num_classes)
```

**Verbal version:** "Let me print the shape here to make sure the dimensions are correct..."

### 2. Sanity Checks (When Time Permits)

Quick checks that verify basic correctness:

```python
# Loss should be positive
assert loss.item() > 0

# Loss should decrease
assert final_loss < initial_loss, "Loss didn't decrease - something's wrong"

# Probabilities should sum to 1
probs = torch.softmax(logits, dim=1)
assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size))

# Predictions should be valid class indices
assert all(0 <= p < num_classes for p in predictions)
```

### 3. Smoke Tests (For Complex Operations)

Verify code runs without error:

```python
# Quick test that model works
sample_input = torch.randn(1, input_dim)
try:
    output = model(sample_input)
    print(f"Smoke test passed: output shape {output.shape}")
except Exception as e:
    print(f"Smoke test failed: {e}")
```

## What NOT to Test in an Interview

1. **Edge cases** - Focus on the main path
2. **Error handling** - Assume valid inputs
3. **Performance tests** - Not relevant for correctness
4. **Integration tests** - Too slow
5. **100% coverage** - Unnecessary

## Verbal Testing

When you don't have time to write tests, verbalize:

> "If I had more time, I'd add a test to verify that..."

> "I'm checking the shape here to make sure..."

> "I'll print the loss to verify it's decreasing..."

This shows testing awareness without implementation overhead.

## Interview-Appropriate Test Patterns

### Pattern 1: Assert Inline

```python
def train_step(model, batch, optimizer, criterion):
    x, y = batch
    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"
    
    output = model(x)
    assert output.shape[0] == x.shape[0], "Batch size mismatch"
    
    loss = criterion(output, y)
    assert not torch.isnan(loss), "Loss is NaN!"
    
    loss.backward()
    optimizer.step()
    return loss.item()
```

### Pattern 2: Quick Validation Function

```python
def validate_model_output(output, expected_shape, num_classes):
    """Quick validation for model output."""
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    return True

# Usage
output = model(x)
validate_model_output(output, (batch_size, num_classes), num_classes)
```

### Pattern 3: Before/After Comparison

```python
# Verify training actually changes something
initial_params = {n: p.clone() for n, p in model.named_parameters()}

train_step(model, batch, optimizer, criterion)

for name, param in model.named_parameters():
    assert not torch.equal(param, initial_params[name]), f"{name} didn't change"
```

## Pytest Patterns for ML

### Basic Test Structure

```python
import pytest
import torch

class TestModel:
    @pytest.fixture
    def model(self):
        return SimpleModel(input_dim=10, output_dim=3)
    
    def test_forward_shape(self, model):
        x = torch.randn(4, 10)
        output = model(x)
        assert output.shape == (4, 3)
    
    def test_gradients_flow(self, model):
        x = torch.randn(4, 10)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
```

### Fixtures for Expensive Operations

```python
@pytest.fixture(scope="module")
def trained_model():
    """Train once, use in multiple tests."""
    model = train_model(epochs=10)
    return model

def test_accuracy(trained_model):
    acc = evaluate(trained_model, test_data)
    assert acc > 0.7
```

### Parametrized Tests

```python
@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_different_batch_sizes(model, batch_size):
    x = torch.randn(batch_size, 10)
    output = model(x)
    assert output.shape[0] == batch_size
```

## When Tests Demonstrate Value

### Shows You Caught a Bug

```python
# "I added this check because I noticed the shapes were wrong..."
assert x.shape[-1] == self.input_dim, \
    f"Input dim {x.shape[-1]} doesn't match model {self.input_dim}"
```

### Shows Production Thinking

```python
# "In production, I'd add validation like this..."
if not isinstance(x, torch.Tensor):
    x = torch.tensor(x)
```

### Shows Debugging Strategy

```python
# "When I hit an error, I'd add prints like this..."
print(f"Input: shape={x.shape}, dtype={x.dtype}")
print(f"Output: shape={output.shape}, range=[{output.min():.2f}, {output.max():.2f}]")
```

## Quick Testing Checklist

Before calling your solution "done":

- [ ] Shapes match expectations at key points
- [ ] Loss is positive and decreasing
- [ ] Outputs are in valid ranges (e.g., probabilities sum to 1)
- [ ] No NaN or Inf in outputs
- [ ] Code runs without error on sample input

## Test Commands for Interview

```bash
# Run all tests
pytest interview_prep/ml/ -v

# Run specific test file
pytest interview_prep/ml/pytorch/test_training_loop.py -v

# Run and stop on first failure
pytest -x

# Run with print statements visible
pytest -s

# Run only tests matching pattern
pytest -k "test_shape"
```
