# PyTorch Debugging and Shape Issues

## The Shape Debugging Workflow

**When something goes wrong, check shapes first.** This is the most common source of bugs.

```python
# Add this liberally when debugging
print(f"x shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
```

## Common Shape Errors and Fixes

### 1. Matrix Multiplication Dimension Mismatch

**Error:**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x10 and 20x5)
```

**Diagnosis:**
```python
print(f"Input shape: {x.shape}")  # (32, 10)
print(f"Weight shape: {linear.weight.shape}")  # (5, 20) -> expects (*, 20)
```

**Fix:** Ensure input features match layer input dimension:
```python
# Either fix the data
x = x.view(32, 20)  # Reshape if data is correct but wrong shape

# Or fix the layer
linear = nn.Linear(10, 5)  # Match input dimension
```

### 2. Batch Dimension Issues

**Error:**
```
RuntimeError: Expected 4-dimensional input for 4-dimensional weight
```

**Diagnosis:** Conv2d expects (batch, channels, height, width)
```python
print(x.shape)  # (3, 224, 224) - missing batch dimension!
```

**Fix:**
```python
x = x.unsqueeze(0)  # Add batch dimension: (1, 3, 224, 224)
```

### 3. Wrong Reduction Dimension

**Error:** Silent bug - wrong output shape

**Diagnosis:**
```python
x = torch.randn(32, 10)  # (batch, features)
wrong = x.mean(dim=0)     # (10,) - averaged over batches!
right = x.mean(dim=1)     # (32,) - averaged over features per sample
```

**Tip:** Always verify with print:
```python
print(f"Before mean: {x.shape}, After mean: {x.mean(dim=1).shape}")
```

### 4. Softmax Dimension

**Error:** Probabilities don't sum to 1 per sample

**Diagnosis:**
```python
x = torch.randn(32, 10)
wrong = torch.softmax(x, dim=0)  # Softmax over batch dimension!
print(wrong.sum(dim=1))  # Not 1.0!
```

**Fix:**
```python
right = torch.softmax(x, dim=1)  # Softmax over class dimension
print(right.sum(dim=1))  # All 1.0
```

### 5. CrossEntropyLoss Input Shape

**Error:**
```
ValueError: Expected target size (32,), got (32, 1)
```

**Diagnosis:**
```python
logits = model(x)  # (32, num_classes)
targets = y        # (32, 1) - wrong!
```

**Fix:**
```python
targets = y.squeeze()  # (32,)
# Or
targets = y.view(-1)   # (32,)
```

### 6. View vs Reshape

**Error:**
```
RuntimeError: view size is not compatible with input tensor's size and stride
```

**Diagnosis:** view() requires contiguous memory

**Fix:**
```python
# Option 1: Make contiguous first
x = x.contiguous().view(new_shape)

# Option 2: Use reshape (handles non-contiguous)
x = x.reshape(new_shape)

# Option 3: Check if contiguous
print(x.is_contiguous())
```

## Device Mismatch Errors

**Error:**
```
RuntimeError: Expected all tensors to be on the same device
```

**Diagnosis:**
```python
print(f"Input device: {x.device}")
print(f"Model device: {next(model.parameters()).device}")
```

**Fix:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
y = y.to(device)
```

## dtype Errors

**Error:**
```
RuntimeError: expected scalar type Float but found Long
```

**Common scenarios:**
```python
# Linear layers expect FloatTensor
x = torch.LongTensor([1, 2, 3])  # Wrong!
x = torch.FloatTensor([1, 2, 3])  # Correct

# CrossEntropyLoss expects LongTensor for targets
y = torch.FloatTensor([0, 1, 2])  # Wrong!
y = torch.LongTensor([0, 1, 2])   # Correct
```

**Conversion:**
```python
x = x.float()  # To FloatTensor
x = x.long()   # To LongTensor
x = x.to(torch.float32)  # Explicit dtype
```

## Gradient Issues

### 1. Gradient is None

**Problem:**
```python
loss.backward()
print(param.grad)  # None
```

**Possible causes:**
- Tensor created without `requires_grad=True`
- Operation broke gradient chain (e.g., `item()`, numpy conversion)
- Parameter not used in forward pass

**Debug:**
```python
print(f"requires_grad: {x.requires_grad}")
print(f"grad_fn: {x.grad_fn}")  # Should not be None
```

### 2. Gradients Accumulating

**Problem:** Loss increasing unexpectedly

**Cause:** Forgot to zero gradients

**Fix:**
```python
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()  # MUST be before forward pass
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### 3. In-place Operation Error

**Error:**
```
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation
```

**Cause:**
```python
x = torch.randn(3, requires_grad=True)
x += 1  # In-place operation on leaf variable!
```

**Fix:**
```python
x = x + 1  # Create new tensor instead
```

## Training Not Converging

### Checklist

1. **Check data**
   ```python
   print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
   print(f"y distribution: {np.bincount(y)}")
   ```

2. **Check loss is decreasing**
   ```python
   if epoch > 0 and loss > prev_loss * 1.1:
       print("Warning: Loss increasing!")
   ```

3. **Check gradients exist and are reasonable**
   ```python
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: grad mean={param.grad.mean():.6f}, std={param.grad.std():.6f}")
   ```

4. **Check model is in correct mode**
   ```python
   print(f"Training mode: {model.training}")  # True during training
   ```

5. **Verify forward pass output**
   ```python
   with torch.no_grad():
       sample_out = model(sample_input)
       print(f"Output range: [{sample_out.min():.2f}, {sample_out.max():.2f}]")
   ```

## Quick Debug Template

```python
def debug_forward(model, x, y, criterion):
    """Run one forward pass with full debugging output."""
    print("=" * 50)
    print(f"Input x: shape={x.shape}, dtype={x.dtype}, device={x.device}")
    print(f"Target y: shape={y.shape}, dtype={y.dtype}, device={y.device}")
    
    # Forward
    model.train()
    output = model(x)
    print(f"Output: shape={output.shape}, range=[{output.min():.2f}, {output.max():.2f}]")
    
    # Loss
    loss = criterion(output, y)
    print(f"Loss: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # Gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Grad {name}: mean={param.grad.mean():.6f}")
        else:
            print(f"Grad {name}: None!")
    
    print("=" * 50)
```

## Common Interview Debugging Scenarios

### "The model outputs NaN"

1. Check for division by zero in custom loss
2. Check for log of zero or negative numbers
3. Lower learning rate
4. Add gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### "Training accuracy is 100% but validation is 50%"

1. Overfitting - add regularization
2. Data leakage - check train/val split
3. Add dropout, reduce model size

### "Loss doesn't decrease at all"

1. Learning rate too high or too low
2. Labels might be shuffled/misaligned with features
3. Model architecture issue (e.g., no gradient flow)
4. Check if shuffle=True in DataLoader for training
