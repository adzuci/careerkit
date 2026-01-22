"""
PyTorch Tensors and Autograd Fundamentals

This file covers the core building blocks you need for any PyTorch interview task.
Run this file to see examples and verify your PyTorch installation works.

Usage:
    python interview_prep/ml/pytorch/tensors_autograd.py
"""

import torch
import torch.nn as nn


def tensor_basics():
    """Core tensor operations you'll use constantly."""
    print("=" * 60)
    print("TENSOR BASICS")
    print("=" * 60)
    
    # Creation
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"From list: {x}, shape: {x.shape}, dtype: {x.dtype}")
    
    zeros = torch.zeros(3, 4)
    ones = torch.ones(2, 3)
    rand = torch.rand(2, 3)  # Uniform [0, 1)
    randn = torch.randn(2, 3)  # Normal distribution
    
    print(f"Zeros shape: {zeros.shape}")
    print(f"Randn:\n{randn}")
    
    # From NumPy (common in interviews)
    import numpy as np
    np_arr = np.array([1, 2, 3])
    tensor_from_np = torch.from_numpy(np_arr)  # Shares memory!
    tensor_copy = torch.tensor(np_arr)  # Creates copy
    print(f"From numpy: {tensor_from_np}")
    
    # Shape operations
    x = torch.randn(2, 3, 4)
    print(f"\nOriginal shape: {x.shape}")
    print(f"Reshaped to (6, 4): {x.reshape(6, 4).shape}")
    print(f"Reshaped with -1: {x.reshape(-1, 4).shape}")  # -1 infers dimension
    print(f"View (same as reshape): {x.view(2, 12).shape}")
    print(f"Transpose: {x.transpose(0, 1).shape}")  # Swap dims 0 and 1
    print(f"Permute: {x.permute(2, 0, 1).shape}")  # Reorder all dims
    
    # Squeeze and unsqueeze
    x = torch.randn(1, 3, 1)
    print(f"\nOriginal: {x.shape}")
    print(f"Squeeze (remove 1s): {x.squeeze().shape}")
    print(f"Unsqueeze at 0: {x.squeeze().unsqueeze(0).shape}")


def device_management():
    """GPU/CPU device handling - essential for interviews."""
    print("\n" + "=" * 60)
    print("DEVICE MANAGEMENT")
    print("=" * 60)
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create tensor on device
    x = torch.randn(3, 3, device=device)
    print(f"Tensor device: {x.device}")
    
    # Move tensor between devices
    x_cpu = x.cpu()
    print(f"Moved to CPU: {x_cpu.device}")
    
    # Common pattern: move model and data to same device
    # model = model.to(device)
    # batch = batch.to(device)


def autograd_basics():
    """Automatic differentiation - the heart of training."""
    print("\n" + "=" * 60)
    print("AUTOGRAD BASICS")
    print("=" * 60)
    
    # requires_grad enables gradient tracking
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = x ** 2  # y = x^2
    z = y.sum()  # z = x1^2 + x2^2
    
    print(f"x: {x}")
    print(f"y = x^2: {y}")
    print(f"z = sum(y): {z}")
    
    # Backward pass computes gradients
    z.backward()
    print(f"dz/dx = 2x: {x.grad}")  # Should be [4.0, 6.0]
    
    # IMPORTANT: gradients accumulate!
    x.grad.zero_()  # Reset gradients
    
    # Detach from computation graph
    x_detached = x.detach()
    print(f"Detached requires_grad: {x_detached.requires_grad}")
    
    # No-grad context for inference
    with torch.no_grad():
        y = x ** 2
        print(f"In no_grad, y.requires_grad: {y.requires_grad}")


def common_operations():
    """Operations you'll use in almost every interview."""
    print("\n" + "=" * 60)
    print("COMMON OPERATIONS")
    print("=" * 60)
    
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    
    # Element-wise operations
    print("Element-wise add:", (x + y).shape)
    print("Element-wise mul:", (x * y).shape)
    
    # Matrix multiplication
    a = torch.randn(2, 3)
    b = torch.randn(3, 4)
    print(f"matmul: {torch.matmul(a, b).shape}")  # (2, 4)
    print(f"@ operator: {(a @ b).shape}")  # Same thing
    
    # Batch matrix multiplication
    batch_a = torch.randn(5, 2, 3)  # 5 matrices of 2x3
    batch_b = torch.randn(5, 3, 4)  # 5 matrices of 3x4
    print(f"Batch matmul: {torch.bmm(batch_a, batch_b).shape}")  # (5, 2, 4)
    
    # Reductions
    x = torch.randn(3, 4)
    print(f"\nReductions on shape {x.shape}:")
    print(f"  sum all: {x.sum().shape}")
    print(f"  sum dim 0: {x.sum(dim=0).shape}")  # (4,)
    print(f"  sum dim 1: {x.sum(dim=1).shape}")  # (3,)
    print(f"  mean: {x.mean():.4f}")
    print(f"  max: {x.max():.4f}")
    print(f"  argmax: {x.argmax()}")
    
    # Concatenation
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    print(f"\nConcat dim 0: {torch.cat([a, b], dim=0).shape}")  # (4, 3)
    print(f"Concat dim 1: {torch.cat([a, b], dim=1).shape}")  # (2, 6)
    print(f"Stack (new dim): {torch.stack([a, b], dim=0).shape}")  # (2, 2, 3)


def neural_network_building_blocks():
    """Common layers you'll use in models."""
    print("\n" + "=" * 60)
    print("NEURAL NETWORK BUILDING BLOCKS")
    print("=" * 60)
    
    batch_size = 4
    
    # Linear layer
    linear = nn.Linear(in_features=10, out_features=5)
    x = torch.randn(batch_size, 10)
    out = linear(x)
    print(f"Linear: {x.shape} -> {out.shape}")
    
    # Activation functions
    print("\nActivation functions:")
    x = torch.randn(2, 3)
    print(f"  ReLU: {torch.relu(x)}")
    print(f"  Sigmoid: {torch.sigmoid(x)}")
    print(f"  Tanh: {torch.tanh(x)}")
    print(f"  Softmax: {torch.softmax(x, dim=1)}")  # Dim 1 for batch
    
    # Dropout
    dropout = nn.Dropout(p=0.5)
    x = torch.ones(5)
    print(f"\nDropout (training mode): {dropout(x)}")
    dropout.eval()  # Switch to eval mode
    print(f"Dropout (eval mode): {dropout(x)}")
    
    # BatchNorm
    bn = nn.BatchNorm1d(num_features=3)
    x = torch.randn(batch_size, 3)
    print(f"\nBatchNorm: {x.shape} -> {bn(x).shape}")
    
    # Embedding (for categorical/text)
    vocab_size = 100
    embed_dim = 16
    embedding = nn.Embedding(vocab_size, embed_dim)
    indices = torch.LongTensor([0, 5, 10])
    print(f"Embedding: indices {indices.shape} -> {embedding(indices).shape}")


def loss_functions():
    """Common loss functions and when to use them."""
    print("\n" + "=" * 60)
    print("LOSS FUNCTIONS")
    print("=" * 60)
    
    batch_size = 4
    num_classes = 3
    
    # Cross-entropy for classification
    # NOTE: CrossEntropyLoss takes raw logits (no softmax!)
    logits = torch.randn(batch_size, num_classes)
    targets = torch.LongTensor([0, 1, 2, 1])  # Class indices
    
    ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(logits, targets)
    print(f"CrossEntropyLoss: {loss:.4f}")
    
    # Binary cross-entropy (for binary classification)
    predictions = torch.sigmoid(torch.randn(batch_size))
    targets = torch.FloatTensor([0, 1, 1, 0])
    
    bce_loss = nn.BCELoss()
    loss = bce_loss(predictions, targets)
    print(f"BCELoss: {loss:.4f}")
    
    # BCEWithLogitsLoss (more stable - takes raw logits)
    logits = torch.randn(batch_size)
    bce_logits = nn.BCEWithLogitsLoss()
    loss = bce_logits(logits, targets)
    print(f"BCEWithLogitsLoss: {loss:.4f}")
    
    # MSE for regression
    predictions = torch.randn(batch_size)
    targets = torch.randn(batch_size)
    
    mse_loss = nn.MSELoss()
    loss = mse_loss(predictions, targets)
    print(f"MSELoss: {loss:.4f}")
    
    # L1 loss (MAE)
    l1_loss = nn.L1Loss()
    loss = l1_loss(predictions, targets)
    print(f"L1Loss (MAE): {loss:.4f}")


def model_patterns():
    """Common model definition patterns."""
    print("\n" + "=" * 60)
    print("MODEL PATTERNS")
    print("=" * 60)
    
    # Pattern 1: nn.Sequential (simple, linear flow)
    model_seq = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 3)
    )
    x = torch.randn(4, 10)
    print(f"Sequential output: {model_seq(x).shape}")
    
    # Pattern 2: nn.Module subclass (flexible, recommended)
    class CustomModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    model = CustomModel(input_dim=10, hidden_dim=32, output_dim=3)
    print(f"Custom model output: {model(x).shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params} total, {trainable} trainable")


if __name__ == "__main__":
    tensor_basics()
    device_management()
    autograd_basics()
    common_operations()
    neural_network_building_blocks()
    loss_functions()
    model_patterns()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
