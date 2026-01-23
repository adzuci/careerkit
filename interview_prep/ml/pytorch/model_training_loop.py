"""
PyTorch Training Loop Template

This is the canonical training loop pattern you should be able to write from memory.
It's designed for clarity and correctness, which is what matters in interviews.

Usage:
    python interview_prep/ml/pytorch/model_training_loop.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import numpy as np


# =============================================================================
# 1. DATASET
# =============================================================================

class SimpleDataset(Dataset):
    """
    Custom Dataset template.
    
    Key methods:
        __len__: Return dataset size
        __getitem__: Return single sample by index
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)  # LongTensor for classification
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_dataset = SimpleDataset(X_train, y_train)
    val_dataset = SimpleDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
    )
    
    return train_loader, val_loader


# =============================================================================
# 2. MODEL
# =============================================================================

class SimpleClassifier(nn.Module):
    """Simple feedforward classifier for demonstration."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# =============================================================================
# 3. TRAINING LOOP
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Key steps:
        1. Set model to training mode
        2. Zero gradients
        3. Forward pass
        4. Compute loss
        5. Backward pass
        6. Update weights
    """
    model.train()  # IMPORTANT: Enable dropout, batchnorm training mode
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (X, y) in enumerate(dataloader):
        # Move to device
        X, y = X.to(device), y.to(device)
        
        # Zero gradients (IMPORTANT: do this before forward pass)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate the model.
    
    Key differences from training:
        1. model.eval() mode
        2. torch.no_grad() context
        3. No gradient computation or weight updates
    """
    model.eval()  # IMPORTANT: Disable dropout, use running stats for batchnorm
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # IMPORTANT: No gradient computation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Full training loop with validation.
    
    Returns history dict for plotting/analysis.
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
    
    return history


# =============================================================================
# 4. INFERENCE
# =============================================================================

def predict(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device
) -> np.ndarray:
    """Make predictions on new data."""
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        _, predicted = outputs.max(1)
    
    return predicted.cpu().numpy()


def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device
) -> np.ndarray:
    """Get probability predictions."""
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    return probabilities.cpu().numpy()


# =============================================================================
# 5. DEMO: RUN COMPLETE TRAINING
# =============================================================================

def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification data for demo."""
    np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Create linearly separable classes with some noise
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5)
    # Use n_classes to determine the number of bins; preserve current behavior
    # when n_classes == 3 by keeping the fixed thresholds at -1 and 1.
    if n_classes == 3:
        bins = [-1, 1]
    else:
        y_min, y_max = y.min(), y.max()
        # n_classes classes require n_classes - 1 bin edges
        bins = np.linspace(y_min, y_max, num=n_classes - 1).astype(np.float32)
    y = np.digitize(y, bins=bins)
    
    # Train/val split
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    return X_train, y_train, X_val, y_val


def main():
    """Run complete training demo."""
    print("=" * 60)
    print("PyTorch Training Loop Demo")
    print("=" * 60)
    
    # Configuration
    config = {
        'input_dim': 20,
        'hidden_dim': 64,
        'num_classes': 3,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'num_epochs': 10
    }
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    X_train, y_train, X_val, y_val = generate_synthetic_data(
        n_features=config['input_dim'],
        n_classes=config['num_classes']
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=config['batch_size']
    )
    
    # Model
    model = SimpleClassifier(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes']
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Train
    print("\nTraining:")
    print("-" * 60)
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config['num_epochs']
    )
    
    # Final evaluation
    print("-" * 60)
    print(f"Final Val Accuracy: {history['val_acc'][-1]:.4f}")
    
    # Verify loss decreased (sanity check)
    loss_decreased = history['train_loss'][-1] < history['train_loss'][0]
    print(f"Training loss decreased: {loss_decreased}")
    
    # Inference example
    print("\nInference example:")
    sample = X_val[:5]
    predictions = predict(model, sample, device)
    probabilities = predict_proba(model, sample, device)
    print(f"Predictions: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    return history


if __name__ == "__main__":
    history = main()
