"""
Tests for PyTorch training loop.

These tests demonstrate interview-appropriate testing patterns:
- Shape verification
- Sanity checks on loss behavior
- Basic functional tests

Run with: pytest interview_prep/ml/pytorch/test_training_loop.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

# Import from our training module
from model_training_loop import (
    SimpleDataset,
    SimpleClassifier,
    create_dataloaders,
    train_epoch,
    validate,
    predict,
    generate_synthetic_data
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")  # Use CPU for tests


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    return generate_synthetic_data(
        n_samples=100,
        n_features=10,
        n_classes=3,
        random_state=42
    )


@pytest.fixture
def model(device):
    """Create a simple model for testing."""
    return SimpleClassifier(
        input_dim=10,
        hidden_dim=32,
        num_classes=3
    ).to(device)


@pytest.fixture
def dataloaders(sample_data):
    """Create dataloaders for testing."""
    X_train, y_train, X_val, y_val = sample_data
    return create_dataloaders(X_train, y_train, X_val, y_val, batch_size=16)


# =============================================================================
# DATASET TESTS
# =============================================================================

class TestSimpleDataset:
    """Tests for dataset implementation."""
    
    def test_dataset_length(self, sample_data):
        """Dataset reports correct length."""
        X_train, y_train, _, _ = sample_data
        dataset = SimpleDataset(X_train, y_train)
        assert len(dataset) == len(X_train)
    
    def test_dataset_getitem_shapes(self, sample_data):
        """Dataset returns correct shapes."""
        X_train, y_train, _, _ = sample_data
        dataset = SimpleDataset(X_train, y_train)
        
        x, y = dataset[0]
        
        assert x.shape == (10,), f"Expected (10,), got {x.shape}"
        assert y.shape == (), f"Expected scalar, got {y.shape}"
    
    def test_dataset_dtypes(self, sample_data):
        """Dataset returns correct dtypes."""
        X_train, y_train, _, _ = sample_data
        dataset = SimpleDataset(X_train, y_train)
        
        x, y = dataset[0]
        
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64  # LongTensor for classification


# =============================================================================
# MODEL TESTS
# =============================================================================

class TestSimpleClassifier:
    """Tests for model architecture."""
    
    def test_forward_shape(self, model, device):
        """Model produces correct output shape."""
        batch_size = 8
        input_dim = 10
        num_classes = 3
        
        x = torch.randn(batch_size, input_dim).to(device)
        output = model(x)
        
        expected_shape = (batch_size, num_classes)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_model_has_parameters(self, model):
        """Model has learnable parameters."""
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0, "Model should have parameters"
    
    def test_parameters_require_grad(self, model):
        """All parameters should require gradients."""
        for name, param in model.named_parameters():
            assert param.requires_grad, f"Parameter {name} should require grad"


# =============================================================================
# TRAINING TESTS
# =============================================================================

class TestTraining:
    """Tests for training loop functionality."""
    
    def test_train_epoch_returns_metrics(self, model, dataloaders, device):
        """train_epoch returns loss and accuracy."""
        train_loader, _ = dataloaders
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert metrics['loss'] > 0, "Loss should be positive"
        assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be between 0 and 1"
    
    def test_validate_returns_metrics(self, model, dataloaders, device):
        """validate returns loss and accuracy."""
        _, val_loader = dataloaders
        criterion = nn.CrossEntropyLoss()
        
        metrics = validate(model, val_loader, criterion, device)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
    
    def test_loss_decreases_over_epochs(self, model, dataloaders, device):
        """Loss should decrease during training (sanity check)."""
        train_loader, _ = dataloaders
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        # Get initial loss
        initial_metrics = validate(model, train_loader, criterion, device)
        
        # Train for a few epochs
        for _ in range(5):
            train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Get final loss
        final_metrics = validate(model, train_loader, criterion, device)
        
        assert final_metrics['loss'] < initial_metrics['loss'], \
            f"Loss should decrease: {initial_metrics['loss']:.4f} -> {final_metrics['loss']:.4f}"
    
    def test_model_eval_mode_during_validation(self, model, dataloaders, device):
        """Model should be in eval mode during validation."""
        _, val_loader = dataloaders
        criterion = nn.CrossEntropyLoss()
        
        # Force train mode
        model.train()
        
        # Validate (should switch to eval)
        _ = validate(model, val_loader, criterion, device)
        
        # Check mode (note: validate sets to eval internally)
        # After validation, model should be in eval mode
        assert not model.training, "Model should be in eval mode after validation"


# =============================================================================
# INFERENCE TESTS
# =============================================================================

class TestInference:
    """Tests for prediction functionality."""
    
    def test_predict_output_shape(self, model, sample_data, device):
        """predict returns correct shape."""
        X_train, _, _, _ = sample_data
        
        predictions = predict(model, X_train[:10], device)
        
        assert predictions.shape == (10,), f"Expected (10,), got {predictions.shape}"
    
    def test_predict_output_values(self, model, sample_data, device):
        """Predictions are valid class indices."""
        X_train, _, _, _ = sample_data
        num_classes = 3
        
        predictions = predict(model, X_train[:10], device)
        
        assert predictions.min() >= 0, "Predictions should be non-negative"
        assert predictions.max() < num_classes, f"Predictions should be < {num_classes}"
    
    def test_predict_is_deterministic_in_eval(self, model, sample_data, device):
        """Same input should give same output in eval mode."""
        X_train, _, _, _ = sample_data
        x = X_train[:5]
        
        pred1 = predict(model, x, device)
        pred2 = predict(model, x, device)
        
        np.testing.assert_array_equal(pred1, pred2)


# =============================================================================
# GRADIENT TESTS
# =============================================================================

class TestGradients:
    """Tests for gradient flow."""
    
    def test_gradients_flow_to_all_parameters(self, model, device):
        """All parameters should receive gradients after backward pass."""
        x = torch.randn(4, 10).to(device)
        y = torch.LongTensor([0, 1, 2, 1]).to(device)
        
        criterion = nn.CrossEntropyLoss()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient for {name} is None"
            assert not torch.all(param.grad == 0), f"Gradient for {name} is all zeros"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
