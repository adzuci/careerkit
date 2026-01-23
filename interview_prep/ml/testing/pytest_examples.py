"""
Pytest Examples for ML Interviews

This file demonstrates interview-appropriate testing patterns.
Focus is on clarity, speed, and demonstrating engineering rigor.

Run with: pytest interview_prep/ml/testing/pytest_examples.py -v
"""

import pytest
import numpy as np


# =============================================================================
# BASIC PYTEST PATTERNS
# =============================================================================

class TestBasicPatterns:
    """
    Basic pytest patterns you should know.
    """
    
    def test_simple_assertion(self):
        """Most basic test - just assert something."""
        result = 2 + 2
        assert result == 4
    
    def test_with_message(self):
        """Add message for debugging when test fails."""
        result = compute_something()
        assert result > 0, f"Expected positive, got {result}"
    
    def test_floating_point(self):
        """Use pytest.approx for floating point comparison."""
        result = 0.1 + 0.2
        assert result == pytest.approx(0.3)
        
        # With tolerance
        assert 0.999 == pytest.approx(1.0, rel=1e-2)
    
    def test_exception_raised(self):
        """Verify that code raises expected exception."""
        with pytest.raises(ValueError):
            raise ValueError("Expected error")
    
    def test_exception_message(self):
        """Check exception message content."""
        with pytest.raises(ValueError, match="invalid"):
            raise ValueError("This is an invalid input")


def compute_something():
    return 42


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    """
    Fixture provides reusable test data.
    
    Fixtures are functions that pytest calls before tests.
    Tests receive fixture output as parameter.
    """
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, size=100)
    return X, y


@pytest.fixture
def simple_model():
    """Fixture for a simple sklearn model."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000)


class TestWithFixtures:
    """Tests using fixtures."""
    
    def test_data_shape(self, sample_data):
        """Access fixture as parameter."""
        X, y = sample_data
        assert X.shape == (100, 10)
        assert y.shape == (100,)
    
    def test_model_fit(self, simple_model, sample_data):
        """Use multiple fixtures."""
        X, y = sample_data
        simple_model.fit(X, y)
        assert hasattr(simple_model, 'coef_')


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================

class TestParametrized:
    """
    Parametrized tests run same logic with different inputs.
    Great for testing multiple cases efficiently.
    """
    
    @pytest.mark.parametrize("input,expected", [
        (0, 0),
        (1, 1),
        (2, 4),
        (3, 9),
        (-2, 4),
    ])
    def test_square(self, input, expected):
        """Test squaring with multiple inputs."""
        assert input ** 2 == expected
    
    @pytest.mark.parametrize("batch_size", [1, 8, 16, 32])
    def test_batch_processing(self, batch_size):
        """Test different batch sizes."""
        data = np.random.randn(batch_size, 10)
        result = process_batch(data)
        assert result.shape[0] == batch_size


def process_batch(data):
    return data * 2


# =============================================================================
# ML-SPECIFIC TEST PATTERNS
# =============================================================================

class TestMLPatterns:
    """
    ML-specific testing patterns for interviews.
    """
    
    def test_shape_preservation(self):
        """Verify operations preserve expected shapes."""
        X = np.random.randn(32, 10)
        
        # Some transformation
        X_transformed = X @ np.random.randn(10, 5)
        
        assert X_transformed.shape == (32, 5), \
            f"Expected (32, 5), got {X_transformed.shape}"
    
    def test_normalization_bounds(self):
        """Verify normalized data is in expected range."""
        X = np.random.randn(100, 5) * 100  # Large values
        
        # Min-max normalize
        X_norm = (X - X.min()) / (X.max() - X.min())
        
        assert X_norm.min() >= 0, "Min should be >= 0"
        assert X_norm.max() <= 1, "Max should be <= 1"
    
    def test_probabilities_sum_to_one(self):
        """Verify probability outputs are valid."""
        logits = np.random.randn(10, 3)
        
        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        # Each row should sum to 1
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0), f"Probabilities don't sum to 1: {row_sums}"
    
    def test_model_improves(self, sample_data):
        """Verify model performance improves with training."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        X, y = sample_data
        
        # Untrained model (random predictions)
        random_preds = np.random.randint(0, 3, size=len(y))
        random_acc = accuracy_score(y, random_preds)
        
        # Trained model
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        trained_preds = model.predict(X)
        trained_acc = accuracy_score(y, trained_preds)
        
        # Trained should be better than random
        assert trained_acc > random_acc, \
            f"Trained ({trained_acc:.3f}) should beat random ({random_acc:.3f})"
    
    def test_predictions_are_valid_classes(self, sample_data):
        """Verify predictions are valid class indices."""
        from sklearn.linear_model import LogisticRegression
        
        X, y = sample_data
        num_classes = len(np.unique(y))
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        predictions = model.predict(X)
        
        # All predictions should be valid class indices
        assert all(0 <= p < num_classes for p in predictions), \
            "Predictions contain invalid class indices"


# =============================================================================
# SANITY CHECKS
# =============================================================================

class TestSanityChecks:
    """
    Quick sanity checks that catch common issues.
    These are the tests worth writing in an interview.
    """
    
    def test_no_nan_in_data(self, sample_data):
        """Data should not contain NaN."""
        X, y = sample_data
        assert not np.isnan(X).any(), "X contains NaN"
        assert not np.isnan(y).any(), "y contains NaN"
    
    def test_no_inf_in_data(self, sample_data):
        """Data should not contain infinity."""
        X, y = sample_data
        assert not np.isinf(X).any(), "X contains infinity"
    
    def test_labels_are_consecutive(self, sample_data):
        """Class labels should be 0, 1, 2, ... n-1."""
        _, y = sample_data
        unique_labels = np.unique(y)
        expected = np.arange(len(unique_labels))
        
        assert np.array_equal(unique_labels, expected), \
            f"Labels not consecutive: {unique_labels}"
    
    def test_no_data_leakage(self, sample_data):
        """Train and test sets should not overlap."""
        from sklearn.model_selection import train_test_split
        
        X, y = sample_data
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2)
        
        # Check no identical rows (simplified check)
        # In practice, you'd use indices
        train_set = set(map(tuple, X_train))
        test_set = set(map(tuple, X_test))
        
        overlap = train_set & test_set
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping samples"


# =============================================================================
# PYTEST MARKS
# =============================================================================

class TestWithMarks:
    """
    Pytest marks for controlling test execution.
    """
    
    @pytest.mark.skip(reason="Demonstration of skip")
    def test_skipped(self):
        """This test is skipped."""
        assert False
    
    @pytest.mark.skipif(
        not _has_torch(),
        reason="PyTorch not installed"
    )
    def test_requires_torch(self):
        """Only runs if PyTorch is available."""
        import torch
        x = torch.randn(3, 3)
        assert x.shape == (3, 3)
    
    @pytest.mark.slow
    def test_slow_operation(self):
        """
        Mark slow tests so they can be excluded:
        pytest -m "not slow"
        """
        import time
        time.sleep(0.1)
        assert True


def _has_torch():
    try:
        import torch
        return True
    except ImportError:
        return False


# =============================================================================
# NUMPY TESTING UTILITIES
# =============================================================================

class TestNumpyUtilities:
    """
    numpy.testing provides useful comparison functions.
    """
    
    def test_array_equal(self):
        """Test exact array equality."""
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        np.testing.assert_array_equal(a, b)
    
    def test_array_almost_equal(self):
        """Test approximate array equality."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0001, 2.0001, 3.0001])
        np.testing.assert_array_almost_equal(a, b, decimal=3)
    
    def test_allclose(self):
        """Test all elements are close."""
        a = np.array([1e-10, 2e-10])
        b = np.array([1.1e-10, 2.1e-10])
        assert np.allclose(a, b, rtol=0.2)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
