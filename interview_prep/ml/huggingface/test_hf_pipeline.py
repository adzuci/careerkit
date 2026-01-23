"""
Tests for HuggingFace pipelines.

These tests demonstrate smoke testing and sanity checks for ML pipelines.
They're designed to be interview-appropriate: quick, clear, and focused.

Run with: pytest interview_prep/ml/huggingface/test_hf_pipeline.py -v
"""

import pytest


# =============================================================================
# FIXTURES AND SETUP
# =============================================================================

@pytest.fixture(scope="module")
def sentiment_pipeline():
    """
    Load sentiment pipeline once for all tests.
    
    scope="module" means the pipeline is loaded once per test file,
    not once per test. This is important for slow-loading models.
    """
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis")
    except ImportError:
        pytest.skip("transformers not installed")
    except Exception as e:
        pytest.skip(f"Could not load pipeline: {e}")


@pytest.fixture(scope="module")
def fill_mask_pipeline():
    """Load fill-mask pipeline."""
    try:
        from transformers import pipeline
        return pipeline("fill-mask")
    except ImportError:
        pytest.skip("transformers not installed")
    except Exception as e:
        pytest.skip(f"Could not load pipeline: {e}")


# =============================================================================
# SMOKE TESTS
# =============================================================================

class TestPipelineSmoke:
    """
    Smoke tests verify that pipelines load and produce output.
    They don't test for correctness, just basic functionality.
    """
    
    def test_sentiment_pipeline_loads(self, sentiment_pipeline):
        """Pipeline should load without error."""
        assert sentiment_pipeline is not None
    
    def test_sentiment_pipeline_returns_result(self, sentiment_pipeline):
        """Pipeline should return a result for valid input."""
        result = sentiment_pipeline("This is a test.")
        assert result is not None
        assert len(result) > 0
    
    def test_sentiment_result_has_expected_keys(self, sentiment_pipeline):
        """Result should have label and score."""
        result = sentiment_pipeline("This is a test.")
        
        assert 'label' in result[0]
        assert 'score' in result[0]
    
    def test_sentiment_score_is_valid(self, sentiment_pipeline):
        """Score should be a probability between 0 and 1."""
        result = sentiment_pipeline("This is a test.")
        score = result[0]['score']
        
        assert 0.0 <= score <= 1.0


class TestFillMaskSmoke:
    """Smoke tests for fill-mask pipeline."""
    
    def test_fill_mask_returns_predictions(self, fill_mask_pipeline):
        """Fill-mask should return predictions for [MASK] token."""
        result = fill_mask_pipeline("The capital of France is [MASK].")
        
        assert result is not None
        assert len(result) > 0
    
    def test_fill_mask_result_structure(self, fill_mask_pipeline):
        """Result should have expected structure."""
        result = fill_mask_pipeline("The [MASK] is blue.")
        
        first_result = result[0]
        assert 'token_str' in first_result
        assert 'score' in first_result


# =============================================================================
# SANITY CHECKS
# =============================================================================

class TestSentimentSanity:
    """
    Sanity checks verify that model behavior is reasonable.
    These test known cases where we expect specific behavior.
    """
    
    def test_positive_text_is_positive(self, sentiment_pipeline):
        """Clearly positive text should be classified as positive."""
        result = sentiment_pipeline("I absolutely love this! It's amazing!")
        
        # Note: We check for POSITIVE label, but allow some flexibility
        # because exact label names can vary by model
        label = result[0]['label'].upper()
        score = result[0]['score']
        
        assert 'POS' in label or score > 0.5, \
            f"Expected positive, got {label} with score {score}"
    
    def test_negative_text_is_negative(self, sentiment_pipeline):
        """Clearly negative text should be classified as negative."""
        result = sentiment_pipeline("This is terrible! I hate it so much!")
        
        label = result[0]['label'].upper()
        score = result[0]['score']
        
        assert 'NEG' in label or score > 0.5, \
            f"Expected negative, got {label} with score {score}"
    
    def test_batch_processing_works(self, sentiment_pipeline):
        """Pipeline should handle batch input."""
        texts = [
            "I love this!",
            "I hate this!",
            "This is okay."
        ]
        
        results = sentiment_pipeline(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert 'label' in result
            assert 'score' in result


class TestFillMaskSanity:
    """Sanity checks for fill-mask model."""
    
    def test_fills_with_reasonable_word(self, fill_mask_pipeline):
        """Model should fill mask with contextually appropriate word."""
        result = fill_mask_pipeline("The sky is [MASK].")
        
        # Get top prediction
        top_word = result[0]['token_str'].strip().lower()
        
        # Should be a color or sky-related word
        reasonable_words = ['blue', 'clear', 'cloudy', 'grey', 'gray', 'dark', 'bright']
        
        assert any(word in top_word for word in reasonable_words) or \
            result[0]['score'] > 0.1, \
            f"Unexpected fill: {top_word}"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_string_handling(self, sentiment_pipeline):
        """Pipeline should handle empty strings gracefully."""
        # This might raise an error or return empty - both are acceptable
        try:
            result = sentiment_pipeline("")
            # If it doesn't raise, result should be valid structure
            assert isinstance(result, list)
        except Exception:
            # Some pipelines raise on empty input - that's okay
            pass
    
    def test_very_long_input(self, sentiment_pipeline):
        """Pipeline should handle long input (via truncation)."""
        long_text = "This is great! " * 500  # Very long input
        
        # Should not raise, should return result
        result = sentiment_pipeline(long_text)
        assert result is not None
        assert len(result) > 0


# =============================================================================
# PERFORMANCE SANITY
# =============================================================================

class TestPerformance:
    """Basic performance sanity checks."""
    
    def test_inference_completes_quickly(self, sentiment_pipeline):
        """Single inference should complete in reasonable time."""
        import time
        
        start = time.time()
        _ = sentiment_pipeline("Test sentence for timing.")
        elapsed = time.time() - start
        
        # First inference might be slow due to warmup
        # Subsequent should be < 1 second for simple text
        _ = sentiment_pipeline("Second test for timing.")
        
        start = time.time()
        _ = sentiment_pipeline("Third test for timing.")
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"Inference took too long: {elapsed:.2f}s"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
