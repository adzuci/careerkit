"""
HuggingFace Pipelines and Inference

Pipelines are the fastest way to get inference working in an interview.
They abstract away tokenization, model loading, and post-processing.

Usage:
    python interview_prep/ml/huggingface/pipelines_and_inference.py
"""

def text_classification_example():
    """
    Sentiment analysis / text classification pipeline.
    
    Good for: sentiment, spam detection, topic classification
    """
    print("=" * 60)
    print("TEXT CLASSIFICATION")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        
        # Load pipeline (downloads model on first run)
        classifier = pipeline("sentiment-analysis")
        
        # Single prediction
        result = classifier("I love using HuggingFace for ML interviews!")
        print(f"Single: {result}")
        
        # Batch prediction
        texts = [
            "This is amazing!",
            "This is terrible.",
            "The weather is okay today."
        ]
        results = classifier(texts)
        for text, res in zip(texts, results):
            print(f"  '{text[:30]}...' -> {res['label']} ({res['score']:.3f})")
        
    except ImportError:
        print("transformers not installed. Run: pip install transformers")
    except Exception as e:
        print(f"Pipeline example skipped: {e}")


def named_entity_recognition_example():
    """
    NER pipeline - extract entities from text.
    
    Good for: information extraction, data parsing
    """
    print("\n" + "=" * 60)
    print("NAMED ENTITY RECOGNITION")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        
        ner = pipeline("ner", aggregation_strategy="simple")
        
        text = "Apple is headquartered in Cupertino, California."
        entities = ner(text)
        
        print(f"Text: {text}")
        print("Entities:")
        for ent in entities:
            print(f"  {ent['word']}: {ent['entity_group']} ({ent['score']:.3f})")
    
    except ImportError:
        print("transformers not installed")
    except Exception as e:
        print(f"NER example skipped: {e}")


def question_answering_example():
    """
    Extractive QA - find answer span in context.
    
    Good for: document search, FAQ systems
    """
    print("\n" + "=" * 60)
    print("QUESTION ANSWERING")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        
        qa = pipeline("question-answering")
        
        context = """
        The Transformer architecture was introduced in the paper 
        "Attention Is All You Need" by Vaswani et al. in 2017.
        It revolutionized NLP by replacing recurrent layers with 
        self-attention mechanisms.
        """
        
        questions = [
            "When was the Transformer introduced?",
            "Who wrote the Transformer paper?",
            "What did the Transformer replace?"
        ]
        
        print(f"Context: {context.strip()[:80]}...")
        print("\nQ&A:")
        for q in questions:
            result = qa(question=q, context=context)
            print(f"  Q: {q}")
            print(f"  A: {result['answer']} (score: {result['score']:.3f})")
    
    except ImportError:
        print("transformers not installed")
    except Exception as e:
        print(f"QA example skipped: {e}")


def text_generation_example():
    """
    Text generation / completion.
    
    Good for: autocomplete, creative writing, code generation
    """
    print("\n" + "=" * 60)
    print("TEXT GENERATION")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        
        # Using a small model for speed
        generator = pipeline("text-generation", model="gpt2")
        
        prompt = "Machine learning is"
        
        result = generator(
            prompt,
            max_length=50,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )
        
        print(f"Prompt: {prompt}")
        print(f"Generated: {result[0]['generated_text']}")
    
    except ImportError:
        print("transformers not installed")
    except Exception as e:
        print(f"Generation example skipped: {e}")


def zero_shot_classification_example():
    """
    Classify text into arbitrary categories without training.
    
    Good for: flexible classification, prototyping
    """
    print("\n" + "=" * 60)
    print("ZERO-SHOT CLASSIFICATION")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        
        classifier = pipeline("zero-shot-classification")
        
        text = "I need to fix a bug in my Python code"
        candidate_labels = ["technology", "sports", "cooking", "politics"]
        
        result = classifier(text, candidate_labels)
        
        print(f"Text: {text}")
        print("Scores:")
        for label, score in zip(result['labels'], result['scores']):
            print(f"  {label}: {score:.3f}")
    
    except ImportError:
        print("transformers not installed")
    except Exception as e:
        print(f"Zero-shot example skipped: {e}")


def summarization_example():
    """
    Text summarization pipeline.
    
    Good for: document summarization, TL;DR
    """
    print("\n" + "=" * 60)
    print("SUMMARIZATION")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        
        summarizer = pipeline("summarization")
        
        long_text = """
        The Amazon rainforest, also known as Amazonia, is a moist broadleaf 
        tropical rainforest in the Amazon biome that covers most of the 
        Amazon basin of South America. This basin encompasses 7,000,000 km2, 
        of which 5,500,000 km2 are covered by the rainforest. This region 
        includes territory belonging to nine nations and 3,344 formally 
        acknowledged indigenous territories. The majority of the forest 
        is contained within Brazil, with 60% of the rainforest, followed 
        by Peru with 13%, Colombia with 10%, and with minor amounts in 
        Bolivia, Ecuador, French Guiana, Guyana, Suriname, and Venezuela.
        """
        
        result = summarizer(long_text, max_length=50, min_length=20)
        
        print(f"Original ({len(long_text)} chars): {long_text[:100]}...")
        print(f"Summary: {result[0]['summary_text']}")
    
    except ImportError:
        print("transformers not installed")
    except Exception as e:
        print(f"Summarization example skipped: {e}")


def fill_mask_example():
    """
    Masked language modeling - predict missing words.
    
    Good for: understanding model embeddings, data augmentation
    """
    print("\n" + "=" * 60)
    print("FILL-MASK (MLM)")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        
        unmasker = pipeline("fill-mask")
        
        text = "Machine learning is a branch of [MASK] intelligence."
        
        results = unmasker(text)
        
        print(f"Text: {text}")
        print("Top predictions:")
        for res in results[:3]:
            print(f"  {res['token_str']}: {res['score']:.3f}")
    
    except ImportError:
        print("transformers not installed")
    except Exception as e:
        print(f"Fill-mask example skipped: {e}")


# =============================================================================
# MANUAL INFERENCE (when you need more control)
# =============================================================================

def manual_inference_example():
    """
    Manual tokenization and model inference.
    
    Use when you need:
    - Custom preprocessing
    - Access to raw logits
    - Batch processing with specific tokenization
    """
    print("\n" + "=" * 60)
    print("MANUAL INFERENCE")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        # Load model and tokenizer
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Tokenize
        texts = ["I love this!", "This is terrible."]
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"  # Return PyTorch tensors
        )
        
        print(f"Input keys: {inputs.keys()}")
        print(f"input_ids shape: {inputs['input_ids'].shape}")
        print(f"attention_mask shape: {inputs['attention_mask'].shape}")
        
        # Inference
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.softmax(logits, dim=1)
        
        print(f"\nLogits shape: {logits.shape}")
        print(f"Predictions:\n{predictions}")
        
        # Get class labels
        labels = ["NEGATIVE", "POSITIVE"]
        for text, pred in zip(texts, predictions):
            label_idx = pred.argmax().item()
            confidence = pred[label_idx].item()
            print(f"'{text}' -> {labels[label_idx]} ({confidence:.3f})")
    
    except ImportError:
        print("transformers or torch not installed")
    except Exception as e:
        print(f"Manual inference example skipped: {e}")


# =============================================================================
# PIPELINE SELECTION GUIDE
# =============================================================================

def print_pipeline_guide():
    """Print quick reference for pipeline selection."""
    print("\n" + "=" * 60)
    print("PIPELINE SELECTION GUIDE")
    print("=" * 60)
    
    guide = """
    Task                    -> Pipeline Name
    ─────────────────────────────────────────────────────────
    Sentiment analysis      -> "sentiment-analysis" or "text-classification"
    Named entities          -> "ner" or "token-classification"
    Question answering      -> "question-answering"
    Text generation         -> "text-generation"
    Summarization           -> "summarization"
    Translation             -> "translation"
    Fill in blanks          -> "fill-mask"
    Embeddings              -> "feature-extraction"
    Zero-shot classify      -> "zero-shot-classification"
    Image classification    -> "image-classification"
    Object detection        -> "object-detection"
    
    Common Parameters:
    ─────────────────────────────────────────────────────────
    model="model-name"      -> Specific model (default varies)
    device=0                -> GPU index (-1 for CPU)
    batch_size=8            -> Batch inference
    
    Interview Tips:
    ─────────────────────────────────────────────────────────
    1. Start with pipeline() - it's the fastest path to working code
    2. Only go to manual inference if you need custom behavior
    3. Mention you'd use smaller models in production if latency matters
    4. Know that pipelines handle tokenization automatically
    """
    print(guide)


if __name__ == "__main__":
    # Show the selection guide first
    print_pipeline_guide()
    
    # Run examples (comment out if you don't have models downloaded)
    print("\n\nRunning examples (requires model downloads)...\n")
    
    # These may take time on first run due to model downloads
    text_classification_example()
    # Uncomment others as needed:
    # named_entity_recognition_example()
    # question_answering_example()
    # text_generation_example()
    # zero_shot_classification_example()
    # summarization_example()
    # fill_mask_example()
    # manual_inference_example()
    
    print("\n" + "=" * 60)
    print("Done! For more examples, uncomment other functions in main.")
    print("=" * 60)
