# HuggingFace Tokenization Notes

## Why Tokenization Matters

Tokenization converts raw text into numbers that models understand. Different tokenizers produce different results, so **you must use the tokenizer that matches your model**.

```python
from transformers import AutoTokenizer

# Always load the tokenizer that matches your model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

## Basic Usage

```python
# Simple tokenization
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
# ['hello', ',', 'how', 'are', 'you', '?']

# Get token IDs
ids = tokenizer.encode(text)
# [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]
#  ^CLS                                    ^SEP

# Full encoding for model input
encoded = tokenizer(text, return_tensors="pt")
# {'input_ids': tensor([[...]]), 'attention_mask': tensor([[...]])}
```

## Key Parameters

```python
encoded = tokenizer(
    text,
    padding=True,           # Pad shorter sequences
    truncation=True,        # Truncate longer sequences
    max_length=128,         # Maximum sequence length
    return_tensors="pt",    # Return PyTorch tensors ("tf" for TensorFlow)
    add_special_tokens=True # Add [CLS], [SEP], etc. (default True)
)
```

## Batch Tokenization

```python
texts = ["First sentence.", "Second sentence."]

# Batch tokenization with padding
batch = tokenizer(
    texts,
    padding=True,      # Pad to longest in batch
    truncation=True,
    return_tensors="pt"
)

print(batch['input_ids'].shape)  # (2, max_len)
print(batch['attention_mask'].shape)  # (2, max_len)
```

## Understanding the Output

```python
encoded = tokenizer("Hello world", return_tensors="pt")

# input_ids: Token indices
print(encoded['input_ids'])
# tensor([[  101,  7592,  2088,   102]])
#          CLS   hello  world   SEP

# attention_mask: 1 for real tokens, 0 for padding
print(encoded['attention_mask'])
# tensor([[1, 1, 1, 1]])

# token_type_ids: Segment IDs (for sentence pairs)
print(encoded.get('token_type_ids'))
# tensor([[0, 0, 0, 0]])  # All same segment
```

## Sentence Pairs

```python
# For tasks like NLI, QA, etc.
question = "What is the capital?"
context = "Paris is the capital of France."

encoded = tokenizer(
    question,
    context,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# token_type_ids distinguishes the two sentences
# 0 0 0 0 0 1 1 1 1 1 1 1 1
# [question] [context     ]
```

## Decoding Back to Text

```python
# Decode token IDs back to text
ids = [101, 7592, 2088, 102]
text = tokenizer.decode(ids)
# "[CLS] hello world [SEP]"

# Skip special tokens
text = tokenizer.decode(ids, skip_special_tokens=True)
# "hello world"

# Batch decode
texts = tokenizer.batch_decode(batch_ids, skip_special_tokens=True)
```

## Special Tokens

```python
# Access special tokens
print(tokenizer.cls_token)  # [CLS]
print(tokenizer.sep_token)  # [SEP]
print(tokenizer.pad_token)  # [PAD]
print(tokenizer.unk_token)  # [UNK]
print(tokenizer.mask_token) # [MASK]

# Get their IDs
print(tokenizer.cls_token_id)  # 101
print(tokenizer.pad_token_id)  # 0
```

## Tokenizer Types

### WordPiece (BERT)
```python
tokenizer.tokenize("unbelievable")
# ['un', '##believ', '##able']
# ## indicates continuation of word
```

### BPE (GPT-2, RoBERTa)
```python
tokenizer.tokenize("unbelievable")
# ['un', 'believ', 'able']  # No ## prefix
```

### SentencePiece (T5, XLNet)
```python
tokenizer.tokenize("unbelievable")
# ['▁un', 'believ', 'able']  # ▁ indicates start of word
```

## Practical Tips

### 1. Always Match Tokenizer to Model
```python
# WRONG: Mismatched tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base")

# CORRECT: Same model name
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

### 2. Handle Long Texts
```python
# Option 1: Truncation (lose information)
encoded = tokenizer(long_text, truncation=True, max_length=512)

# Option 2: Sliding window (process in chunks)
def chunk_text(text, tokenizer, max_length=512, stride=128):
    """Process long text in overlapping chunks."""
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        return_tensors="pt"
    )
    return encoded
```

### 3. Vocabulary Size
```python
vocab_size = len(tokenizer)
print(f"Vocabulary size: {vocab_size}")
# BERT: ~30,522
# GPT-2: ~50,257
```

### 4. Padding Strategy
```python
# Pad to max length in batch (efficient)
tokenizer(texts, padding=True, return_tensors="pt")

# Pad to fixed length (needed for some setups)
tokenizer(texts, padding='max_length', max_length=128, return_tensors="pt")
```

## Interview Talking Points

1. **Why not split on spaces?**
   - Handles unknown words via subword tokenization
   - Works across languages
   - Fixed vocabulary size

2. **What's the attention mask for?**
   - Tells the model which tokens are padding
   - Prevents model from attending to padding tokens

3. **Why max_length=512?**
   - BERT-style models have position embeddings up to 512
   - Longer requires different models (Longformer, BigBird)

4. **Tokenization affects model behavior**
   - Different tokenizers = different representations
   - "New York" might be ["new", "york"] or ["new", "##york"]
   - This affects downstream tasks
