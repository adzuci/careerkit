# HuggingFace Fine-Tuning Outline

This outlines the high-level fine-tuning workflow. In an interview, you'd describe these steps and implement the critical parts.

## When to Fine-Tune

**Fine-tune when:**
- Pretrained model isn't accurate enough on your task
- You have labeled data for your specific domain
- Zero-shot/few-shot approaches aren't sufficient

**Skip fine-tuning when:**
- Pipeline with pretrained model works well enough
- You don't have enough labeled data (< 100 examples)
- Time-to-solution matters more than accuracy

## High-Level Workflow

```
1. Load pretrained model and tokenizer
2. Prepare dataset (tokenize, format)
3. Configure training (Trainer or manual loop)
4. Train and evaluate
5. Save and deploy
```

## Step 1: Load Model and Tokenizer

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "bert-base-uncased"
num_labels = 2  # Binary classification

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)
```

## Step 2: Prepare Dataset

```python
from datasets import load_dataset, Dataset

# Option 1: Load existing dataset
dataset = load_dataset("imdb")

# Option 2: From pandas DataFrame
import pandas as pd
df = pd.DataFrame({"text": texts, "label": labels})
dataset = Dataset.from_pandas(df)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
```

## Step 3: Configure Training

### Option A: Trainer (Recommended for Interviews)

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)
```

### Option B: Manual Training Loop

```python
from torch.utils.data import DataLoader
import torch

# DataLoader
train_loader = DataLoader(tokenized_dataset["train"], batch_size=16, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1} completed")
```

## Step 4: Evaluate

```python
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# With Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,  # Add this
)

# Run evaluation
results = trainer.evaluate()
print(results)
```

## Step 5: Save and Load

```python
# Save
model.save_pretrained("./my_finetuned_model")
tokenizer.save_pretrained("./my_finetuned_model")

# Load for inference
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./my_finetuned_model",
    tokenizer="./my_finetuned_model"
)

result = classifier("This movie was great!")
```

## Key Training Arguments to Know

| Argument | Purpose | Typical Values |
|----------|---------|----------------|
| `num_train_epochs` | Training epochs | 2-5 |
| `per_device_train_batch_size` | Batch size | 8, 16, 32 |
| `learning_rate` | Learning rate | 2e-5 to 5e-5 |
| `warmup_steps` | LR warmup | 100-1000 |
| `weight_decay` | Regularization | 0.01 |
| `evaluation_strategy` | When to eval | "epoch", "steps" |
| `save_strategy` | When to save | "epoch", "steps" |
| `fp16` | Mixed precision | True (if GPU) |

## Common Gotchas

1. **Labels column name**: HuggingFace expects `labels`, not `label`
   ```python
   dataset = dataset.rename_column("label", "labels")
   ```

2. **Model output**: When labels provided, loss is first output
   ```python
   outputs = model(**inputs, labels=labels)
   loss = outputs.loss
   logits = outputs.logits
   ```

3. **Freezing layers** (for efficiency):
   ```python
   for param in model.base_model.parameters():
       param.requires_grad = False
   # Only classifier head is trained
   ```

4. **Class imbalance**: Use weighted loss
   ```python
   from torch.nn import CrossEntropyLoss
   weights = torch.tensor([1.0, 5.0])  # Weight minority class higher
   criterion = CrossEntropyLoss(weight=weights)
   ```

## Interview Discussion Points

1. **Why fine-tune vs. train from scratch?**
   - Pretrained models encode general language understanding
   - Fine-tuning adapts to specific task with less data
   - Much faster and often more accurate

2. **How much data do you need?**
   - Minimum: 100-500 examples
   - Good results: 1,000-10,000 examples
   - More always helps, with diminishing returns

3. **What if fine-tuning doesn't improve performance?**
   - Check data quality and label accuracy
   - Try different learning rates (lower usually)
   - Try a different pretrained model
   - Consider if the task is too different from pretraining

4. **How to prevent overfitting?**
   - Use validation set for early stopping
   - Increase weight decay
   - Use dropout
   - Reduce epochs
   - Freeze more layers
