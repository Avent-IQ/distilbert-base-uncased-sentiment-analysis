# DistilBERT Base Uncased Quantized Model for Sentiment Analysis

This repository hosts a quantized version of the DistilBERT model, fine-tuned for sentiment analysis tasks. The model has been optimized for efficient deployment while maintaining high accuracy, making it suitable for resource-constrained environments.

## Model Details

- **Model Architecture:** DistilBERT Base Uncased  
- **Task:** Sentiment Analysis  
- **Dataset:** IMDB Reviews  
- **Quantization:** Float16  
- **Fine-tuning Framework:** Hugging Face Transformers  

## Usage

### Installation

```sh
pip install transformers torch
```

### Loading the Model

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

model_name = "AventIQ-AI/distilbert-base-uncased-sentiment-analysis"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return "Positive" if predicted_class_id == 1 else "Negative"

# Test the model with a sample sentence
test_text = "I absolutely loved the movie! It was fantastic."
print(f"Sentiment: {predict_sentiment(test_text)}")
```

## Performance Metrics

- **Accuracy:** 0.56  
- **F1 Score:** 0.56  
- **Precision:** 0.68  
- **Recall:** 0.56  

## Fine-Tuning Details

### Dataset

The IMDb Reviews dataset was used, containing both positive and negative sentiment examples.

### Training

- Number of epochs: 3  
- Batch size: 16  
- Evaluation strategy: epoch  
- Learning rate: 2e-5  

### Quantization

Post-training quantization was applied using PyTorch's built-in quantization framework to reduce the model size and improve inference efficiency.

## Repository Structure

```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safensors/     # Fine Tuned Model
├── README.md            # Model documentation
```

## Limitations

- The model may not generalize well to domains outside the fine-tuning dataset.  
- Quantization may result in minor accuracy degradation compared to full-precision models.  

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.

