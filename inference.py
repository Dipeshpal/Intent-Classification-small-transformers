from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained(
    "dipesh/distilbert-base-uncased-Intent-Classification-Commands-balanced-ds-large")
model = AutoModelForSequenceClassification.from_pretrained(
    "dipesh/distilbert-base-uncased-Intent-Classification-Commands-balanced-ds-large")
classifier_return_all_scores = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True)
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

sequence = "make me laugh"

# Classify the sequence
results = classifier(sequence)

# Print the results
print(results)
