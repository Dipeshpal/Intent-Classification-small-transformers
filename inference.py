import transformers
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

transformers.logging.set_verbosity_error()

# dipesh/Intent-Classification-bart-large-mnli
# dipesh/autotrain-intent-classification-2953985512
# dipesh/Intent-Classification-roberta-base-finetuned-intent  # RECOMMENDED
tokenizer = AutoTokenizer.from_pretrained('dipesh/Intent-Classification-roberta-base-finetuned-intent')
model = AutoModelForSequenceClassification.from_pretrained("dipesh/Intent-Classification-roberta-base-finetuned-intent")


nlp = pipeline('zero-shot-classification', model=model, tokenizer=tokenizer)

classes = ['asking date', 'asking time', 'tell me joke', 'tell me news',
           'asking weather', 'tell me about', 'open website',
           'play on youtube', 'send whatsapp message', 'send email',
           'greet and hello hi kind of things, general check in', 'goodbye',
           'take screenshot', 'click photo', 'download youtube video',
           'covid cases', 'play games', 'places near me', 'i am bored',
           'volume control', 'what can you do']


# TESTING ON SAME DATASET, YOU CAN USE DIFFERENT ONE
df = pd.read_csv("data_generator/train.csv")


wrong = 0
correct = 0
total = 0
for index, row in df.iterrows():
    sequence = row['text']
    results = nlp(sequence, classes)
    if results['labels'][0] == row['label']:
        print("TRUE: ", sequence, "| ", results['labels'][0], ": ", results['scores'][0], "| ", row['label'])
        correct += 1
    else:
        print("FALSE: ", sequence, "| ", results['labels'][0], ": ", results['scores'][0], "| ", row['label'])
        wrong += 1
    total += 1

print("Correct: ", correct)
print("Wrong: ", wrong)
print("Total: ", total)
print("Accuracy: ", correct/total)
