import transformers

transformers.logging.set_verbosity_error()

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# dipesh/Intent-Classification-bart-large-mnli
# dipesh/autotrain-intent-classification-2953985512
# dipesh/Intent-Classification-roberta-base-finetuned-intent
tokenizer = AutoTokenizer.from_pretrained('dipesh/Intent-Classification-roberta-base-finetuned-intent')
model = AutoModelForSequenceClassification.from_pretrained("dipesh/Intent-Classification-roberta-base-finetuned-intent")

from transformers import pipeline

nlp = pipeline('zero-shot-classification', model=model, tokenizer=tokenizer)

sequence = "how are you"
classes = ['asking date', 'asking time', 'tell me joke', 'tell me news',
           'asking weather', 'tell me about', 'open website',
           'play on youtube', 'send whatsapp message', 'send email',
           'greet and hello hi kind of things, general check in', 'goodbye',
           'take screenshot', 'click photo', 'download youtube video',
           'covid cases', 'play games', 'places near me', 'i am bored',
           'volume control', 'what can you do']

results = nlp(sequence, classes)
print(results)

print(results['labels'][0])
