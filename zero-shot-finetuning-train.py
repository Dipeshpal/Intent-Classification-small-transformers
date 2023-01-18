import shutup
from transformers import logging
from datasets import load_dataset
from setfit import SetFitModel, SetFitTrainer
import os
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

shutup.please()
logging.set_verbosity_error()

# Set up the dataset
dataset_id = "dipesh/Intent-Classification-Commands"
# set up pre-trained model
model_id = "facebook/bart-large-mnli"  # "typeform/distilbert-base-uncased-mnli" #facebook/bart-large-mnli

# your model name
YOUR_MODEL_NAME = "Intent-Classification-bart-large-mnli"

reference_dataset = load_dataset(dataset_id)
reference_dataset = reference_dataset.rename_column("labels", "label")
print(reference_dataset)

model = SetFitModel.from_pretrained(model_id)
trainer = SetFitTrainer(
    model=model,
    train_dataset=reference_dataset["train"],
    eval_dataset=reference_dataset["test"]
)

trainer.train()
zeroshot_metrics = trainer.evaluate()
print(zeroshot_metrics)

# save the model
if not os.path.exists(YOUR_MODEL_NAME):
    os.mkdir(YOUR_MODEL_NAME)

save_directory = YOUR_MODEL_NAME
trainer.model._save_pretrained(save_directory=save_directory)

# make predictions
print("Making predictions")
print("what is the weather like today?: ", trainer.model.predict(["what is the weather like today?"]))

print("YOUR MODEL NAME: ", YOUR_MODEL_NAME, "saved in ", save_directory, "directory")
print("PUSH TO HUGGING FACE...")
print(trainer.model.push_to_hub(YOUR_MODEL_NAME))
