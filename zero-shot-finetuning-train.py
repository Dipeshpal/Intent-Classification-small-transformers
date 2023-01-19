import torch
import shutup
import os
from datetime import datetime
from transformers import logging
from datasets import load_dataset
from setfit import SetFitModel, SetFitTrainer

shutup.please()
logging.set_verbosity_error()

epochs = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


def save_and_evaluate_model(trainer):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists('model/'):
        os.mkdir('model/')
    save_directory = f"model/{timestamp}-Intent-Classification-roberta-base-finetuned-intent"
    os.mkdir(save_directory)
    trainer.model._save_pretrained(save_directory=save_directory)
    print(f"Model saved to {save_directory}")

    zeroshot_metrics = trainer.evaluate()
    print(zeroshot_metrics)


dataset_id = "dipesh/Intent-Classification-Commands-large"
model_id = "dipesh/Intent-Classification-roberta-base-finetuned-intent"  # "typeform/distilbert-base-uncased-mnli" #facebook/bart-large-mnli #zhiyil/roberta-base-finetuned-intent

reference_dataset = load_dataset(dataset_id)
reference_dataset = reference_dataset.rename_column("intent", "label")
print(reference_dataset)


def train_and_load_model(model_id):
    model = SetFitModel.from_pretrained(model_id)
    trainer = SetFitTrainer(
        model=model,
        train_dataset=reference_dataset["train"],
        eval_dataset=reference_dataset["test"],
        batch_size=64,
        num_iterations=16,  # The number of text pairs to generate for contrastive learning
        num_epochs=1,  # The number of epochs to use for constrastive learning
    )

    trainer.train()
    return trainer


for i in range(epochs):
    trainer = train_and_load_model(model_id)
    save_and_evaluate_model(trainer)
    print('Checking ["how are you", "what is the time now", "who is narendra modi"]: ',
          trainer.model.predict(["how are you", "what is the time now", "who is narendra modi"]))
    print(trainer.model.push_to_hub('Intent-Classification-roberta-base-finetuned-intent'))

print("Done")
