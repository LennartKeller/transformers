import math
import copy
import random
from turtle import forward
from typing import Optional, Tuple, List, Union

import torch
from torch import Tensor, nn

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from transformers import EvalPrediction
import numpy as np



from transformers import PyramidionsConfig, PyramidionsModel, RobertaTokenizerFast, PyramidionsForSequenceClassification


print("Creating model...")
tokenizer = RobertaTokenizerFast.from_pretrained("uklfr/gottbert-base")
tokenizer.model_max_length = 512
tokenizer.init_kwargs["model_max_length"] = 512


config = PyramidionsConfig()
config.update({"num_hidden_layers": 9, "max_position_embeddings": 514, "type_vocab_size": 1, "num_labels": 9})
print(config)
model = PyramidionsForSequenceClassification(config)

model.resize_token_embeddings(len(tokenizer))


print("Loading pretrained model and copying weights...")
from transformers import RobertaModel

pretrained_model = RobertaModel.from_pretrained("uklfr/gottbert-base")



from collections import OrderedDict
def rename_roberta_state_dict(state_dict):
    return OrderedDict([(f"pyramidions.{key}", value) for key, value in state_dict.items()])



model.load_state_dict(rename_roberta_state_dict(pretrained_model.state_dict()), strict=False)

print("Preparing data")
from datasets import load_dataset

dataset = load_dataset("gnad10")


dataset = dataset.map(lambda row: tokenizer(row["text"], truncation=True))

dataset.set_format("torch")
dataset = dataset.rename_column("label", "labels")
dataset = dataset.remove_columns("text")

#dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)["test"]
#dataset = dataset.train_test_split(test_size=0.05, seed=42)

from sklearn.metrics import f1_score
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item(), "f1_score": f1_score(p.label_ids, preds, average="macro")}


from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
# I think that the trainer is not using the right padding strategy....

training_args = TrainingArguments(
    num_train_epochs=3,
    output_dir="test_runs/run/pyramid_classif_gnad10",
    logging_dir="test_runs/logs/pyramid_classif_gnad10",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    learning_rate=3e-5,
    logging_strategy="steps",
    logging_steps=1,
    evaluation_strategy="steps",
    eval_steps=20
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=DataCollatorWithPadding(padding="max_length", max_length=tokenizer.model_max_length, tokenizer=tokenizer),
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)
print("Fire...")
trainer.train()

