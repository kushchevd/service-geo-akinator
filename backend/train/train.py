import os
from datetime import datetime

import onnx
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

from lib.model import BertForLatLonInference, GeoAkinatorModel

CSV_FILE = "train/data/full_markup.csv"


def tokenize(example, tokenizer):
    return tokenizer(
        example["text"], padding="max_length", truncation=True, max_length=128
    )


def add_labels(example):
    example["labels"] = torch.tensor(
        [example["lat"], example["lon"]], dtype=torch.float
    )
    return example


def convert_to_onnx(model: GeoAkinatorModel, output_path: str):
    onnx_model_path = os.path.join(output_path, f"{model.exp_name}.onnx")
    commit_hash = os.getenv("GIT_COMMIT_HASH", "failed")

    save_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        "commit_hash": commit_hash,
        "model_save_date": save_date,
        "experiment_name": model.exp_name,
    }

    example_text = "Shibuya Crossing"
    inputs = model.tokenizer(
        example_text, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    inference_model = BertForLatLonInference.from_pretrained(
        model.model_path, config=model.config
    )
    torch.onnx.export(
        inference_model,
        (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
        onnx_model_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["latlon"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "latlon": {0: "batch_size"},
        },
        verbose=False,
    )
    onnx_model = onnx.load(onnx_model_path)
    for key, value in metadata.items():
        meta = onnx_model.metadata_props.add()
        meta.key = str(key)
        meta.value = str(value)
    onnx.save(onnx_model, onnx_model_path)


def get_exp_name(output_path: str):
    exist_vers = sorted(
        [
            foldername.split("_")[-1]
            for foldername in os.listdir("train/data")
            if os.path.isdir(os.path.join("train/data", foldername))
        ]
    )
    if exist_vers:
        exp_name = f"model_v{int(exist_vers[-1][1:])+1}"
    else:
        exp_name = "model_v1"
    return exp_name


def train_model(
    markup_path: str = CSV_FILE, output_path: str = "train/data", exp_name: str = None
):
    if exp_name is None:
        exp_name = get_exp_name(output_path)
    model_dir = os.path.join(output_path, exp_name)
    os.mkdir(model_dir)
    df = pd.read_csv(markup_path)[["text", "lat", "lon"]].dropna()
    train_df, val_df = train_test_split(df, test_size=0.01, random_state=42)

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

    model = GeoAkinatorModel(
        model_path="bert-base-uncased",
        tokenizer_path="bert-base-uncased",
        exp_name=exp_name,
    )

    train_dataset = train_dataset.map(
        lambda x: tokenize(x, model.tokenizer), batched=True
    )
    val_dataset = val_dataset.map(lambda x: tokenize(x, model.tokenizer), batched=True)

    train_dataset = train_dataset.map(add_labels)
    val_dataset = val_dataset.map(add_labels)

    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    val_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        num_train_epochs=1,
        weight_decay=0.0001,
        logging_dir=os.path.join(model_dir, "logs"),
        logging_steps=50,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    convert_to_onnx(model, model_dir)


if __name__ == "__main__":
    train_model()
