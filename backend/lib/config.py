from dataclasses import dataclass


@dataclass()
class ModelConfig:
    model_path: str
    tokenizer_path: str


@dataclass()
class TrainConfig:
    csv_save_path: str
    output_path: str
