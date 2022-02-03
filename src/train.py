import json
import os

import fire
import pytorch_lightning as pl
import torch
from flash import Trainer
from flash.text import QuestionAnsweringData, QuestionAnsweringTask

from src import OUTPUT_FOLDER_PATH, RANDOM_SEED, TRAIN_DATA_PATH, VAL_DATA_PATH

pl.seed_everything(RANDOM_SEED)


def train(
    max_epochs: int = 5,
    backbone: str = "xlm-roberta-base",
    optimizer: str = "adamw",
    learning_rate: float = 1e-5,
    batch_size: int = 2,
    accumulate_grad_batches: int = 2,
    path_train_data: str = TRAIN_DATA_PATH,
    path_val_data: str = VAL_DATA_PATH,
):
    datamodule = QuestionAnsweringData.from_csv(
        train_file=path_train_data,
        val_file=path_val_data,
        batch_size=batch_size,
        backbone=backbone,
    )

    model = QuestionAnsweringTask(
        backbone=backbone,
        learning_rate=learning_rate,
        optimizer=optimizer,
    )

    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer.fit(model, datamodule=datamodule)


def main(epochs: int = 1, study_name: str = None, output_folder: str = OUTPUT_FOLDER_PATH):
    train_args = {"max_epochs": epochs}

    if study_name is not None:
        study_path = os.path.join(output_folder, f"{study_name}")
        with open(os.path.join(study_path, "best_params.json")) as fp:
            best_params = json.load(fp)
        train_args.update(best_params)

    train(**train_args)


if __name__ == "__main__":
    fire.Fire(main)
