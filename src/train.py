import json
import os
from argparse import ArgumentParser

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
):
    datamodule = QuestionAnsweringData.from_csv(
        train_file=TRAIN_DATA_PATH,
        val_file=VAL_DATA_PATH,
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=1, required=False)
    parser.add_argument("-s", "--study_name", type=str, default=None, required=False)
    args = parser.parse_args()

    train_args = {"max_epochs": args.epochs}

    if args.study_name is not None:
        STUDY_PATH = os.path.join(OUTPUT_FOLDER_PATH, f"{args.study_name}")
        with open(os.path.join(STUDY_PATH, "best_params.json")) as f:
            best_params = json.load(f)

    train_args.update(best_params)

    train(**train_args)
