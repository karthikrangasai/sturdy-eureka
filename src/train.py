import torch
import pytorch_lightning as pl
import wandb

from argparse import ArgumentParser
from dataclasses import asdict, dataclass

from flash import Trainer
from flash.text import QuestionAnsweringData, QuestionAnsweringTask
from pytorch_lightning.loggers import WandbLogger

from src import TRAIN_DATA_PATH, VAL_DATA_PATH, RANDOM_SEED

pl.seed_everything(RANDOM_SEED)


def train(
    max_epochs: int = 5,
    unfreeze_epoch: int = 2,
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

    wandb_logger = WandbLogger(
        project="chaii-competition",
        config={
            "max_epochs": max_epochs,
            "backbone": backbone,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "accumulate_grad_batches": accumulate_grad_batches,
            "finetuning_strategy": ("freeze_unfreeze", unfreeze_epoch),
        },
        log_model=False,
    )

    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    wandb_logger.watch(model)
    trainer.finetune(model, datamodule, strategy=("freeze_unfreeze", unfreeze_epoch))
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--trials", type=int, default=1, required=False)
    parser.add_argument("-e", "--epochs", type=int, default=5, required=False)
    parser.add_argument("-u", "--unfreeze_epoch", type=int, default=2, required=False)
    args = parser.parse_args()

    train(
        max_epochs=args.max_epochs,
        unfreeze_epoch=args.unfreeze_epoch,
    )
