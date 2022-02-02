import gc
import json
import os
from argparse import ArgumentParser
from datetime import datetime
from functools import partial

import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
from flash import Trainer
from flash.text import QuestionAnsweringData, QuestionAnsweringTask
from plotly.graph_objs._figure import Figure

from src import OUTPUT_FOLDER_PATH, RANDOM_SEED, TRAIN_DATA_PATH, VAL_DATA_PATH


# The objective function would look something like this
def objective(
    trial: optuna.Trial,
    max_epochs: int = 5,
):

    # A unique set of hyperparameter combination is sampled.
    learning_rate = trial.suggest_uniform("learning_rate", 1e-5, 5e-4)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    backbone = trial.suggest_categorical("backbone", ["xlm-roberta-base", "bert-base-multilingual-uncased"])
    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])

    pl.seed_everything(RANDOM_SEED)

    # Setup the machine learning pipeline with the new set
    # of hyperparameters.
    datamodule = QuestionAnsweringData.from_csv(
        train_file=TRAIN_DATA_PATH,
        val_file=VAL_DATA_PATH,
        backbone=backbone,
        batch_size=batch_size,
    )

    model = QuestionAnsweringTask(
        backbone=backbone,
        learning_rate=learning_rate,
        optimizer=optimizer,
    )

    trainer = Trainer(max_epochs=max_epochs, gpus=torch.cuda.device_count(), accumulate_grad_batches=2)

    # Train the model for Optuna to understand the current
    # Hyperparameter combination's behaviour.
    trainer.fit(model, datamodule)

    # The extra step to tell Optuna which value to base the
    # optimization routine on.
    value = trainer.callback_metrics["val_loss"].item()

    del datamodule, model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return value


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--trials", type=int, default=1, required=False)
    parser.add_argument("-e", "--epochs", type=int, default=5, required=False)
    args = parser.parse_args()

    study = optuna.create_study(study_name=f"optuna_flash_{datetime.now()}")
    study.optimize(partial(objective, max_epochs=args.max_epochs), n_trials=args.trials, gc_after_trial=True)

    # Save the study to a Pandas DataFrame
    STUDY_PATH = os.path.join(OUTPUT_FOLDER_PATH, f"{study.study_name}")
    if not os.path.exists(STUDY_PATH):
        os.mkdir(STUDY_PATH)

    best_params = study.best_params
    with open(os.path.join(STUDY_PATH, "best_params.json"), "w") as f:
        f.write(json.dumps(best_params))

    trials_df: pd.DataFrame = study.trials_dataframe()
    trials_df.to_csv(path=os.path.join(STUDY_PATH, "all_trials.csv"))

    # Save the plots of the Study
    if optuna.visualization.is_available():
        parallel_coordinate_fig: Figure = optuna.visualization.plot_parallel_coordinate(study, params=["x", "y"])
        parallel_coordinate_fig.write_image(os.path.join(STUDY_PATH, "plot_parallel_coordinate.jpeg"))

        param_importances_fig: Figure = optuna.visualization.plot_param_importances(study)
        param_importances_fig.write_image(os.path.join(STUDY_PATH, "param_importances.jpeg"))
