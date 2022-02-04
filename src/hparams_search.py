import gc
import json
import os
from datetime import datetime
from functools import partial

import fire
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
    gpus: int = torch.cuda.device_count(),
    batch_size: int = 2,
    accumulate_grad_batches: int = 2,
    path_train_data: str = TRAIN_DATA_PATH,
    path_val_data: str = VAL_DATA_PATH,
):

    # A unique set of hyperparameter combination is sampled.
    learning_rate = trial.suggest_uniform("learning_rate", 5e-6, 1e-4)
    backbone = trial.suggest_categorical("backbone", ["xlm-roberta-base", "bert-base-multilingual-uncased"])
    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])

    pl.seed_everything(RANDOM_SEED)

    # Setup the machine learning pipeline with the new set of hyperparameters.
    dm = QuestionAnsweringData.from_csv(
        train_file=path_train_data,
        val_file=path_val_data,
        backbone=backbone,
        batch_size=batch_size,
    )

    model = QuestionAnsweringTask(
        backbone=backbone,
        learning_rate=learning_rate,
        optimizer=optimizer,
    )

    trainer = Trainer(max_epochs=max_epochs, gpus=gpus, accumulate_grad_batches=accumulate_grad_batches)

    # Train the model for Optuna to understand the current
    # Hyperparameter combination's behaviour.
    trainer.fit(model, dm)

    # The extra step to tell Optuna which value to base the
    # optimization routine on.
    value = trainer.callback_metrics["val_loss"].item()

    del dm, model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return value


def main(
    trials: int = 1,
    epochs: int = 5,
    gpus: int = torch.cuda.device_count(),
    batch_size: int = 2,
    accumulate_grad_batches: int = 2,
    output_folder: str = OUTPUT_FOLDER_PATH,
):

    study = optuna.create_study(study_name=f"optuna_flash_{datetime.now()}")
    adjusted_objective = partial(
        objective,
        max_epochs=epochs,
        gpus=gpus,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    study.optimize(adjusted_objective, n_trials=trials, gc_after_trial=True, show_progress_bar=True)

    # Save the study to a Pandas DataFrame
    study_path = os.path.join(output_folder, f"{study.study_name}")
    if not os.path.exists(study_path):
        os.mkdir(study_path)

    best_params = study.best_params
    with open(os.path.join(study_path, "best_params.json"), "w") as f:
        f.write(json.dumps(best_params))

    trials_df: pd.DataFrame = study.trials_dataframe()
    trials_df.to_csv(os.path.join(study_path, "all_trials.csv"))

    # Save the plots of the Study
    if optuna.visualization.is_available():
        parallel_coordinate_fig: Figure = optuna.visualization.plot_parallel_coordinate(study, params=["x", "y"])
        parallel_coordinate_fig.write_image(os.path.join(study_path, "plot_parallel_coordinate.jpeg"))

        param_importances_fig: Figure = optuna.visualization.plot_param_importances(study)
        param_importances_fig.write_image(os.path.join(study_path, "param_importances.jpeg"))


if __name__ == "__main__":
    fire.Fire(main)
