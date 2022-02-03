# Study Eureka

## Content

- **install.sh** should install the requirements, download the dataset, and run the preprocessing beforehand
  - before downloading the kaggle dataset you need fist enter competition and accept rules
  - to download it you need to export you credentials (username/key) to a file or use env. variables (`KAGGLE_USERNAME`, `KAGGLE_KEY`)
- **src/train.py**  Runs the training (if study name is provided as an input to the script, automatically starts training with the best params from that study)
- **src/hparams_search.py** runs the optuna search, saves the study to a folder and the best params; this script also saves the parameter vs performance plots

## Hints

- if you cannot see the `src` package from the running script, add `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`
