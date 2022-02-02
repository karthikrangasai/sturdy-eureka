# Study Eureka


- **install.sh** should install the requirements, download the dataset, and run the preprocessing beforehand
- **src/train.py**  Runs the training (if study name is provided as an input to the script, automatically starts training with the best params from that study)
- **src/hparams_search.py** runs the optuna search, saves the study to a folder and the best params; this script also saves the parameter vs performance plots