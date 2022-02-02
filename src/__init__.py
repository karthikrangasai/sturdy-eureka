import os

SRC_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(SRC_ROOT_PATH)

DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT_PATH, "data", "chaii-hindi-and-tamil-question-answering")

INPUT_DATA_PATH = os.path.join(DATA_FOLDER_PATH, "train.csv")
TRAIN_DATA_PATH = os.path.join(DATA_FOLDER_PATH, "_train.csv")
VAL_DATA_PATH = os.path.join(DATA_FOLDER_PATH, "_val.csv")
PREDICT_DATA_PATH = os.path.join(DATA_FOLDER_PATH, "test.csv")

OUTPUT_FOLDER_PATH = os.path.join(PROJECT_ROOT_PATH, "outputs")
if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.mkdir(OUTPUT_FOLDER_PATH)

RANDOM_SEED = 42
