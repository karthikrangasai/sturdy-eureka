import fire
import pandas as pd

from src import INPUT_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH


def create_spilts(
    train_val_split: float = 0.1,
    path_input_data: str = INPUT_DATA_PATH,
    path_train_data: str = TRAIN_DATA_PATH,
    path_val_data: str = VAL_DATA_PATH,
):
    df = pd.read_csv(path_input_data)

    fraction = 1 - train_val_split

    # Splitting data into train and val beforehand since preprocessing will be different for datasets.
    tamil_examples = df[df["language"] == "tamil"]
    train_split_tamil = tamil_examples.sample(frac=fraction, random_state=200)
    val_split_tamil = tamil_examples.drop(train_split_tamil.index)

    hindi_examples = df[df["language"] == "hindi"]
    train_split_hindi = hindi_examples.sample(frac=fraction, random_state=200)
    val_split_hindi = hindi_examples.drop(train_split_hindi.index)

    train_split = pd.concat([train_split_tamil, train_split_hindi]).reset_index(drop=True)
    val_split = pd.concat([val_split_tamil, val_split_hindi]).reset_index(drop=True)

    train_split.to_csv(path_train_data, index=False)
    val_split.to_csv(path_val_data, index=False)


if __name__ == "__main__":
    fire.Fire(create_spilts)
