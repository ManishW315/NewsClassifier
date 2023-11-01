import os
import re
from typing import Dict, Tuple
from warnings import filterwarnings

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from newsclassifier.config.config import Cfg, logger
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

filterwarnings("ignore")


def load_dataset(filepath: str, print_i: int = 0) -> pd.DataFrame:
    """load data from source into a Pandas DataFrame.

    Args:
        filepath (str): file location.
        print_i (int): Print number of instances.

    Returns:
        pd.DataFrame: Pandas DataFrame of the data.
    """
    logger.info("Loading Data.")
    df = pd.read_csv(filepath)
    if print_i:
        print(df.head(print_i), "\n")
    return df


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separate headlines instance and feature selection.

    Args:
        df: original dataframe.

    Returns:
        df: new dataframe with appropriate features.
        headlines_df: dataframe cintaining "headlines" category instances.
    """
    df = df[["Title", "Category"]]
    df.rename(columns={"Title": "Text"}, inplace=True)
    df, headlines_df = df[df["Category"] != "Headlines"].reset_index(drop=True), df[df["Category"] == "Headlines"].reset_index(drop=True)

    return df, headlines_df


def clean_text(text: str) -> str:
    """Clean text (lower, puntuations removal, blank space removal)."""
    # lower case the text
    text = text.lower()  # necessary to do before as stopwords are in lower case

    # remove stopwords
    stp_pattern = re.compile(r"\b(" + r"|".join(Cfg.STOPWORDS) + r")\b\s*")
    text = stp_pattern.sub("", text)

    # custom cleaning
    text = text.strip()  # remove space at start or end if any
    text = re.sub(" +", " ", text)  # remove extra spaces
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove characters that are not alphanumeric

    return text


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """Preprocess the data.

    Args:
        df: Dataframe on which the preprocessing steps need to be performed.

    Returns:
        df: Preprocessed Data.
        class_to_index: class labels to indices mapping
        class_to_index: indices to class labels mapping
    """
    df, headlines_df = prepare_data(df)

    cats = df["Category"].unique().tolist()
    num_classes = len(cats)
    class_to_index = {tag: i for i, tag in enumerate(cats)}
    index_to_class = {v: k for k, v in class_to_index.items()}

    df["Text"] = df["Text"].apply(clean_text)  # clean text
    df = df[["Text", "Category"]]
    df["Category"] = df["Category"].map(class_to_index)  # label encoding
    return df, headlines_df, class_to_index, index_to_class


def data_split(df: pd.DataFrame, split_size: float = 0.2, stratify_on_target: bool = True, save_dfs: bool = False):
    """Split data into train and test sets.

    Args:
        df (pd.DataFrame): Data to be split.
        split_size (float): train-test split ratio (test ratio).
        stratify_on_target (bool): Whether to do stratify split on target.
        target_sep (bool): Whether to do target setting for train and test sets.
        save_dfs (bool): Whether to save dataset splits in artifacts.

    Returns:
        train-test splits (with/without target setting)
    """
    logger.info("Splitting Data.")
    if stratify_on_target:
        stra = df["Category"]
    else:
        stra = None

    train, test = train_test_split(df, test_size=split_size, random_state=42, stratify=stra)
    train_ds = pd.DataFrame(train, columns=df.columns)
    test_ds = pd.DataFrame(test, columns=df.columns)

    if save_dfs:
        logger.info("Saving and storing data splits.")

        os.makedirs(Cfg.preprocessed_data_path, exist_ok=True)
        train.to_csv(os.path.join(Cfg.preprocessed_data_path, "train.csv"))
        test.to_csv(os.path.join(Cfg.preprocessed_data_path, "test.csv"))

        return train_ds, test_ds


def prepare_input(tokenizer: RobertaTokenizer, text: str) -> Dict:
    """Tokenize and prepare the input text using the provided tokenizer.

    Args:
        tokenizer (RobertaTokenizer): The Roberta tokenizer to encode the input.
        text (str): The input text to be tokenized.

    Returns:
        inputs (dict): A dictionary containing the tokenized input with keys such as 'input_ids',
            'attention_mask', etc.
    """
    inputs = tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=Cfg.add_special_tokens,
        max_length=Cfg.max_len,
        pad_to_max_length=Cfg.pad_to_max_length,
        truncation=Cfg.truncation,
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class NewsDataset(Dataset):
    def __init__(self, ds):
        self.texts = ds["Text"].values
        self.labels = ds["Category"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        inputs = prepare_input(tokenizer, self.texts[item])
        labels = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, labels


def collate(inputs: Dict) -> Dict:
    """Collate and modify the input dictionary to have the same sequence length for a particular input batch.

    Args:
        inputs (dict): A dictionary containing input tensors with varying sequence lengths.

    Returns:
        modified_inputs (dict): A modified dictionary with input tensors trimmed to have the same sequence length.
    """
    max_len = int(inputs["input_ids"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :max_len]
    return inputs


if __name__ == "__main__":
    df = load_dataset(Cfg.dataset_loc)
    df, headlines_df, class_to_index, index_to_class = preprocess(df)
    print(df)
    print(class_to_index)
    train_ds, val_ds = data_split(df, save_dfs=True)
    dataset = NewsDataset(df)
    print(dataset.__getitem__(0))
