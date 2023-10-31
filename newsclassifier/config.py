import os
from dataclasses import dataclass
from pathlib import Path

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")


@dataclass
class Cfg:
    STOPWORDS = stopwords.words("english")

    dataset_loc = os.path.join((Path(__file__).parent.parent), "dataset", "raw", "news_dataset.csv")
    preprocessed_data_path = os.path.join((Path(__file__).parent.parent), "dataset", "preprocessed")

    test_size = 0.2

    add_special_tokens = True
    max_len = 50
    pad_to_max_length = True
    truncation = True

    change_config = False

    dropout_pb = 0.5
    lr = 1e-4
    lr_redfactor = 0.7
    lr_redpatience = 4
    epochs = 10
    batch_size = 128
