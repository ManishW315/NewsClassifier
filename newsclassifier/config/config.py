import logging
import os
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

import nltk

from rich.logging import RichHandler

# from nltk.corpus import stopwords
# nltk.download("stopwords")


@dataclass
class Cfg:
    STOPWORDS = [
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "you're",
        "you've",
        "you'll",
        "you'd",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "she's",
        "her",
        "hers",
        "herself",
        "it",
        "it's",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "that'll",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "don't",
        "should",
        "should've",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "aren't",
        "couldn",
        "couldn't",
        "didn",
        "didn't",
        "doesn",
        "doesn't",
        "hadn",
        "hadn't",
        "hasn",
        "hasn't",
        "haven",
        "haven't",
        "isn",
        "isn't",
        "ma",
        "mightn",
        "mightn't",
        "mustn",
        "mustn't",
        "needn",
        "needn't",
        "shan",
        "shan't",
        "shouldn",
        "shouldn't",
        "wasn",
        "wasn't",
        "weren",
        "weren't",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
    ]

    dataset_loc = os.path.join((Path(__file__).parent.parent.parent), "dataset", "raw", "news_dataset.csv")
    preprocessed_data_path = os.path.join((Path(__file__).parent.parent.parent), "dataset", "preprocessed")
    sweep_config_path = os.path.join((Path(__file__).parent), "sweep_config.yaml")

    # Logs path
    logs_path = os.path.join((Path(__file__).parent.parent.parent), "logs")
    artifacts_path = os.path.join((Path(__file__).parent.parent.parent), "artifacts")
    model_path = os.path.join((Path(__file__).parent.parent.parent), "artifacts", "model.pt")

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
    num_classes = 7

    sweep_run = 10


# Create logs folder
os.makedirs(Cfg.logs_path, exist_ok=True)

# Get root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create handlers
console_handler = RichHandler(markup=True)
console_handler.setLevel(logging.INFO)

info_handler = RotatingFileHandler(
    filename=Path(Cfg.logs_path, "info.log"),
    maxBytes=10485760,  # 1 MB
    backupCount=10,
)
info_handler.setLevel(logging.INFO)

error_handler = RotatingFileHandler(
    filename=Path(Cfg.logs_path, "error.log"),
    maxBytes=10485760,  # 1 MB
    backupCount=10,
)
error_handler.setLevel(logging.ERROR)

# Create formatters
minimal_formatter = logging.Formatter(fmt="%(message)s")
detailed_formatter = logging.Formatter(fmt="%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n")

# Hook it all up
console_handler.setFormatter(fmt=minimal_formatter)
info_handler.setFormatter(fmt=detailed_formatter)
error_handler.setFormatter(fmt=detailed_formatter)
logger.addHandler(hdlr=console_handler)
logger.addHandler(hdlr=info_handler)
logger.addHandler(hdlr=error_handler)
