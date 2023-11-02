import os
from typing import Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from tqdm.auto import tqdm

import torch
from newsclassifier.config.config import Cfg, logger
from newsclassifier.data import NewsDataset, collate
from newsclassifier.models import CustomModel
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_step(test_loader: DataLoader, model) -> Tuple[np.ndarray, np.ndarray]:
    """Eval step."""
    model.eval()
    y_trues, y_preds = [], []
    with torch.inference_mode():
        for step, (inputs, labels) in tqdm(enumerate(test_loader)):
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels = labels.to(device)
            y_pred = model(inputs)
            y_trues.extend(labels.cpu().numpy())
            y_preds.extend(torch.argmax(y_pred, dim=1).cpu().numpy())
    return np.vstack(y_trues), np.vstack(y_preds)


def inference() -> None:
    """Do inference prediction."""
    logger.info("Loading inference data.")
    try:
        test_dataset = NewsDataset(os.path.join(Cfg.preprocessed_data_path, "test.csv"))
        test_loader = DataLoader(test_dataset, batch_size=Cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    except Exception as e:
        logger.error(e)

    logger.info("loading model.")
    try:
        model = CustomModel(num_classes=Cfg.num_classes)
        model.load_state_dict(torch.load(Cfg.model_path, map_location=torch.device("cpu")))
        model.to(device)
    except Exception as e:
        logger.error(e)

    y_true, y_pred = test_step(test_loader, model)

    print(
        f'Precision: {precision_score(y_true, y_pred, average="weighted")} \n Recall: {recall_score(y_true, y_pred, average="weighted")} \n F1: {f1_score(y_true, y_pred, average="weighted")} \n Accuracy: {accuracy_score(y_true, y_pred)}'
    )
