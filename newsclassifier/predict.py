import os
from typing import List

import torch
from newsclassifier.config.config import Cfg, logger
from newsclassifier.data import clean_text, prepare_input
from newsclassifier.models import CustomModel
from transformers import RobertaTokenizer


def predict(text: str) -> List:
    """Predict target for user input.

    Args:
        text (str): User input in the form of string.

    Returns:
        List: Prediction probabilities of each class labels.
    """
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = CustomModel(num_classes=7)
    model.load_state_dict(torch.load(os.path.join(Cfg.artifacts_path, "model.pt"), map_location=torch.device("cpu")))
    sample_input = prepare_input(tokenizer, text)
    input_ids = torch.unsqueeze(sample_input["input_ids"], 0).to("cpu")
    attention_masks = torch.unsqueeze(sample_input["attention_mask"], 0).to("cpu")
    test_sample = dict(input_ids=input_ids, attention_mask=attention_masks)

    with torch.no_grad():
        logger.info("Predicting labels.")
        y_pred_test_sample = model.predict_proba(test_sample)
        prediction = y_pred_test_sample[0]

    return prediction


if __name__ == "__main__":
    txt = clean_text("Funds punished for owning too few Nvidia")
    pred_prob = predict(txt)
    print(pred_prob)
