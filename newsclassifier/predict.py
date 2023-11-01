import os

import numpy as np

import torch
from newsclassifier.config.config import Cfg, logger
from newsclassifier.data import clean_text, prepare_input
from newsclassifier.models import CustomModel
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = CustomModel(num_classes=7)
model.load_state_dict(torch.load(os.path.join(Cfg.artifacts_path, "model.pt"), map_location=torch.device("cpu")))


def predict(text: str):
    index_to_class = Cfg.index_to_class
    sample_input = prepare_input(tokenizer, text)
    input_ids = torch.unsqueeze(sample_input["input_ids"], 0).to("cpu")
    attention_masks = torch.unsqueeze(sample_input["attention_mask"], 0).to("cpu")
    test_sample = dict(input_ids=input_ids, attention_mask=attention_masks)

    with torch.no_grad():
        y_pred_test_sample = model.predict_proba(test_sample)
        print(index_to_class[np.argmax(y_pred_test_sample)])


if __name__ == "__main__":
    txt = clean_text("Funds punished for owning too few Nvidia")
    predict(txt)
