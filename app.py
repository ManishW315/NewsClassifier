import os

import gradio as gr
import torch
from newsclassifier.config.config import Cfg, logger
from newsclassifier.data import prepare_input
from newsclassifier.models import CustomModel
from transformers import RobertaTokenizer

labels = list(Cfg.index_to_class.values())

# load and compile the model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = CustomModel(num_classes=7)
model.load_state_dict(torch.load(os.path.join(Cfg.artifacts_path, "model.pt"), map_location=torch.device("cpu")))


def prediction(text):
    sample_input = prepare_input(tokenizer, text)
    input_ids = torch.unsqueeze(sample_input["input_ids"], 0).to("cpu")
    attention_masks = torch.unsqueeze(sample_input["attention_mask"], 0).to("cpu")
    test_sample = dict(input_ids=input_ids, attention_mask=attention_masks)

    with torch.no_grad():
        y_pred_test_sample = model.predict_proba(test_sample)
        pred_probs = y_pred_test_sample[0]

    return {labels[i]: float(pred_probs[i]) for i in range(len(labels))}


title = "NewsClassifier"
description = "Enter a news headline, and this app will classify it into one of the categories."
instructions = "Type or paste a news headline in the textbox and press Enter."

iface = gr.Interface(
    fn=prediction,
    inputs=gr.Textbox(),
    outputs=gr.Label(num_top_classes=7),
    title=title,
    description=description,
    examples=[
        ["Global Smartphone Shipments Will Hit Lowest Point in a Decade, IDC Says"],
        ["John Wick's First Spinoff is the Rare Prequel That Justifies Its Existence"],
        ["Research provides a better understanding of how light stimulates the brain"],
        ["Lionel Messi scores free kick golazo for Argentina in World Cup qualifiers"],
    ],
    article=instructions,
)

iface.launch(share=True)
