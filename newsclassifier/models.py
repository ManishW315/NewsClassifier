import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


class CustomModel(nn.Module):
    def __init__(self, num_classes, change_config=False, dropout_pb=0.0):
        super(CustomModel, self).__init__()
        if change_config:
            pass
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.hidden_size = self.model.config.hidden_size
        self.num_classes = num_classes
        self.dropout_pb = dropout_pb
        self.dropout = torch.nn.Dropout(self.dropout_pb)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, inputs):
        output = self.model(**inputs)
        z = self.dropout(output[1])
        z = self.fc(z)
        return z

    @torch.inference_mode()
    def predict(self, inputs):
        self.eval()
        z = self(inputs)
        y_pred = torch.argmax(z, dim=1).cpu().numpy()
        return y_pred

    @torch.inference_mode()
    def predict_proba(self, inputs):
        self.eval()
        z = self(inputs)
        y_probs = F.softmax(z, dim=1).cpu().numpy()
        return y_probs

    def save(self, dp):
        with open(Path(dp, "args.json"), "w") as fp:
            contents = {
                "dropout_pb": self.dropout_pb,
                "hidden_size": self.hidden_size,
                "num_classes": self.num_classes,
            }
            json.dump(contents, fp, indent=4, sort_keys=False)
        torch.save(self.state_dict(), os.path.join(dp, "model.pt"))

    @classmethod
    def load(cls, args_fp, state_dict_fp):
        with open(args_fp, "r") as fp:
            kwargs = json.load(fp=fp)
        llm = RobertaModel.from_pretrained("roberta-base")
        model = cls(llm=llm, **kwargs)
        model.load_state_dict(torch.load(state_dict_fp, map_location=torch.device("cpu")))
        return model
