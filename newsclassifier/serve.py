import os
from typing import List

# Import your model and required libraries
import torch
from fastapi import FastAPI
from newsclassifier.config.config import Cfg
from newsclassifier.models import CustomModel
from pydantic import BaseModel
from transformers import RobertaTokenizer

app = FastAPI()

# Load the pre-trained model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = CustomModel(num_classes=7)
model.load_state_dict(torch.load(os.path.join(Cfg.artifacts_path, "model.pt"), map_location=torch.device("cpu")))
model.eval()


class InputText(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    class_index: int
    class_name: str
    probabilities: List[float]


@app.post("/predict/", response_model=PredictionResponse)
def predict_text(input_data: InputText):
    # Tokenize the input text
    inputs = prepare_input(tokenizer, input_data.text)

    # Perform inference
    with torch.no_grad():
        y_probs = model.predict_proba(inputs)
        class_index = int(torch.argmax(y_probs, dim=1))
        class_name = index_to_class[class_index]

    return PredictionResponse(class_index=class_index, class_name=class_name, probabilities=y_probs[0].tolist())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
