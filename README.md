# NewsClassifier

## Overview
News Classifier project implemented by finetuning Roberta-base Transform on a text classification task. The goal of this project is to classify news articles into Business, Entertainment, Health, Science, Sports, Technology, and WorldWide categories.

## Table of Contents
- [App](#app)
- [Docs](#docs)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [Training](#training)
- [Tune](#tune)
- [Inference](#inference)

## App
**The app is created using Gradio and deployed on [Hugging Face Spaces](https://huggingface.co/spaces/ManishW/News-Classifier)**

| title Value | app_file | sdk | sdk_version |
| ---| ---| ---| ---|
| News-Classifier | app.py | gradio | 4.0.2 |



## Docs
**See docs for here: [NewsClassifier Docs](https://ManishW315.github.io/NewsClassifier/)**

---

## Project Structure
The project is organized as follows:

*Core project files:*

<pre>
NewsClassifier
│   app.py
│
├───dataset
│   ├───preprocessed
│   │       test.csv
│   │       train.csv
│   │
│   └───raw
│           news_dataset.csv
│
├───newsclassifier
│   │   data.py
│   │   models.py
│   │   train.py
│   │   tune.py
│   │   inference.py
│   │   predict.py
│   │   utils.py
│   │
│   └───config
│           config.py
│           sweep_config.yaml
│
└───notebooks
        eda.ipynb
        newsclassifier-roberta-base-wandb-track-sweep.ipynb
</pre>
  
## Dataset
The dataset is obtained from [Kaggle](https://www.kaggle.com/datasets/crxxom/daily-google-news). The dataset is split into train and test sets after preprocessing, and are stored in the ``preprocessed/`` folder.

## Setup
To set up the project environment, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/news-classifier.git
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
  ```bash
  pip install -r requirements.txt
```

## Training
To train the model, run the following command:

```bash
python newsclassifier/train.py
```
This script will train the model using the specified hyperparameters and save the trained model checkpoints in the models/ directory.

**We can change the training parameters in ``config.py``**

## Tune
To perform hyperparameter tuning, run the following command:

```bash
python newsclassifier/tune.py
```
This script will provide evaluation metrics such as accuracy, precision, recall, and F1-score on a test dataset.

**The hyperparameter search range can be changed in ``sweep_config.yaml``.**

## Inference
You can use the trained model for inference by loading it and making predictions on new news articles. Example code for inference can be found in the provided Jupyter notebooks.
