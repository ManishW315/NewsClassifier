---
title: News-Classifier
app_file: app.py
sdk: gradio
sdk_version: 4.0.2
---
# NewsClassifier

## Overview
News Classifier project implemented finetuning Roberta-base transform on a text classification task. The goal of this project is to classify news articles into predefined categories, such as sports, politics, technology, and entertainment.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](README.md#training)
- [Tune](#tune)
- [Inference](#inference)
- [Results](#results)

## Project Structure
The project is organized as follows:

Displaying only the core project files.

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

4. Customize the config.yml file to set hyperparameters, data paths, and other configuration options.

See docs here: [NewsClassifier Docs](https://ManishW315.github.io/NewsClassifier/)

## Data Preprocessing
Data preprocessing is essential for preparing the dataset for training. You can use the provided Jupyter notebooks (`data_preprocessing.ipynb`) or the Python scripts in the `src/` directory for data preprocessing.

## Model Architecture
The deep learning model architecture is defined in `src/model.py`.

## Training
To train the model, run the following command:

```bash
python newsclassifier/train.py
```
This script will train the model using the specified hyperparameters and save the trained model checkpoints in the models/ directory.

## Tune
To perform hyperparameter tuning, run the following command. The hyperparameter search range can be changed:

```bash
python newsclassifier/tune.py
```
This script will provide evaluation metrics such as accuracy, precision, recall, and F1-score on a test dataset.

## Inference
You can use the trained model for inference by loading it and making predictions on new news articles. Example code for inference can be found in the provided Jupyter notebooks or in the evaluate.py script.

## Results
Share the results and insights obtained from the trained model, including accuracy, loss curves, and any additional observations.
