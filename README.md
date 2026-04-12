# Financial News Sentiment

Project to classify sentiment in financial news and market texts using an NLP pipeline and a neural network model.

## Overview

This repository contains the full pipeline to:
- download financial sentiment datasets,
- preprocess and transform text,
- train a Keras sentiment classification model,
- evaluate the model and save metrics.

## Project Structure

- `data/`
  - `raw/` — raw dataset CSV files
  - `preprocessed/` — vectorized and encoded data saved as Joblib files
- `models/` — trained model saved to `models/trained_model.keras`
- `metrics/` — evaluation results and metrics
- `src/` — pipeline source code
  - `src/data_download` — dataset download scripts
  - `src/data_preprocessing` — data cleaning, vectorization, and splitting
  - `src/model_training` — model creation, training, and saving
  - `src/model_evaluation` — model evaluation and result export
- `params.yaml` — training and preprocessing parameters
- `pyproject.toml` — Python dependencies and project metadata

## Prerequisites

Install dependencies using Poetry:

```bash
poetry install
poetry shell
```

## How to Run

### 1. Download the datasets

```bash
python3 -m src.data_download.download_data
```

### 2. Preprocess the data

```bash
python3 -m src.data_preprocessing.preprocess_data
```

This step cleans text, removes stopwords, applies TF-IDF, reduces dimensionality with SVD, and encodes labels.

### 3. Train the model

```bash
python3 -m src.model_training.train_model
```

This script loads the preprocessed data, creates and trains a Keras model, and saves the trained model to `models/trained_model.keras`.

### 4. Evaluate the model

```bash
python3 -m src.model_evaluation.evaluate_model
```

This script loads the trained model and test data, prints the classification report and confusion matrix, and saves results to `metrics/evaluation_results.json`.


## Notes

- The `financialnewssentiment/app/` directory exists but is currently empty.
- The main focus of this project is the data pipeline and model training/evaluation.
- Make sure raw data files are present in `data/raw/` before running preprocessing.
