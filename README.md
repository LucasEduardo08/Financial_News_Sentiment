# Financial News Sentiment

An end-to-end MLOps pipeline to classify sentiment (Positive, Neutral, Negative) in financial news and market texts using Deep Learning.

## Overview

This project implements a complete machine learning lifecycle:
- **Data Ingestion**: Automated downloading of financial datasets.
- **NLP Pipeline**: Text cleaning, TF-IDF vectorization, and SVD dimensionality reduction.
- **Model Training**: Multi-layer Perceptron (MLP) built with Keras/TensorFlow.
- **Experiment Tracking**: Full integration with **MLflow** for metrics and artifact logging.
- **Version Control**: Data and pipeline orchestration using **DVC**.
- **Deployment**: High-performance API using **FastAPI**.

## Project Structure
```text
.
├── artifacts/                # Model artifacts and metrics
│   ├── metrics/              # Confusion matrix and JSON reports
│   └── models/               # Saved Keras models (.keras)
├── data/                     # Data storage
│   ├── preprocessed/         # Vectorized data (Joblib files)
│   └── raw/                  # Raw CSV datasets
├── financialnewssentiment/   # Application source code
│   └── app/                  # FastAPI application (schemas, main)
├── src/                      # ML pipeline source code
│   ├── data_download/        # Scripts for data ingestion
│   ├── data_preprocessing/   # Cleaning and transformation logic
│   ├── model_evaluation/     # Model performance analysis
│   └── model_training/       # Training and architecture definitions
├── params.yaml               # Pipeline hyperparameters
└── pyproject.toml            # Poetry dependencies and metadata
```

## Prerequisites

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

If you are on Windows, use: 
```bash
poetry env activate
 ```
And then copy the results of command on the terminal. On Windows, the command above fails:
```bash
poetry shell
```

## How to Run?

### 1. To run pipeline with dvc

```bash
dvc repro
```

### 2. To run the Fastapi aplication

```bash
fastapi dev financialnewssentiment/app/main.py
```

### 3. To run the tests using pytest (Optional, this step is to analyse the working of the system)

```bash
pytest --cov=financialnewssentiment/app
```


## Notes

- The main focus of this project is the data pipeline and model training/evaluation.
- Make sure raw data files are present in `data/raw/` before running preprocessing.
