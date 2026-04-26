import logging
import json
import os

import joblib
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


logger = logging.getLogger("src.model_evaluation.evaluate_model")


def load_model() -> keras.Model:
    """Load the trained model from a file.
    
    Returns:
        keras.Model: The loaded Keras model.
    """
    logger.info("Loading the trained model...")
    model = tf.keras.models.load_model("artifacts/models/trained_model.keras")
    return model


def load_data() -> tuple[any]:
    """Load preprocessed data from CSV files.
    
    Returns:
        tuple: A tuple containing the preprocessed training, testing, and validation data.
    """
    logger.info("Loading preprocessed data...")
    X_train = joblib.load("data/preprocessed/X_train.joblib")
    y_train = joblib.load("data/preprocessed/y_train.joblib")
    X_test = joblib.load("data/preprocessed/X_test.joblib")
    y_test = joblib.load("data/preprocessed/y_test.joblib")
    X_val = joblib.load("data/preprocessed/X_val.joblib")
    y_val = joblib.load("data/preprocessed/y_val.joblib")

    return X_train, y_train, X_test, y_test, X_val, y_val


def evaluate_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Evaluate the model on the test set and print the classification report and confusion matrix.
    
    Args:
        model (keras.Model): The trained Keras model to evaluate.
        X_test (np.ndarray): The test set features.
        y_test (np.ndarray): The test set labels.
    """

    sentiment = ['negative', 'neutral', 'positive']

    logger.info("Evaluating the model...")

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_pred_decoder = np.array([sentiment[i] for i in y_pred])
    y_test = np.array([sentiment[i] for i in np.argmax(y_test, axis=1)])

    cm = confusion_matrix(y_test, y_pred_decoder, labels=sentiment)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in sentiment],
        columns=[f"pred_{label}" for label in sentiment],
    )

    return classification_report(y_test, y_pred_decoder), cm_df


def save_evaluation_results(classification_report: str, confusion_matrix: pd.DataFrame) -> None:
    """Save the evaluation results to a JSON file.
    
    Args:
        classification_report (str): The classification report as a string.
        confusion_matrix (pd.DataFrame): The confusion matrix as a DataFrame.
    """
    logger.info("Saving evaluation results...")
    results = {
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix.to_dict(),
    }
    with open("artifacts/metrics/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)


def main() -> None:
    # Load model and data
    model = load_model()
    _, _, X_test, y_test, _, _ = load_data()

    # Evaluate model
    classification_report, confusion_matrix = evaluate_model(model, X_test, y_test)
    print("Classification Report:\n", classification_report)
    print("Confusion Matrix:\n", confusion_matrix)
    save_evaluation_results(classification_report, confusion_matrix)


if __name__ == "__main__":
    main()
