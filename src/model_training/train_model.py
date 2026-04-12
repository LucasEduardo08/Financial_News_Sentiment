import logging
import os
import yaml
import numpy as np

import joblib
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout


logger = logging.getLogger("src.model_training.train_model")


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


def load_params() -> dict:
    """Load model training parameters from a YAML file.
    
    Returns:
        dict: A dictionary containing the model training parameters.
    """

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    logger.info("Loading parameters of training from params.yaml")
    return params["train"]


def create_model(input_dim: int, output_dim: int) -> any:
    """Create a neural network model.
    
    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output classes.
    
    Returns:
        any: A compiled Keras model.
    """
    
    logger.info("Creating the model...")
    model = Sequential()
    model.add(Dense(load_params()["hidden_layer_1_neurons"], activation='relu', input_dim=input_dim))
    model.add(Dropout(load_params()["dropout_rate"]))

    model.add(Dense(load_params()["hidden_layer_2_neurons"], activation='relu'))
    model.add(Dropout(load_params()["dropout_rate"]))

    model.add(Dense(output_dim, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def train_model(model: any, X_train: any, y_train: any, X_val: any, y_val: any) -> any:
    """Train the model on the training data and validate on the validation data.
    
    Args:
        model (any): The compiled Keras model to be trained.
        X_train (any): The training features.
        y_train (any): The training labels.
        X_val (any): The validation features.
        y_val (any): The validation labels.
    
    Returns:
        any: The trained Keras model.
    """

    logger.info("Training the model...")
    history = model.fit(
        X_train, 
        y_train, 
        epochs=load_params()["epochs"], 
        batch_size=load_params()["batch_size"], 
        validation_data=(X_val, y_val)
    )
    
    return model


def save_model(model: any) -> None:
    """Save the trained model to a file.
    
    Args:
        model (any): The trained Keras model to be saved.
    """
    logger.info("Saving the model...")
    model.save("models/trained_model.keras")


def main() -> None:
    # Load data
    X_train, y_train, X_test, y_test, X_val, y_val = load_data()

    # Create model
    model = create_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
    
    # Train model
    model = train_model(model, X_train, y_train, X_val, y_val)

    # Save model
    save_model(model)


if __name__ == "__main__":
    main()
