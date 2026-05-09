import tensorflow as tf


MODEL_PATH = "artifacts/models/trained_model.keras"


def load_model() -> any:
    """
    Load the model

    Returns:
        any: The trained Keras model.
    """
    return tf.keras.models.load_model(MODEL_PATH)
