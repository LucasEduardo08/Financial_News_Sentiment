import logging
import os
import yaml
import re

import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
import scipy


logger = logging.getLogger("src.data_loading.load_data")
PATH_CSV = ["data/raw/train_stockemotions.csv", "data/raw/val_stockemotions.csv", "data/raw/test_stockemotions.csv",
            "data/raw/train_tfn.csv", "data/raw/val_tfn.csv"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The loaded datasets.
    """
    
    train_stockemotions = pd.read_csv(PATH_CSV[0])
    val_stockemotions = pd.read_csv(PATH_CSV[1])
    test_stockemotions = pd.read_csv(PATH_CSV[2])
    train_tfn = pd.read_csv(PATH_CSV[3])
    val_tfn = pd.read_csv(PATH_CSV[4])
    for _ in range(PATH_CSV.__len__()):
        logger.info(f"Loading raw data from {PATH_CSV[_]}")
    return train_stockemotions, val_stockemotions, test_stockemotions, train_tfn, val_tfn


def load_params() -> dict[str, float | int]:
    """
    Load the parameters from the YAML file.

    Returns:
        dict[str, Any]: Dict of the loaded parameters.
    """

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    logger.info("Loading parameters from params.yaml")
    return params["preprocess_data"]


def clean_text(text: str) -> str:
    """Clean the input text.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('  ', ' ')
    return text


def preprocess_data_stockemotions(
    dataset: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform preprocessing steps on dataset of stockemotions.

    Args:
        dataset (pd.DataFrame): The dataset to preprocess

    Returns:
        Tuple containing:
            pd.DataFrame: Processed training data
            pd.DataFrame: Processed test data
    """

    logger.info("Preprocessing data...")
    
    # Separate target column
    target = dataset['senti_label']
    features = dataset['processed']
    
    # Clean text data
    features_clean = features.apply(clean_text)

    # Remove stop words
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    features_stop_word = features_clean.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Replace labels
    target_encoder = target.replace({"bearish": 'negative', "bullish": 'positive'})

    return features_stop_word, target_encoder


def preprocess_data_tfn(
    dataset: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform preprocessing steps on dataset of tfn.

    Args:
        dataset (pd.DataFrame): The dataset to preprocess

    Returns:
        Tuple containing:
            pd.DataFrame: Processed training data
            pd.DataFrame: Processed test data
    """

    logger.info("Preprocessing data...")
    
    # Separate target column
    target = dataset['label']
    features = dataset['text']
    
    # Clean text data
    features_clean = features.apply(clean_text)

    # Remove stop words
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    features_stop_word = features_clean.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    return features_stop_word, target


def tf_idf_vectorization(features: pd.DataFrame) -> scipy.sparse.csr.csr_matrix:
    """Apply TF-IDF vectorization to the features.

    Args:
        features (pd.DataFrame): The input features to be vectorized.

    Returns:
        scipy.sparse.csr.csr_matrix: The TF-IDF vectorized features.
    """

    logger.info("Applying TF-IDF vectorization...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        smooth_idf=True,
        norm="l2"
    )

    # Apply TF-IDF vectorization
    dataset_tf_idf = vectorizer.fit_transform(features.squeeze().tolist())

    return dataset_tf_idf


def svd_reduction(features: scipy.sparse.csr.csr_matrix, n_components: int, random_state: int) -> np.ndarray:
    """Apply SVD dimensionality reduction to the features.

    Args:
        features (scipy.sparse.csr.csr_matrix): The input features to be reduced.
        n_components (int): The number of components to keep.
        random_state (int): The random state for reproducibility.

    Returns:
        np.ndarray: The SVD reduced features.
    """

    logger.info("Applying SVD dimensionality reduction...")

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    dataset_svd = svd.fit_transform(features)

    return dataset_svd


def encoder_sentiment(target: pd.DataFrame) -> pd.DataFrame:
    """Encode the sentiment labels.

    Args:
        target (pd.DataFrame): The input target labels to be encoded.

    Returns:
        pd.DataFrame: The encoded target labels.
    """

    logger.info("Encoding sentiment labels...")

    target_encoded = target.replace({0: "negative", 1: "positive", 2: "neutral"})

    return target_encoded


def one_hot_encoding(target: pd.DataFrame) -> pd.DataFrame:
    """Apply one-hot encoding to the target labels.

    Args:
        target (pd.DataFrame): The input target labels to be one-hot encoded.

    Returns:
        pd.DataFrame: The one-hot encoded target labels.
    """

    target_encoded = encoder_sentiment(target)

    logger.info("Applying one-hot encoding to target labels...")

    # Apply one-hot encoding
    encoder = OneHotEncoder(sparse_output=False)
    target_one_hot = encoder.fit_transform(target_encoded.values.reshape(-1, 1))

    return target_one_hot


def split_data(features: pd.DataFrame, target: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data into training and test sets.

    Args:
        features (pd.DataFrame): The input features to be split.
        target (pd.DataFrame): The input target labels to be split.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The random state for reproducibility.

    Returns:
        Tuple containing:
            pd.DataFrame: Training features
            pd.DataFrame: Test features
            pd.DataFrame: Training target labels
            pd.DataFrame: Test target labels
            pd.DataFrame: Validation features
            pd.DataFrame: Validation target labels
    """

    logger.info("Splitting data into training and test sets...")

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)

    # Split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test, X_val, y_val


def save_preprocessed_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame) -> None: 
    """Save the preprocessed data to disk.

    Args:
        X_train (pd.DataFrame): The training features to be saved.
        X_test (pd.DataFrame): The test features to be saved.
        y_train (pd.DataFrame): The training target labels to be saved.
        y_test (pd.DataFrame): The test target labels to be saved.
        X_val (pd.DataFrame): The validation features to be saved.
        y_val (pd.DataFrame): The validation target labels to be saved.
    """

    logger.info("Saving preprocessed data to disk...")

    joblib.dump(X_train, "data/preprocessed/X_train.joblib")
    joblib.dump(X_test, "data/preprocessed/X_test.joblib")
    joblib.dump(y_train, "data/preprocessed/y_train.joblib")
    joblib.dump(y_test, "data/preprocessed/y_test.joblib")
    joblib.dump(X_val, "data/preprocessed/X_val.joblib")
    joblib.dump(y_val, "data/preprocessed/y_val.joblib")


def merge_dataset(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge the datasets"""
    return pd.concat(datasets)


def main() -> None:
    # Load data
    train_stockemotions, val_stockemotions, test_stockemotions, train_tfn, val_tfn = load_data()

    # Preprocess data
    train_stockemotions_processed, train_stockemotions_target = preprocess_data_stockemotions(train_stockemotions)
    val_stockemotions_processed, val_stockemotions_target = preprocess_data_stockemotions(val_stockemotions)
    test_stockemotions_processed, test_stockemotions_target = preprocess_data_stockemotions(test_stockemotions)
    
    train_tfn_processed, train_tfn_target = preprocess_data_tfn(train_tfn)
    val_tfn_processed, val_tfn_target = preprocess_data_tfn(val_tfn)
    
    # Merge datasets
    feature_merge = merge_dataset([train_stockemotions_processed, val_stockemotions_processed, test_stockemotions_processed, train_tfn_processed, val_tfn_processed])
    target_merge = merge_dataset([train_stockemotions_target, val_stockemotions_target, test_stockemotions_target, train_tfn_target, val_tfn_target])

    # TF-IDF
    feature_tf_idf = tf_idf_vectorization(feature_merge)
    
    # SVD
    params = load_params()
    feature_svd = svd_reduction(feature_tf_idf, n_components=params["n_components"], random_state=params["random_seed"])

    # One-hot encoding
    target_one_hot = one_hot_encoding(target_merge)
    
    # Split data
    X_train, X_test, y_train, y_test, X_val, y_val = split_data(feature_svd, target_one_hot, test_size=params["test_size"], random_state=params["random_seed"])

    # Save preprocessed data
    save_preprocessed_data(X_train, X_test, y_train, y_test, X_val, y_val)


if __name__ == "__main__":
    main()
