import joblib
from nltk.corpus import stopwords
from src.data_preprocessing.preprocess_data import clean_text


TFIDF_PATH = "artifacts/models/tfidf_vectorizer.pkl"
SVD_PATH = "artifacts/models/svd_model.pkl"


vectorizer = joblib.load(TFIDF_PATH)
svd = joblib.load(SVD_PATH)
stop_words = stopwords.words('english')


def preprocess_text(text: str):
    """
    Pipeline to processing text before to give the model
    """

    # Clean text
    text = clean_text(text)

    text_stop_word = ' '.join(
            [word for word in text.split() if word not in stop_words]
    )

    # TF-IDF transform
    text_tfidf = vectorizer.transform([text_stop_word])

    # SVD transform
    text_svd = svd.transform(text_tfidf)

    return text_svd
