from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import numpy as np

from financialnewssentiment.app import schemas
from financialnewssentiment.app.model_loader import load_model
from financialnewssentiment.app.preprocess_text import preprocess_text


LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


app = FastAPI()
model = load_model()


@app.get("/health")
def health() -> dict:
    """
    Return status of API

    Returns:
        dict: API Status
    """
    return {
        "status": "Ok"
        }

@app.get("/model_info")
def model_info() -> dict:
    """
    Return info of model
    
    Returns:
        dict: Info of model
    """
    return {
        "model_type": "Keras",
        "version": "0.1.0",
    }

@app.post("/predict")
def predict(data: schemas.TextInput) -> dict:
    """
    Return the sentiment of data
    """
    text_preprocess = preprocess_text(data.text)

    prediction = model.predict(text_preprocess)

    predicted_class = np.argmax(prediction, axis=1)[0]

    return {
        "sentiment": LABELS[predicted_class],
        "confidence": float(np.max(prediction))
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Financial News Sentiment</title>
        <style>
            body {
                font-family: Arial;
                max-width: 700px;
                margin: auto;
                padding: 40px;
            }

            textarea {
                width: 100%;
                height: 120px;
                margin-top: 10px;
            }

            button {
                margin-top: 10px;
                padding: 10px;
                cursor: pointer;
            }

            .result {
                margin-top: 20px;
                padding: 15px;
                border: 1px solid #ccc;
            }
        </style>
    </head>
    <body>

        <h1>Financial News Sentiment API</h1>

        <h2>Health Check</h2>
        <button onclick="checkHealth()">Check API Health</button>
        <div id="health-result"></div>

        <h2>Model Info</h2>
        <button onclick="getModelInfo()">Get Model Info</button>
        <div id="model-result"></div>

        <h2>Predict Sentiment</h2>

        <textarea id="news-text" placeholder="Enter financial news text here..."></textarea>
        <br>
        <button onclick="predictSentiment()">Predict</button>

        <div class="result" id="prediction-result"></div>

        <script>
            async function checkHealth() {
                const response = await fetch('/health');
                const data = await response.json();

                document.getElementById('health-result').innerHTML =
                    `<p>Status: ${data.status}</p>`;
            }

            async function getModelInfo() {
                const response = await fetch('/model_info');
                const data = await response.json();

                document.getElementById('model-result').innerHTML =
                    `<p>Model Type: ${data.model_type}</p>
                     <p>Version: ${data.version}</p>`;
            }

            async function predictSentiment() {
                const text = document.getElementById('news-text').value;

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: text
                    })
                });

                const data = await response.json();

                document.getElementById('prediction-result').innerHTML =
                    `<h3>Prediction</h3>
                     <p><strong>Sentiment:</strong> ${data.sentiment}</p>
                     <p><strong>Confidence:</strong> ${data.confidence}</p>`;
            }
        </script>

    </body>
    </html>
    """
