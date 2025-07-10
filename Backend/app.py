import os
import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS

app = Flask(
    __name__,
    static_folder="../Frontend",         # Path to static files (JS, CSS, etc.)
    template_folder="../Frontend"        # Path to HTML files
)
CORS(app)

# Load config from environment or default values
MODEL_PATH = os.getenv("MODEL_PATH", "diabetes_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
PORT = int(os.getenv("FLASK_PORT", 10000))

# Load model and scaler
try:
    model = joblib.load(os.path.join(os.path.dirname(__file__), MODEL_PATH))
    scaler = joblib.load(os.path.join(os.path.dirname(__file__), SCALER_PATH))
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# ======================
# Routes
# ======================

@app.route("/")
def home():
    """Serve the main HTML page"""
    return render_template("DiabetesPrediction.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request"}), 400

    features = np.array(data["features"]).reshape(1, -1)
    scaled = scaler.transform(features)
    prediction = int(model.predict(scaled)[0])
    probability = round(model.predict_proba(scaled)[0][1], 2)

    return jsonify({
        "prediction": prediction,
        "probability": probability
    })


@app.route("/<path:path>")
def static_files(path):
    """Serve JS, CSS, and other static assets"""
    return send_from_directory(app.static_folder, path)


@app.route("/health")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "diabetes-prediction",
        "model_loaded": model is not None
    })

# ======================
# Run the Flask App
# ======================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=PORT)
