import os
import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS

# Flask App Setup — Serve static + template from correct paths
app = Flask(
    __name__,
    static_folder="../static",         # JS, CSS are here
    template_folder="../templates"     # HTML is here
)
CORS(app)

# Load paths and port from environment (Render-safe)
MODEL_PATH = os.getenv("MODEL_PATH", "diabetes_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
PORT = int(os.getenv("PORT", 10000))  # Render sets $PORT

# Load model and scaler
try:
    model = joblib.load(os.path.join(os.path.dirname(__file__), MODEL_PATH))
    scaler = joblib.load(os.path.join(os.path.dirname(__file__), SCALER_PATH))
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    model = None
    scaler = None

# ==========================
# Routes
# ==========================

@app.route("/")
def home():
    return render_template("DiabetesPrediction.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request"}), 400

    try:
        features = np.array(data["features"]).reshape(1, -1)
        scaled = scaler.transform(features)
        prediction = int(model.predict(scaled)[0])
        probability = round(model.predict_proba(scaled)[0][1], 2)

        return jsonify({
            "prediction": prediction,
            "probability": probability
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

# ==========================
# Entry Point
# ==========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
