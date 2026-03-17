import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

DetectorFactory.seed = 0

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# -------------------------------
# Load models once at startup
# -------------------------------
print("🧠 Loading main abuse detection model...")
model = joblib.load("comment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
print("✅ Model and vectorizer loaded successfully!")


@app.route("/")
def home():
    return "✅ Flask backend is running! Use POST /check_comment to test comments."


def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "unknown"


def translate_to_english(text):
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text


@app.route("/check_comment", methods=["POST"])
def check_comment():
    try:
        data = request.get_json()
        comment = data.get("comment", "").strip()

        if not comment:
            return jsonify({"error": "Comment cannot be empty"}), 400

        detected_language = detect_language(comment)

        processed_comment = comment
        if detected_language not in ["en", "unknown"]:
            processed_comment = translate_to_english(comment)

        vec = vectorizer.transform([processed_comment])
        pred = model.predict(vec)[0]

        labels = {0: "Offensive", 1: "Neutral", 2: "Safe"}
        result = labels.get(pred, "Unknown")

        response = {
            "original_comment": comment,
            "detected_language": detected_language,
            "processed_comment": processed_comment,
            "result": result
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)