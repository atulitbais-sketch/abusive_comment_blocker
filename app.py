import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import threading

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend access

# -------------------------------
# Load models once at startup
# -------------------------------
print("🧠 Loading main abuse detection model...")
model = joblib.load("comment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("✅ Model and vectorizer loaded successfully!")


@app.route("/check_comment", methods=["POST"])
def check_comment():
    try:
        data = request.get_json()
        comment = data.get("comment", "").strip()

        if not comment:
            return jsonify({"error": "Comment cannot be empty"}), 400

        # Predict abuse level (Offensive / Neutral / Safe)
        vec = vectorizer.transform([comment])
        pred = model.predict(vec)[0]
        labels = {0: "Offensive", 1: "Neutral", 2: "Safe"}
        result = labels.get(pred, "Unknown")

        # Combine results
        response = {
            "comment": comment,
            "result": result
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    threading.Thread(target=run_flask).start()
