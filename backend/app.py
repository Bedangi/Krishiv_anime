from flask import Flask, request, jsonify, render_template, session
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import secrets
import nltk
import os

# -------------------------------
# Base directory (backend/)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# NLTK setup (Render-safe)
# -------------------------------
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

nltk.data.path.append(NLTK_DATA_DIR)
nltk.download("wordnet", download_dir=NLTK_DATA_DIR)
nltk.download("omw-1.4", download_dir=NLTK_DATA_DIR)

# -------------------------------
# Flask app
# -------------------------------
app = Flask(__name__, template_folder="templates")
app.secret_key = secrets.token_hex(32)

# -------------------------------
# Load data & models (SAFE PATHS)
# -------------------------------
csv_path = os.path.join(BASE_DIR, "anime.csv")
vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
model_path = os.path.join(BASE_DIR, "logistic_model.pkl")

df = pd.read_csv(csv_path)
df = df.dropna(subset=["genre"])

vectorizer = joblib.load(vectorizer_path)
quality_model = joblib.load(model_path)

anime_vectors = vectorizer.transform(df["genre"])

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/result")
def result():
    return render_template(
        "index_output.html",
        predicted_quality=session.get("predicted_quality"),
        quality_confidence=session.get("quality_confidence"),
        anime=session.get("top_anime"),
        recommendations=session.get("recommendations"),
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    genres = data.get("genres", [])

    input_text = " ".join(genres)
    input_vector = vectorizer.transform([input_text])

    similarity_scores = cosine_similarity(input_vector, anime_vectors)[0]

    df["similarity"] = similarity_scores
    recommendations = df.sort_values(by="similarity", ascending=False).head(10)

    quality_pred = quality_model.predict(input_vector)[0]
    quality_probs = quality_model.predict_proba(input_vector)[0]
    quality_confidence = float(round(max(quality_probs) * 100, 2))

    quality_label = "Good Anime" if int(quality_pred) == 1 else "Low Rated Anime"
    top_anime = recommendations.iloc[0].to_dict()

    session["predicted_quality"] = quality_label
    session["quality_confidence"] = quality_confidence
    session["top_anime"] = top_anime
    session["recommendations"] = recommendations.to_dict(orient="records")

    return jsonify({"ok": True})


# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
