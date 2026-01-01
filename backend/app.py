from flask import Flask, request, jsonify, render_template, session
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

df = pd.read_csv("anime.csv")
df = df.dropna(subset=["genre"])

vectorizer = joblib.load("tfidf_vectorizer.pkl")
quality_model = joblib.load("logistic_model.pkl")
anime_vectors = vectorizer.transform(df["genre"])

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
        recommendations=session.get("recommendations")
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

# def analyze():
#     data = request.get_json()
#     genres = data.get("genres", [])

#     input_text = " ".join(genres)
#     input_vector = vectorizer.transform([input_text])

#     similarity_scores = cosine_similarity(input_vector, anime_vectors)[0]

#     df["similarity"] = similarity_scores
#     recommendations = df.sort_values(by="similarity", ascending=False).head(10)

#     quality_pred = quality_model.predict(input_vector)[0]
#     quality_probs = quality_model.predict_proba(input_vector)[0]
#     quality_confidence = float(round(max(quality_probs) * 100, 2))

#     quality_label = "Good Anime" if int(quality_pred) == 1 else "Low Rated Anime"

#     result = recommendations.to_dict(orient="records")

#     return jsonify({
#         "input_genres": genres,
#         "predicted_quality": quality_label,
#         "quality_confidence": quality_confidence,
#         "recommendations": result
#     })


# def analyze():
#     data = request.get_json()
#     genres = data.get("genres", [])

#     input_text = " ".join(genres)
#     vectorized_input = vectorizer.transform([input_text])

#     prediction = model.predict(vectorized_input)[0]
#     probs = model.predict_proba(vectorized_input)[0]
#     confidence = round(max(probs) * 100, 2)
#     print(prediction)
#     return jsonify({
#         "genre": prediction,
#         "confidence": confidence
#     })

if __name__ == "__main__":
    app.run(debug=True)