import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



nltk.download("wordnet")

def preprocess_text(text_series):
    lemmatizer = WordNetLemmatizer()

    def process(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    return text_series.dropna().apply(process)

def extract_tfidf_features(text_series, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(text_series)
    return features, vectorizer

df = pd.read_csv("anime.csv")
df = df.dropna(subset=["genre", "rating"])

df["label"] = df["rating"].apply(lambda x: 1 if x >= 7.5 else 0)

processed_text = preprocess_text(df["genre"])

X, vectorizer = extract_tfidf_features(processed_text)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("TF-IDF Feature Shape:", X.shape)
print("Model Used: Logistic Regression")
print("Accuracy:", round(accuracy * 100, 2), "%")

