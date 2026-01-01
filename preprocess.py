import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("wordnet")

def show_sample(title, series, n=5):
    print("\n" + title)
    for i, text in enumerate(series.head(n), 1):
        print(f"{i}. {text}")

def clean_and_lowercase(text_series):
    cleaned = text_series.dropna().str.lower().apply(
        lambda x: re.sub(r"[^a-zA-Z,\s]", "", x)
    )
    show_sample("After Lowercasing & Cleaning:", cleaned)
    return cleaned

def remove_punctuation_and_stopwords(text_series):
    stopwords = set(ENGLISH_STOP_WORDS)

    def process(text):
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in stopwords]
        return " ".join(tokens)

    processed = text_series.apply(process)
    show_sample("After Punctuation & Stopword Removal:", processed)
    return processed

def tokenize_and_lemmatize(text_series):
    lemmatizer = WordNetLemmatizer()

    def process(text):
        tokens = text.replace(",", " ").split()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    lemmatized = text_series.apply(process)
    show_sample("After Tokenization & Lemmatization:", lemmatized)
    return lemmatized

def convert_to_tfidf(text_series, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(text_series)

    print("\nTF-IDF Feature Matrix Shape:", features.shape)
    print("\nTop 10 TF-IDF Features:")
    for f in vectorizer.get_feature_names_out()[:10]:
        print(f)

    return features, vectorizer

df = pd.read_csv("anime.csv")

print("Original Genre Samples:")
show_sample("Raw Genre Text:", df["genre"])

text_data = df["genre"]

text_data = clean_and_lowercase(text_data)
text_data = remove_punctuation_and_stopwords(text_data)
text_data = tokenize_and_lemmatize(text_data)

tfidf_features, tfidf_vectorizer = convert_to_tfidf(text_data)
