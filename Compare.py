import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

df = pd.read_csv("anime.csv")
df = df.dropna(subset=["genre", "rating"])

df["label"] = df["rating"].apply(lambda x: 1 if x >= 7.5 else 0)

processed_text = preprocess_text(df["genre"])

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(processed_text)
y = df["label"]

X_train, X_temp, y_train, y_temp, df_train, df_temp = train_test_split(
    X, y, df, test_size=0.15, random_state=42
)

X_val, X_test, y_val, y_test, df_val, df_test = train_test_split(
    X_temp, y_temp, df_temp, test_size=0.5, random_state=42
)

lr_model = LogisticRegression(max_iter=1000)
svm_model = SVC(kernel="linear")

lr_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Performance")
    print("Accuracy:", round(accuracy_score(y_true, y_pred) * 100, 2), "%")
    print("Precision:", round(precision_score(y_true, y_pred), 3))
    print("Recall:", round(recall_score(y_true, y_pred), 3))
    print("F1-Score:", round(f1_score(y_true, y_pred), 3))
    return confusion_matrix(y_true, y_pred)

lr_cm = evaluate_model("Logistic Regression", y_test, lr_pred)
svm_cm = evaluate_model("SVM", y_test, svm_pred)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(lr_cm)
plt.title("LR Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(svm_cm)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

plt.tight_layout()
plt.show()

misclassified = df_test[lr_pred != y_test]
print("\nSample Misclassified Anime (Logistic Regression):")
print(misclassified[["name", "genre", "rating"]].head())