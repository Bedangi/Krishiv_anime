import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

precision = precision_score(y_test, test_pred)
recall = recall_score(y_test, test_pred)
f1 = f1_score(y_test, test_pred)
cm = confusion_matrix(y_test, test_pred)

print("Training Accuracy:", round(train_acc * 100, 2), "%")
print("Validation Accuracy:", round(val_acc * 100, 2), "%")
print("Test Accuracy:", round(test_acc * 100, 2), "%")
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1-Score:", round(f1, 3))
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6,4))
plt.plot(["Training", "Validation"], [train_acc, val_acc], marker="o")
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.show()

misclassified = df_test[test_pred != y_test]
print("\nSample Misclassified Anime:")
print(misclassified[["name", "genre", "rating"]].head())
