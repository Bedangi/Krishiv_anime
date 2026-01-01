import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def load_and_prepare_data(path):
    df = pd.read_csv(path)
    genres = df["genre"].dropna().str.lower().str.split(", ")
    return df, genres

def plot_genre_frequency(genres, top_n=10):
    all_genres = [g for sublist in genres for g in sublist]
    freq = Counter(all_genres)
    top = dict(freq.most_common(top_n))

    plt.figure(figsize=(8,5))
    plt.barh(list(top.keys()), list(top.values()))
    plt.xlabel("Frequency")
    plt.ylabel("Genre")
    plt.title("Top 10 Most Frequent Anime Genres")
    plt.gca().invert_yaxis()
    plt.show()

def plot_stopword_like_genres(genres):
    all_genres = [g for sublist in genres for g in sublist]
    freq = Counter(all_genres)
    top5 = dict(freq.most_common(5))
    others = sum(freq.values()) - sum(top5.values())

    plt.figure(figsize=(6,4))
    plt.bar(["Top 5 Genres", "Other Genres"], [sum(top5.values()), others])
    plt.xlabel("Genre Groups")
    plt.ylabel("Total Occurrences")
    plt.title("Stopword-like Genre Dominance")
    plt.show()

def plot_genre_count_distribution(genres):
    genre_counts = genres.apply(len)

    plt.figure(figsize=(8,5))
    plt.hist(genre_counts, bins=10)
    plt.xlabel("Number of Genres per Anime")
    plt.ylabel("Number of Anime")
    plt.title("Distribution of Genre Count per Anime")
    plt.show()

def plot_genre_count_vs_rating(df):
    df = df.dropna(subset=["rating", "genre"])
    df["genre_count"] = df["genre"].str.split(", ").apply(len)

    high = df[df["rating"] >= 7.5]["genre_count"]
    low = df[df["rating"] < 7.5]["genre_count"]

    plt.figure(figsize=(8,5))
    plt.hist(high, bins=10, alpha=0.7, label="High Rated Anime")
    plt.hist(low, bins=10, alpha=0.7, label="Low Rated Anime")
    plt.xlabel("Number of Genres")
    plt.ylabel("Number of Anime")
    plt.title("Genre Count Distribution by Rating Category")
    plt.legend()
    plt.show()

df, genres = load_and_prepare_data("anime.csv")
plot_genre_frequency(genres)
plot_stopword_like_genres(genres)
plot_genre_count_distribution(genres)
plot_genre_count_vs_rating(df)