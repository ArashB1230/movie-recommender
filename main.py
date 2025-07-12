# ------------------ Imports & Config ------------------
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")
if not API_KEY:
    print("❌ TMDB_API_KEY not found in .env"); exit()

# ------------------ Load & Merge ----------------------
try:
    movies = pd.read_csv("movies.csv")
    descs  = pd.read_csv("movie_descriptions.csv")
except FileNotFoundError as e:
    print(f"❌ File not found: {e.filename}"); exit(1)

data = movies.merge(descs, on="movieId", how="inner") \
             .dropna(subset=["description"]) \
             .reset_index(drop=True)

# ------------------ TF-IDF ----------------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(data["description"])
cosine_sim   = cosine_similarity(tfidf_matrix, dense_output=False)

title_to_idx = (
    pd.Series(data.index, index=data["title"].str.lower())
      .drop_duplicates(keep="first")
)

# ------------------ Recommender -----------------------
def recommend(title: str, n: int = 10):
    idx = title_to_idx.get(title.lower())
    if idx is None:
        return f"❌ '{title}' not found."
    sims = cosine_sim[idx].toarray().ravel()
    best = np.argsort(-sims)[1:n+1]
    return data["title"].iloc[best].tolist()

# ------------------ Smart title matcher --------------
def find_title(user_input: str, title_series: pd.Series, k: int = 5):
    cleaned = user_input.strip().lower()
    titles_lower = title_series.str.lower()

    # 1) exact match
    if cleaned in titles_lower.values:
        idx = titles_lower[titles_lower == cleaned].index[0]
        return title_series.iloc[idx]

    # 2) starts-with / contains
    shortlist = title_series[
        titles_lower.str.startswith(cleaned) | titles_lower.str.contains(cleaned)
    ]
    if len(shortlist) == 1:
        return shortlist.iloc[0]
    if 1 < len(shortlist) <= k:
        return shortlist.tolist()

    # 3) fuzzy
    close = get_close_matches(cleaned, titles_lower.values, n=k, cutoff=0.6)
    if close:
        return title_series[titles_lower.isin(close)].tolist()

    return None

# ------------------ Main loop ------------------------
if __name__ == "__main__":
    print("🎬 TF-IDF recommender ready.")
    user_in = input("Enter a movie title: ")
    match = find_title(user_in, data["title"])

    if match is None:
        print("❌ No similar title found.")
    elif isinstance(match, list):
        print("Did you mean:")
        for t in match: print(" •", t)
    else:
        print(f"\nTop recommendations for → {match}\n")
        for rec in recommend(match, 10):
            print(" •", rec)

