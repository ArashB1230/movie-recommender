# ==============================================================
#   Movie Recommender (Content-Based, TF-IDF)
#   ------------------------------------------------------------
#   • Loads movie metadata + descriptions
#   • Builds a TF-IDF matrix and cosine-similarity lookup
#   • Provides fuzzy title matching and Top-N recommendations
#   • Requires a .env file with TMDB_API_KEY (NOT committed!)
# ==============================================================

# ------------------ Imports & Config ------------------
import os
from difflib import get_close_matches

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1) Load API key (never print it) -----------------
load_dotenv()                                   # read .env file
API_KEY = os.getenv("TMDB_API_KEY")             # get key

if not API_KEY or API_KEY.strip() in ("your_tmdb_key_here",
                                      "PLACEHOLDER", ""):
    print("❌  TMDB_API_KEY not set.  Create a .env file with your own key.")
    exit(1)

# ------------------ 2) Load & Merge -------------------
try:
    movies = pd.read_csv("movies.csv")                # movieId, title, genres
    descs  = pd.read_csv("movie_descriptions.csv")    # movieId, description
except FileNotFoundError as e:
    print(f"❌  File not found: {e.filename}")
    exit(1)

data = (
    movies.merge(descs, on="movieId", how="inner")    # join on movieId
          .dropna(subset=["description"])             # keep rows with text
          .reset_index(drop=True)
)

# ------------------ 3) TF-IDF Model ------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(data["description"])

# cosine-similarity as sparse matrix (memory-efficient)
cosine_sim = cosine_similarity(tfidf_matrix, dense_output=False)

# map lowercase title → row index (duplicates removed)
title_to_idx = (
    pd.Series(data.index, index=data["title"].str.lower())
      .drop_duplicates(keep="first")
)

# ------------------ 4) Recommendation ----------------
def recommend(title: str, n: int = 10):
    """
    Return a list with *n* titles most similar to *title* (exclude itself).
    """
    idx = title_to_idx.get(title.lower())
    if idx is None:
        return f"❌  '{title}' not found."
    sims = cosine_sim[idx].toarray().ravel()     # 1×N vector
    best = np.argsort(-sims)[1:n + 1]            # skip itself (rank 0)
    return data["title"].iloc[best].tolist()

# ------------------ 5) Fuzzy Title Finder ------------
def find_title(user_input: str,
               title_series: pd.Series,
               k: int = 5):
    """
    Map free-text *user_input* to an existing movie title.

    Returns:
    • exact title  → str
    • multiple suggestions → list[str]
    • not found → None
    """
    cleaned = user_input.strip().lower()
    titles_lower = title_series.str.lower()

    # 1) exact match
    if cleaned in titles_lower.values:
        idx = titles_lower[titles_lower == cleaned].index[0]
        return title_series.iloc[idx]

    # 2) starts-with / contains
    shortlist = title_series[
        titles_lower.str.startswith(cleaned) |
        titles_lower.str.contains(cleaned)
    ]
    if len(shortlist) == 1:
        return shortlist.iloc[0]
    if 1 < len(shortlist) <= k:
        return shortlist.tolist()

    # 3) fuzzy (Levenshtein)
    close = get_close_matches(cleaned, titles_lower.values, n=k, cutoff=0.6)
    if close:
        return title_series[titles_lower.isin(close)].tolist()

    return None

# ------------------ 6) CLI Entry-Point ---------------
if __name__ == "__main__":
    print("🎬  TF-IDF Recommender ready.")
    user_in = input("Enter a movie title: ")

    match = find_title(user_in, data["title"])

    if match is None:
        print("❌  No similar title found.")
    elif isinstance(match, list):
        print("Did you mean:")
        for t in match:
            print(" •", t)
    else:
        print(f"\nTop recommendations for → {match}\n")
        for rec in recommend(match, 10):
            print(" •", rec)
