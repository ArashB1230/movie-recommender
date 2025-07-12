import os
import pandas as pd
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Read links.csv to get the mapping between MovieLens IDs and TMDb IDs
LINKS_CSV_PATH = os.getenv("LINKS_CSV_PATH", "links.csv")
links_df = pd.read_csv(LINKS_CSV_PATH, usecols=["movieId", "tmdbId"])
links_df = links_df.dropna().astype({"movieId": "int", "tmdbId": "int"})

API_KEY = os.getenv("TMDB_API_KEY")
if not API_KEY:
    raise ValueError("TMDB_API_KEY not found. Please ensure the .env file exists and contains the TMDB_API_KEY variable.")
BASE_URL = "https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US"
descriptions = []

print("📡 Fetching movie descriptions from TMDb...")
for _, row in tqdm(links_df.iterrows(), total=len(links_df)):
    movie_id = row["movieId"]
    tmdb_id = row["tmdbId"]

    try:
        url = BASE_URL.format(tmdb_id, API_KEY)
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            overview = data.get("overview", "")
        else:
            overview = ""
    except:
        overview = ""

    descriptions.append({
        "movieId": movie_id,
        "description": overview
    })

# Save to CSV
desc_df = pd.DataFrame(descriptions)
desc_df.to_csv("movie_descriptions.csv", index=False)
print("✅ Done! Descriptions saved to movie_descriptions.csv")
