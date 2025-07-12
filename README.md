# 🎬 Movie Recommender System

A simple content-based movie recommendation system using TF-IDF and cosine similarity. Built in Python with scikit-learn and pandas.

---

## 🚀 Features

- Content-based movie recommendations
- Fuzzy title matching to handle typos
- Recommends top-10 most similar movies
- Lightweight and easy to run locally

---

## 📦 Dataset

We use the [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) dataset and enrich it with descriptions from the TMDb API.

> ❗ Due to file size limits, CSV files are not included in this repo.

📥 **Download required CSV files from Google Drive**:  
🔗 [Click here to access]([https://your-google-drive-link.com](https://drive.google.com/drive/folders/1vgJmyNKUmcCaajBipXN3VJS7VRzieoBS?usp=sharing)) 

---
## 🛠 Installation

```bash
# 1) Clone the repo
git clone https://github.com/ArashB1230/movie-recommender.git
cd movie-recommender

# 2) Install dependencies
pip install -r requirements.txt

# 3) Add your TMDb API key
echo TMDB_API_KEY=your_tmdb_key_here > .env
