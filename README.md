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

1. **Clone this repo**:
   ```bash
   git clone https://github.com/your-username/movie-recommender.git
   cd movie-recommender

pip install -r requirements.txt
3. **Create a `.env` file** and add your TMDb API key:
