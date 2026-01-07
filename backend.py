from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load data and model
df = pd.read_csv("data/movies_merged.csv")
with open("utils/model.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare genre matrix
df["genres_list"] = df["genres"].str.split("|")
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df.drop_duplicates("movieId")["genres_list"])

movie_ids = df.drop_duplicates("movieId")["movieId"].values
movie_genres = {
    movie_id: genre_matrix[i]
    for i, movie_id in enumerate(movie_ids)
}

def genre_similarity(candidate_movie_id, rated_movies):
    if len(rated_movies) == 0:
        return 0
    candidate_vec = movie_genres[candidate_movie_id].reshape(1, -1)
    rated_vecs = np.array([movie_genres[m] for m in rated_movies])
    return cosine_similarity(candidate_vec, rated_vecs).mean()

def hybrid_recommend_adaptive(model, df, user_id, user_ratings, k=10):
    all_movies = df["movieId"].unique()
    rated_movies = np.array(list(user_ratings.keys()))
    n_ratings = len(rated_movies)
    
    movies_to_predict = np.setdiff1d(all_movies, rated_movies)
    alpha = min(0.8, 0.3 + 0.1 * n_ratings)
    
    results = []
    for movie_id in movies_to_predict:
        cf_score = model.predict(user_id, movie_id).est
        genre_score = genre_similarity(movie_id, rated_movies)
        final_score = alpha * cf_score + (1 - alpha) * genre_score * 5
        
        row = df[df["movieId"] == movie_id].iloc[0]
        results.append({
            "movieId": int(movie_id),
            "title": row["title"],
            "genres": row["genres"],
            "score": round(final_score, 2)
        })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)[:k]

@app.route('/')
def index():
    return render_template('front.html')

@app.route('/api/popular', methods=['GET'])
def get_popular():
    popular = (
        df.groupby(["movieId", "title", "genres"])
        .agg(avgRating=("rating", "mean"), reviews=("rating", "count"))
        .reset_index()
    )
    popular = popular[popular["reviews"] >= 50] \
        .sort_values("avgRating", ascending=False) \
        .head(10)
    
    return jsonify(popular.to_dict('records'))

@app.route('/api/genres', methods=['GET'])
def get_genres():
    all_genres = sorted(set(g for gs in df["genres_list"] for g in gs))
    return jsonify(all_genres)

@app.route('/api/movies', methods=['GET'])
def get_movies():
    # Get unique movies
    unique_movies = df.drop_duplicates("movieId")[["movieId", "title", "genres"]].to_dict('records')
    return jsonify(unique_movies)

@app.route('/api/movies/search', methods=['GET'])
def search_movies():
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    
    results = df[df["title"].str.contains(query, case=False, na=False)]
    results = results.drop_duplicates("movieId")[["movieId", "title", "genres"]].head(20)
    return jsonify(results.to_dict('records'))

@app.route('/api/movies/by-genres', methods=['POST'])
def get_movies_by_genres():
    data = request.json
    selected_genres = data.get('genres', [])
    
    if not selected_genres:
        return jsonify([])
    
    def has_genre(genres):
        return any(g in genres for g in selected_genres)
    
    genre_movies = df[df["genres_list"].apply(has_genre)] \
        .groupby(["movieId", "title", "genres"]) \
        .agg(avgRating=("rating", "mean"), reviews=("rating", "count")) \
        .reset_index() \
        .sort_values("avgRating", ascending=False) \
        .head(30)
    
    return jsonify(genre_movies.to_dict('records'))

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_ratings = {int(k): v for k, v in data.get('ratings', {}).items()}
    
    if not user_ratings:
        return jsonify({"error": "No ratings provided"}), 400
    
    recommendations = hybrid_recommend_adaptive(
        model=model,
        df=df,
        user_id=9999,
        user_ratings=user_ratings,
        k=10
    )
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True, port=5000)