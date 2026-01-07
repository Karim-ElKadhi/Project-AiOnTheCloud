from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
CORS(app)
STREAMING_API_KEY = "c8442e2b69msh29ee07b58678614p1dc075jsnd301611895a0"  
STREAMING_API_HOST = "streaming-availability.p.rapidapi.com"
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
@app.route('/api/streaming/<int:movie_id>', methods=['GET'])
def get_streaming_info(movie_id):
    """Get streaming availability for a movie"""
    # Get movie info from dataframe
    movie = df[df["movieId"] == movie_id]
    if movie.empty:
        return jsonify({"error": "Movie not found"}), 404
    
    title = movie.iloc[0]["title"]
    
    # Extract title and year
    import re
    match = re.match(r"(.+?)\s*\((\d{4})\)", title)
    if match:
        movie_title = match.group(1).strip()
        year = match.group(2)
    else:
        movie_title = title
        year = None
    
    try:
        # Search for the movie using Streaming Availability API
        headers = {
            "X-RapidAPI-Key": STREAMING_API_KEY,
            "X-RapidAPI-Host": STREAMING_API_HOST
        }
        
        search_url = "https://streaming-availability.p.rapidapi.com/shows/search/title"
        params = {
            "title": movie_title,
            "country": "us",
            "output_language": "en"
        }
        
        if year:
            params["year"] = year
        
        response = requests.get(search_url, headers=headers, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            if data and len(data) > 0:
                # Get the first (best) match
                show = data[0]
                streaming_options = show.get("streamingOptions", {}).get("us", [])
                
                # Organize by service
                services = {}
                for option in streaming_options:
                    service_info = option.get("service", {})
                    service_id = service_info.get("id", "")
                    service_name = service_info.get("name", "")
                    link = option.get("link", "")
                    stream_type = option.get("type", "")  # subscription, free, rent, buy
                    
                    if service_id not in services:
                        services[service_id] = {
                            "name": service_name,
                            "link": link,
                            "type": stream_type,
                            "logo": f"https://www.justwatch.com/images/icon/{service_id}.png"
                        }
                
                return jsonify({
                    "movieId": movie_id,
                    "title": title,
                    "available": len(services) > 0,
                    "services": services,
                    "imdbId": show.get("imdbId", ""),
                    "tmdbId": show.get("tmdbId", "")
                })
        
        # Fallback if no streaming found
        return jsonify({
            "movieId": movie_id,
            "title": title,
            "available": False,
            "services": {},
            "fallbackLinks": {
                "justwatch": f"https://www.justwatch.com/us/search?q={movie_title.replace(' ', '-').lower()}",
                "google": f"https://www.google.com/search?q=watch+{movie_title}+{year if year else ''}+online+streaming"
            }
        })
        
    except Exception as e:
        print(f"Error fetching streaming info: {e}")
        # Return fallback links
        return jsonify({
            "movieId": movie_id,
            "title": title,
            "available": False,
            "services": {},
            "error": str(e),
            "fallbackLinks": {
                "justwatch": f"https://www.justwatch.com/us/search?q={movie_title.replace(' ', '-').lower()}",
                "google": f"https://www.google.com/search?q=watch+{movie_title}+{year if year else ''}+online+streaming"
            }
        })

@app.route('/api/recommendations/with-streaming', methods=['POST'])
def recommend_with_streaming():
    """Get recommendations and fetch streaming info for all of them"""
    data = request.json
    user_ratings = {int(k): v for k, v in data.get('ratings', {}).items()}
    
    if not user_ratings:
        return jsonify({"error": "No ratings provided"}), 400
    
    # Get recommendations
    recommendations = hybrid_recommend_adaptive(
        model=model,
        df=df,
        user_id=9999,
        user_ratings=user_ratings,
        k=10
    )
    
    # Fetch streaming info for each recommendation
    for rec in recommendations:
        movie_id = rec['movieId']
        
        # Get movie title and extract year
        title = rec['title']
        import re
        match = re.match(r"(.+?)\s*\((\d{4})\)", title)
        if match:
            movie_title = match.group(1).strip()
            year = match.group(2)
        else:
            movie_title = title
            year = None
        
        try:
            headers = {
                "X-RapidAPI-Key": STREAMING_API_KEY,
                "X-RapidAPI-Host": STREAMING_API_HOST
            }
            
            search_url = "https://streaming-availability.p.rapidapi.com/shows/search/title"
            params = {
                "title": movie_title,
                "country": "us",
                "output_language": "en"
            }
            
            if year:
                params["year"] = year
            
            response = requests.get(search_url, headers=headers, params=params, timeout=3)
            
            if response.status_code == 200:
                search_data = response.json()
                
                if search_data and len(search_data) > 0:
                    show = search_data[0]
                    streaming_options = show.get("streamingOptions", {}).get("us", [])
                    
                    services = {}
                    for option in streaming_options:
                        service_info = option.get("service", {})
                        service_id = service_info.get("id", "")
                        service_name = service_info.get("name", "")
                        link = option.get("link", "")
                        stream_type = option.get("type", "")
                        
                        if service_id not in services:
                            services[service_id] = {
                                "name": service_name,
                                "link": link,
                                "type": stream_type
                            }
                    
                    rec['streaming'] = {
                        "available": len(services) > 0,
                        "services": services
                    }
                else:
                    rec['streaming'] = {"available": False, "services": {}}
            else:
                rec['streaming'] = {"available": False, "services": {}}
                
        except Exception as e:
            print(f"Error fetching streaming for {movie_title}: {e}")
            rec['streaming'] = {"available": False, "services": {}}
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True, port=5000)