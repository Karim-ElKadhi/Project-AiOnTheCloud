import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD
from streamlit_star_rating import st_star_rating

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("movies_merged.csv")
    return df

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

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
            "title": row["title"],
            "genres": row["genres"],
            "score": round(final_score, 2)
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:k]

# -----------------------------
# UI
# -----------------------------
st.title("🎬 Interactive Movie Recommendation System")

# Session state
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}


st.header("🔥 Top 10 Most Popular Movies")

popular = (
    df.groupby(["movieId", "title", "genres"])
    .agg(avg_rating=("rating", "mean"), count=("rating", "count"))
    .reset_index()
)

popular = popular[popular["count"] >= 50] \
    .sort_values("avg_rating", ascending=False) \
    .head(10)

st.dataframe(
    popular[["title", "genres", "avg_rating", "count"]],
    use_container_width=True
)


st.header("🎭 Choose Your Favorite Genres")

all_genres = sorted(set(g for gs in df["genres_list"] for g in gs))
selected_genres = st.multiselect("Select genres:", all_genres)


if selected_genres:
    st.header("🎬 Movies Based on Your Genres")

    def has_genre(genres):
        return any(g in genres for g in selected_genres)

    genre_movies = df[df["genres_list"].apply(has_genre)] \
        .groupby(["movieId", "title", "genres"]) \
        .agg(avg_rating=("rating", "mean")) \
        .reset_index() \
        .sort_values("avg_rating", ascending=False) \
        .head(20)

    for row in genre_movies.itertuples():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{row.title}**")
            st.caption(row.genres)
        with col2:
            rating = st_star_rating(
                "Rate",
                maxValue=5,
                defaultValue=0,
                key=f"rate_{row.movieId}"
            )
            if rating > 0:
                st.session_state.user_ratings[row.movieId] = rating


if len(st.session_state.user_ratings) > 0:
    st.header("✨ Personalized Recommendations")

    recs = hybrid_recommend_adaptive(
        model=model,
        df=df,
        user_id=9999,
        user_ratings=st.session_state.user_ratings,
        k=10
    )

    for r in recs:
        st.markdown(f"**{r['title']}**")
        st.caption(f"{r['genres']} — ⭐ {r['score']}")

st.sidebar.success(f"Movies rated: {len(st.session_state.user_ratings)}")
