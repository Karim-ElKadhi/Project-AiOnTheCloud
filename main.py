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

def reset_ratings():
    for k in list(st.session_state.keys()):
        if k.startswith("rate_"):
            del st.session_state[k]
    st.session_state.user_ratings = {}



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


if "page" not in st.session_state:
    st.session_state.page = "home"

if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

if "selected_genres" not in st.session_state:
    st.session_state.selected_genres = set()
# -----------------------------
# UI
# -----------------------------
st.title("🎬 Interactive Movie Recommendation System")

# Session state
if st.session_state.page == "home":
    st.header("🔥 Top 10 Most Popular Movies")

    popular = (
        df.groupby(["movieId", "title", "genres"])
        .agg(AverageRating=("rating", "mean"), ReviewsNumber=("rating", "count"))
        .reset_index()
    )

    popular = popular[popular["ReviewsNumber"] >= 50] \
        .sort_values("AverageRating", ascending=False) \
        .head(10)

    st.dataframe(popular[["title", "genres", "AverageRating", "ReviewsNumber"]], use_container_width=True)
    if st.button("🎯 Get Recommendations"):
        reset_ratings()
        st.session_state.page = "genres"
        st.rerun()

elif st.session_state.page == "genres":
    st.title("🎭 Choose Your Favorite Genres")
    all_genres = sorted(set(g for gs in df["genres_list"] for g in gs))

    # Colored toggle cards
    cols = st.columns(4)
    colors = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D"]

    for i, genre in enumerate(all_genres):
        col = cols[i % 4]
        color = colors[i % len(colors)]
        selected = genre in st.session_state.selected_genres
        button_label = f"✅ {genre}" if selected else genre
        if col.button(button_label, key=f"genre_{genre}", use_container_width=True):
            if selected:
                st.session_state.selected_genres.remove(genre)
            else:
                st.session_state.selected_genres.add(genre)

    st.markdown("---")
    st.write("### Selected genres:")
    st.write(", ".join(st.session_state.selected_genres) if st.session_state.selected_genres else "None")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back"):
            reset_ratings()
            st.session_state.selected_genres = set()
            st.session_state.page = "home"
            st.rerun()
    with col2:
        if st.button("🎬 Start Rating"):
            if len(st.session_state.selected_genres) == 0:
                st.warning("Please select at least one genre!")
            else:
                reset_ratings()
                st.session_state.page = "rating"
                st.rerun()

# ----- RATING PAGE -----
elif st.session_state.page == "rating":
    st.title("⭐ Rate Movies")

    if len(st.session_state.selected_genres) == 0:
        st.warning("Please select at least one genre first.")
        if st.button("⬅ Back to Genre Selection"):
            st.session_state.page = "genres"
            st.rerun()
    else:
        # ---------- SEARCH FIELD ----------
        st.subheader("🔍 Search for a movie")
        search_text = st.text_input("Type movie name:", "")

        if search_text:
            filtered_movies = df[df["title"].str.contains(search_text, case=False, na=False)]
            filtered_movies = filtered_movies.drop_duplicates(subset="movieId") # Limit to first 10 results
            if not filtered_movies.empty:
                st.write(f"### Movies matching '{search_text}':")
                cols = st.columns(4)
                for idx, row in enumerate(filtered_movies.itertuples()):
                    col = cols[idx % 4]
                    with col:
                        st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
                        st.markdown(f'<div class="movie-title">{row.title}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="movie-genres">{row.genres}</div>', unsafe_allow_html=True)
                        key = f"rate_{row.movieId}"
                        rating = st_star_rating(
                            "Rate",
                            maxValue=5,
                            defaultValue=0,
                            key=key)
                        if rating > 0:
                            st.session_state.user_ratings[row.movieId] = rating
            else:
                st.info("No movies found matching your search.")

        st.markdown("---")
        st.markdown("Recommendations based on your selected genres:")

        def has_genre(genres):
            return any(g in genres for g in st.session_state.selected_genres)

        genre_movies = df[df["genres_list"].apply(has_genre)] \
            .groupby(["movieId", "title", "genres"]) \
            .agg(avg_rating=("rating", "mean")) \
            .reset_index() \
            .sort_values("avg_rating", ascending=False) \
            .head(20)

        # CSS for movie cards
        st.markdown("""
        <style>
        .movie-card, .rec-card {
            display: inline-block;
            margin: 10px;
            width: 200px;
            border-radius: 15px;
            overflow: hidden;
            background-color: #f5f5f5;  /* light gray background */
            color: black;
            border: 2px solid black;     /* <- visible border */
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            text-align: center;
            transition: transform 0.2s;
        }

        .movie-card:hover, .rec-card:hover {
            transform: scale(1.05);
        }

        .movie-title, .rec-title {
            font-weight: bold;
            margin: 5px 0;
        }

        .movie-genres, .rec-genres {
            font-size: 0.8rem;
            color: #555;
            margin-bottom: 5px;
        }

        .movie-rating, .rec-score {
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        cols = st.columns(4)
        for idx, row in enumerate(genre_movies.itertuples()):
            col = cols[idx % 4]
            with col:
                # Card container
                st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
                if getattr(row, "poster_url", None):
                    st.image(row.poster_url, use_column_width=True)
                st.markdown(f'<div class="movie-title">{row.title}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="movie-genres">{row.genres}</div>', unsafe_allow_html=True)
                key = f"rate_genre_{row.movieId}_{idx}"
                rating = st_star_rating(
                    "Rate",
                    maxValue=5,
                    defaultValue=0,
                    key=key
                )
                if rating > 0:
                    st.session_state.user_ratings[row.movieId] = rating

        if st.button("✨ Show Recommendations"):
            st.session_state.page = "results"
            st.rerun()

# ----- RESULTS PAGE -----
elif st.session_state.page == "results":
    st.title("✨ Personalized Recommendations")

    if len(st.session_state.user_ratings) == 0:
        st.warning("No ratings found. Please rate some movies first.")
        if st.button("⬅ Back to Rating"):
            st.session_state.page = "rating"
            st.rerun()
    else:
        recs = hybrid_recommend_adaptive(
            model=model,
            df=df,
            user_id=9999,
            user_ratings=st.session_state.user_ratings,
            k=10
        )

        # Display recommendation cards
        cols = st.columns(4)
        for idx, r in enumerate(recs):
            col = cols[idx % 4]
            with col:
                if r.get("poster"):
                    st.image(r["poster"], use_column_width=True)
                st.markdown(f"**{r['title']}**")
                st.caption(f"{r['genres']} — ⭐ {r['score']}")

        if st.button("🔁 Start Again"):
            reset_ratings()
            st.session_state.selected_genres = set()
            st.session_state.page = "home"
            st.rerun()
