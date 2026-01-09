# 🎬 Interactive Movie Recommendation System  

## 📌 Project Overview

This project implements an **interactive movie recommendation system** that adapts dynamically to user preferences.  
It combines **Collaborative Filtering** and **Content-Based Filtering** into a **hybrid recommendation approach**, deployed as a **Flask web application with a modern HTML/CSS/JavaScript frontend** on **Google Cloud (Vertex AI / Cloud Run)**.
The system supports:
- Cold-start users
- Real-time interaction
- Adaptive recommendations as more ratings are provided
- Genre-aware personalization

---

## 🧠 Architecture Overview


User (Web Browser)

│

▼

Web Application 

│

├── Hybrid Recommendation Engine

│ ├── Collaborative Filtering (SVD)

│ └── Content-Based Filtering (Genres)

│

├── Pre-trained Model (model.pkl)

└── Movie Dataset (CSV or BigQuery)



---

## 🧩 Technologies Used

- Python
- Streamlit (Web Interface)
- Scikit-surprise (SVD)
- Scikit-learn
- Pandas & NumPy
- Google Cloud Platform (Vertex AI, Cloud Run)
- Docker

### Frontend
- **HTML5/CSS3** (Modern UI Design)
- **Vanilla JavaScript** (Dynamic Interactions)
- **Fetch API** (Backend Communication)
- **Responsive Grid Layout**

---
## 📂 Project Structure

project/
│

├── data

│ └──  movies_merged.csv # Merged dataset (ratings + movies)

├── utils

│ └──  model.pkl # Trained SVD model

│ └──  main.ipynb # script for EDA + model training

├── templates

│ └──  front.html # web interface

├── main.py # Streamlit application

├── requirements.txt # Python dependencies

├── Dockerfile # Container configuration

└── README.md # Documentation


---

## 📊 Dataset Description

The dataset is a merged version of movies and ratings data.

| Column     | Description |
|-----------|-------------|
| userId    | User identifier |
| movieId   | Movie identifier |
| rating    | Rating (1–5) |
| timestamp | Rating timestamp |
| title     | Movie title |
| genres    | Pipe-separated genres |

**Statistics:**
- 105,338 ratings
- 10,323 unique movies
- 938 unique genres


**Evaluation Metrics:** 
•	RMSE ≈ 0.73
•	MAE ≈ 0.54
Interpretation:
•	Predictions are on average within ±0.5 rating points


---

## 🤖 Recommendation Models

### 1️⃣ Collaborative Filtering (Model-Based)

- Algorithm: **SVD (Singular Value Decomposition)**
- Library: `scikit-surprise`
- Learns latent user–item factors
- Provides personalized predictions once enough ratings exist

---

### 2️⃣ Content-Based Filtering

- Uses movie **genres**
- Genres encoded using **MultiLabelBinarizer**
- Similarity computed using **cosine similarity**
- Effective for cold-start users

---

### 3️⃣ Hybrid Recommendation Strategy

The final recommendation score is computed as:
Final Score = α × Collaborative Score + (1 − α) × Genre Similarity


- `α` increases as the user provides more ratings
- Ensures smooth transition from content-based to collaborative filtering
  
•	α increases as the user provides more ratings
•	Early stage → content-based dominant
•	Later stage → collaborative dominant

---

## 🧭 User Interaction Flow

1. **Home Page** → User opens the web application
2. **Popular Movies** → Top 10 most popular movies are displayed
3. **Genre Selection** → User selects preferred genres (Action, Drama, Sci-Fi, etc.)
4. **Rating Interface** → 
   - System displays movies from selected genres
   - Search functionality to find specific movies
   - User rates movies (1–5 stars using interactive star rating)
5. **Dynamic Recommendations** → Hybrid algorithm generates personalized recommendations
6. **Results Page** → 
   - Top 10 recommended movies displayed with scores
   - Clickable titles linking to JustWatch for streaming availability
7. **Iteration** → User can return to rate more movies or start over with new genre preferences

---

## ▶️ Run the Application Locally

### 1️⃣ Clone the repository
```bash
git clone 
cd movie-recommender
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare the data
Ensure you have:
- `movies_merged.csv` in the project root
- `model.pkl` (trained SVD model) in the project root

### 4️⃣ Run the Flask server
```bash
python app.py
```
The application will start on `http://localhost:5000`

### 5️⃣ Open in browser
Navigate to `http://localhost:5000` in your web browser

