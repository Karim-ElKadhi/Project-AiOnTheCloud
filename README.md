# 🎬 Interactive Movie Recommendation System  

## 📌 Project Overview

This project implements an **interactive movie recommendation system** that adapts dynamically to user preferences.  
It combines **Collaborative Filtering** and **Content-Based Filtering** into a **hybrid recommendation approach**, deployed as a **Streamlit web application** on **Google Cloud (Vertex AI / Cloud Run)**.

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

---

## 🧭 User Interaction Flow

1. User opens the web application
2. Top 10 most popular movies are displayed
3. User selects preferred genres
4. System recommends 20 movies based on selected genres
5. User rates movies (1–5 stars)
6. Recommendations update dynamically
7. Process continues as the user rates more movies

---

## ▶️ Run the Application Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
