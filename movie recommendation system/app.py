from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

app = Flask(__name__)

# ===================== LOAD DATA =====================
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge ratings with movie titles
movie_ratings = pd.merge(ratings, movies, on='movieId')

# ===================== CONTENT-BASED MODEL =====================
# Combine title and genres into a single string
movies['content'] = movies['title'] + " " + movies['genres']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# ===================== COLLABORATIVE FILTERING (SVD) =====================
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
svd = SVD()
svd.fit(trainset)  # Train the model

# ===================== HYBRID RECOMMENDER =====================
def hybrid_recommendations(title, user_id=1):
    idx = indices.get(title)
    if idx is None:
        return ["Movie not found in database."]
    
    # --- Content-Based ---
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:21]
    movie_indices = [i[0] for i in sim_scores]
    similar_movies = movies.iloc[movie_indices][['movieId', 'title']]

    # --- Collaborative Filtering using SVD ---
    similar_movies['est_rating'] = similar_movies['movieId'].apply(lambda x: svd.predict(user_id, x).est)
    hybrid_top = similar_movies.sort_values('est_rating', ascending=False).head(10)

    return hybrid_top['title'].tolist()

# ===================== FLASK ROUTES =====================
@app.route('/')
def index():
    all_titles = movies['title'].sort_values().tolist()
    return render_template('index.html', movies=all_titles)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie = request.form['movie']
    recommendations = hybrid_recommendations(movie)
    return render_template('recommend.html', movie=movie, recommendations=recommendations)

# ===================== MAIN =====================
if __name__ == '__main__':
    app.run(debug=True)
