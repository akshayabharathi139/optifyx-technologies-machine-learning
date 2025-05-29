# 🎬 Movie Recommendation System

A **hybrid movie recommendation system** built using **Machine Learning** (collaborative & content-based filtering) and deployed as a **Flask web application** with a modern, responsive UI.

---

## 🚀 Features

- 🔁 **Hybrid Recommendation**: Collaborative Filtering (SVD) + Content-Based Filtering
- 🎯 **Personalized Suggestions**: Based on user input (movie name or user ID)
- 🌐 **Web App Interface**: Built using Flask and Bootstrap 5
- ✨ **Attractive UI**: Dark theme with glassmorphism styling
- 📦 **Reusable & Scalable**: Clean structure for future improvements

---

## 📁 Project Structure

movie-recommendation-system/
│
├── app.py # Flask app
├── model/ # Trained ML models
│ └── trained_model.pkl
├── dataset/ # Movie & ratings data (e.g., MovieLens)
│ └── movies.csv
│ └── ratings.csv
├── templates/ # HTML templates
│ ├── index.html
│ └── recommend.html
├── static/ # CSS styles
│ └── styles.css
└── requirements.txt # Python dependencies


---

## ⚙️ Installation & Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system

2. **Create a virtual environment & activate**

   python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # macOS/Linux


Demo Video

![movie recommendation system - Visual Studio Code 2025-05-26 19-52-27](https://github.com/user-attachments/assets/f7543231-babd-4d3a-88ce-ba7102ba2a2b)


🧠 Machine Learning Techniques
Collaborative Filtering: SVD model from scikit-surprise

Content-Based Filtering: Cosine similarity on movie features

Hybrid Approach: Combined results from both methods for enhanced accuracy


💻 Tech Stack
**Python
**Flask
**scikit-surprise
**Pandas, NumPy
**Bootstrap 5
**HTML, CSS (Glassmorphism UI)

📌 To Do / Future Enhancements
🎥 Integrate TMDB API for movie posters

💬 Add user reviews/ratings

📱 Deploy on Streamlit or Hugging Face Spaces






