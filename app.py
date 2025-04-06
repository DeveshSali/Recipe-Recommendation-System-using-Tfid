from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("food_recipes.csv")
df = df.dropna(subset=['ingredients'])
df.reset_index(drop=True, inplace=True)

# TF-IDF model
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['ingredients'])

# Recommendation function
def recommend_recipes(user_ingredients, top_n=5):
    user_ingredients_tfidf = tfidf.transform([user_ingredients])
    sim_scores = cosine_similarity(user_ingredients_tfidf, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]

    recommendations = []
    for idx in top_indices:
        rec = df.iloc[idx]
        recommendations.append({
            "title": rec['recipe_title'],
            "url": rec['url'],
            "rating": rec['rating'],
            "ingredients": rec['ingredients'],
            "description": rec['description']
        })
    return recommendations

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        ingredients = request.form["ingredients"]
        recommendations = recommend_recipes(ingredients)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
