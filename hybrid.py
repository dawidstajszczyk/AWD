from CF import CF
from CBF import CBF
import pandas as pd

def hybrid_recommendation(title):
    # Wczytaj dostępne filmy
    movies = pd.read_csv('https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv')

    # Collaborative Filtering
    _, similar_movies_cf = CF(title)

    # Content-Based Filtering
    _, similar_movies_cb = CBF(title)

    # Połącz wyniki, unikając duplikatów
    combined_results = list(set(similar_movies_cf + similar_movies_cb))

    return movies['title'].iloc[combined_results]
