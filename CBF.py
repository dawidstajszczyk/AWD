def CBF(title):


    import numpy as np
    import pandas as pd


    # Wczytaj dostępne filmy
    movies = pd.read_csv('https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv')
    movies.head()

    # Wczytaj dostępne oceny
    ratings = pd.read_csv('https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv')
    ratings.head()

    from scipy.sparse import csr_matrix

    def user_item_matrix(df):

        # Pobierz wymiary macierzy
        rows_num = df['userId'].nunique()
        columns_num = df['movieId'].nunique()

        # Pobierz unikalne ID użytkowników i filmów
        unique_users = np.unique(df["userId"])
        unique_movies = np.unique(df["movieId"])

        # Utwórz mapper dla użytkowników (which userId correspond to which row 'utility' matrix)
        user_mapper = {user_id: index for index, user_id in enumerate(unique_users)}

        # Utwórz mapper dla filmów (which movieId correspond to which column 'utility' matrix)
        movie_mapper = {movie_id: index for index, movie_id in enumerate(unique_movies)}

        # Utwórz mapper odwrotny dla użytkowników
        user_inv_mapper = {index: user_id for index, user_id in enumerate(unique_users)}

        # Utwórz mapper odwrotny dla filmów
        movie_inv_mapper = {index: movie_id for index, movie_id in enumerate(unique_movies)}

        # Pobierz indeksy użytkowników i filmów
        user_indices = [user_mapper[i] for i in df['userId']]
        item_indices = [movie_mapper[i] for i in df['movieId']]

        # Utwórz user-item matrix
        X = csr_matrix((df["rating"], (user_indices, item_indices)), shape=(rows_num, columns_num))

        return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

    # Utwórz user-item matrix (X)
    X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = user_item_matrix(ratings)

    # Utwórz zestaw wszystkich gatunków filmowych
    genres = set()
    for genre_list in movies['genres']:
        for genre in genre_list.split('|'):
            genres.add(genre)

    # Dodaj nową kolumnę z gatunkiem 'genre' do ramki danych 'movies'.
    # Wartość w tej kolumnie to 1, jeśli film należy do danego gatunku, a 0 w przeciwnym przypadku.
    for genre in genres:
        movies[genre] = movies['genres'].transform(lambda x: int(genre in x))

    # Utwórz kopię DataFrame'u i usuń wymienione kolumny
    movie_genres = movies.drop(columns=['movieId', 'title', 'genres'])

    # Podejrzyj fragment DataFrame'u
    movie_genres.head()

    from sklearn.metrics.pairwise import cosine_similarity

    # Oblicz cosine_similatiry
    # cosine_sim[i, j] będzie reprezentować podobieństwo kosinusowe między i-tym a j-tym filmem.
    cosine_sim = cosine_similarity(movie_genres, movie_genres)

    print("Cosine similarity matrix")

    # Przekształć tablicę numpy na ramkę danych DataFrame i wyświelt
    df = pd.DataFrame(cosine_sim)
    df

    def get_content_based_recommendations(title, n_recommendations=10):

        # Utwórz słownik. Klucz - tytuł filmu, wartość - indeks filmu
        movie_idx = dict(zip(movies['title'], list(movies.index)))

        # Przypisz indeks wybranego filmu
        idx = movie_idx[title]

        # Wyodrębnij podobieństwo kosinusowe pomiędzy wybranym filmem, a pozostałymi
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sortuj malejąco listę 'sim_scores'
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Wybierz 'n_recommendations' podobnych filmów do wybranego
        sim_scores = sim_scores[1:(n_recommendations + 1)]

        # Utwórz listę z indeksami wybranych filmów
        similar_movies = [i[0] for i in sim_scores]

        return similar_movies

    similar_movies = get_content_based_recommendations(title, 10)

    # Wyświetl filmy wraz z indeksami.
    print(f"Na podstawie filmu {title}:")
    print(movies['title'].iloc[similar_movies])

    return movies['title'].iloc[similar_movies], similar_movies
