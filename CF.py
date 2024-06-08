def CF(title):


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

    # Pobierz fragment macierzy X
    data = X[:5, :5].toarray()

    # Utwórz DataFrame na podstawie danych 'data'
    df = pd.DataFrame(data)
    df.index.name = 'user'
    df.columns.name = 'item'

    # Wyświetl fragment macierzy z opisanymi osiami
    print(df)

    from sklearn.neighbors import NearestNeighbors

    def find_similar_movies(movie_id, movie_mapper, movie_inv_mapper, X, k, metric='cosine'):
        # Pobierz indeks wybranego filmu
        movie_index = movie_mapper[movie_id]

        # Pobierz wektor cech (ocen) dla wybranego filmu
        X = X.T
        movie_vector = X[movie_index]

        # Jeśli movie_vector jest tablicą numpy, spłasz ją do jednego wymiaru
        if isinstance(movie_vector, (np.ndarray)):
            movie_vector = movie_vector.reshape(1, -1)

        # Zainicjuj obiekt NearestNeighbors
        kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric)

        # Dopasuj model k-Nearest-Neighbours do danych
        kNN.fit(X)

        # Znajdź k najbliższych sąsiadów dla wybranego filmu
        neighbour = kNN.kneighbors(movie_vector, return_distance=False)

        # Zainicjuj listę do przechowywania indeksów najbliższych sąsiadów
        neighbour_indices = []

        # Pobierz movieId wybrane przez algorytm kNN
        for i in range(0, k):
            n = neighbour.item(i)
            neighbour_indices.append(movie_inv_mapper[n])

        # Usuń film, dla którego przeprowadzana jest rekomendacja
        neighbour_indices.pop(0)

        return neighbour_indices

    similar_movies = find_similar_movies(1, movie_mapper, movie_inv_mapper, X, k=11)
    similar_movies

    # Wyświetl filmy wraz z indeksami.
    print(f"Na podstawie filmu {title}:")
    print(movies['title'].iloc[similar_movies])

    return movies['title'].iloc[similar_movies], similar_movies
