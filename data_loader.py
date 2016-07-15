import os
import numpy as np
from scipy import sparse

class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def create_review_matrix(file_path):
        data = np.array([[int(tok) for tok in line.split('\t')[:3]]
                         for line in open(file_path)])

        ij = data[:, :2]
        ij -= 1
        values = data[:, 2]
        review_matrix = sparse.csc_matrix((values, ij.T)).astype(float)
        return review_matrix

movielens_file_path = '%s/ml-100k/u1.base' % os.environ['PWD']

my_reviews = DataLoader.create_review_matrix(movielens_file_path)

user_reviews = my_reviews[8]
user_reviews = user_reviews.toarray().ravel()
user_rated_movies,  = np.where(user_reviews > 0)
user_ratings = user_reviews[user_rated_movies]

movie_reviews = my_reviews[:, 201]
movie_reviews = movie_reviews.toarray().ravel()
movie_rated_users,  = np.where(movie_reviews > 0)
movie_ratings = movie_reviews[movie_rated_users]

user_pseudo_average_ratings = {}
user_pseudo_average_ratings[8] = np.mean(user_ratings)
user_pseudo_average_ratings[9] = np.mean(user_ratings)
user_pseudo_average_ratings[10] = np.mean(user_ratings)
users, movies = my_reviews.nonzero()

users_matrix = np.empty((3, 3))
users_matrix[:] = 0.1

movies_matrix = np.empty((3, 3))
movies_matrix[:] = 0.1

result = users_matrix[0] * movies_matrix[0]
otro = movies_matrix[:, 2]
otro[2] = 8