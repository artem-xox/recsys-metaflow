import logging
import pandas as pd
from scipy import sparse
import warnings

from recommenders.datasets import movielens
from implicit.als import AlternatingLeastSquares
from pandas.api.types import CategoricalDtype
from metaflow import FlowSpec, step

warnings.filterwarnings("ignore")


class MovielensALSFlow(FlowSpec):
    """
    """
    ratings: pd.DataFrame
    train_ratings: pd.DataFrame
    test_ratings: pd.DataFrame
    user_items_train: sparse.csc

    model: AlternatingLeastSquares

    @step
    def start(self):
        """
        """
        self.next(self.data)

    @step
    def data(self):
        """
        """
        self.ratings = movielens.load_pandas_df(
            size='1m',
            title_col='title',
            genres_col='genre',
            year_col='year',
        )

        train_ratings, test_ratings = [], []
        num_test_samples = 10

        for userId, user_data in self.ratings.groupby('userID'):
            train_ratings += [user_data[:-num_test_samples]]
            test_ratings += [user_data[-num_test_samples:]]

        self.train_ratings = pd.concat(train_ratings)
        self.test_ratings = pd.concat(test_ratings)

        users = self.ratings["userID"].unique()
        movies = self.ratings["itemID"].unique()
        shape = (len(users), len(movies))

        user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
        movie_cat = CategoricalDtype(categories=sorted(movies), ordered=True)
        user_index = self.ratings["userID"].astype(user_cat).cat.codes
        movie_index = self.ratings["itemID"].astype(movie_cat).cat.codes

        self.user_items_train = sparse.coo_matrix(
            (self.ratings["rating"], (user_index, movie_index)), shape=shape
        ).tocsr()

        self.next(self.train)

    @step
    def train(self):
        """
        """
        self.model = AlternatingLeastSquares(
            iterations=20,
            factors=50
        )
        self.model.fit(
            self.user_items_train.T
        )

        self.next(self.end)

    @step
    def end(self):
        """
        """
        logging.info(self.model.similar_items(1))


if __name__ == "__main__":
    MovielensALSFlow()
