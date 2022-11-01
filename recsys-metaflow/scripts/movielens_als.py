import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import sparse
import warnings

from recommenders.datasets import movielens
from implicit.als import AlternatingLeastSquares
from metaflow import FlowSpec, step, project, current

warnings.filterwarnings("ignore")


@project(name='recsys')
class MovielensALSFlow(FlowSpec):
    """
    """
    ratings: pd.DataFrame
    user_items_train: sparse.csc

    params: dict
    model: AlternatingLeastSquares

    @step
    def start(self):
        """
        """
        print('project name:', current.project_name)
        print('project branch:', current.branch_name)
        print('is this a production run?', current.is_production)
        self.next(self.data)

    @step
    def data(self):
        """
        """
        self.ratings = movielens.load_pandas_df(
            size='10m',
            title_col='title',
            genres_col='genre',
        )

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
        self.params = {
            "iterations": 50,
            "factors": 50,
            "regularization": 0.1,
        }
        self.model = AlternatingLeastSquares(
            **self.params
        )
        self.model.fit(
            self.user_items_train
        )

        self.next(self.end)

    @step
    def end(self):
        """
        """


if __name__ == "__main__":
    MovielensALSFlow()
