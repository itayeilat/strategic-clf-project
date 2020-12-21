from abc import ABC, abstractmethod
from cost_functions import *
from tqdm import tqdm
import pandas as pd



class TrainModel(ABC):
    @abstractmethod
    def __call__(self, X):
        """

        :param X: features to predict
        :return: the prediction result
        """

    # @abstractmethod
    # def train(self, X, y, *args, **kwargs):
    #     """
    #
    #     :param X: data to learn from
    #     :param y: True values
    #     """
    # @abstractmethod
    # def get_stratigic_new_feature(self, x: dict):
    #     """
    #
    #     :param x: original feature of member. Must be dictionary!
    #     :return: the new features.
    #     """



class HardtAlgo(TrainModel):
    def __init__(self, separable_cost: SeparableCost):
        self.min_si = None
        self.separable_cost = separable_cost

    def __call__(self, X):
        def apply_single_prdictive(x):
            return 1 if self.separable_cost.apply_cost2(x) >= self.min_si else -1

        if self.min_si is None:
            print("model hasn't trained yet. please train first")
            return

        return X.apply(apply_single_prdictive, axis=1)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        def apply_cost_with_thresh(x):
            return 1 if self.separable_cost.apply_cost1(x) >= thresh else -1

        min_err_si = np.inf
        S = X.apply(self.separable_cost.apply_cost2, axis=1) + 2
        with tqdm(total=len(S)) as t:
            for i, s_i in enumerate(S):
                thresh = s_i - 2
                err_si = np.sum(y != X.apply(apply_cost_with_thresh, axis=1)) / len(X)
                if min_err_si > err_si:
                    min_err_si = err_si
                    self.min_si = s_i
                t.update(1)






