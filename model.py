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

    @abstractmethod
    def fit(self, X, y):
        """

        :param X: data to learn from
        :param y: True values
        """
    @abstractmethod
    def predict(self, X):
        """

        :param X: data to predict
        :return: the new model prediction
        """

class RandomModel(TrainModel):
    def __init__(self):
        self.coef = np.random.normal(size=6)
        self.intercept = np.random.normal(size=1)

    def fit(self, X, y):
        return

    def predict(self, X):
        return self(X)

    def __call__(self, X):
        def return_single_prediction(single_float_score):
            return 1 if single_float_score >= 0 else -1

        return np.vectorize(return_single_prediction)(X @ self.coef + self.intercept)


class HardtAlgo(TrainModel):
    def __init__(self, separable_cost: SeparableCost):
        self.min_si = None
        self.separable_cost = separable_cost
        self.coef_ = (separable_cost.a, None)
        self.intercept_ = None

    def __call__(self, X):
        def apply_single_prdictive(x):
            return 1 if self.separable_cost.apply_cost2(x) >= self.min_si else -1

        if self.min_si is None:
            print("model hasn't trained yet. please train first")
            return

        if isinstance(X, np.ndarray):
            # vfunc = np.vectorize(apply_single_prdictive)
            # return vfunc(X)
            return np.apply_along_axis(apply_single_prdictive, 1, X)
        return X.apply(apply_single_prdictive, axis=1)

    def predict(self, X):
        return self(X)
    ''''
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):

         def apply_err_si_calc(s_i):
             def apply_cost_with_thresh(x):
                 return 1 if self.separable_cost.apply_cost1(x) >= thresh else -1

             thresh = s_i - 2
             err_si = np.sum(y != X.apply(apply_cost_with_thresh, axis=1)) / len(X)
             return err_si

         min_err_si = np.inf
         S = X.apply(self.separable_cost.apply_cost2, axis=1) + 2

         # with tqdm(total=len(S)) as t:
         #     for i, s_i in enumerate(S):
         #         thresh = s_i - 2
         #         err_si = np.sum(y != X.apply(apply_cost_with_thresh, axis=1)) / len(X)
         #         if min_err_si > err_si:
         #             min_err_si = err_si
         #             self.min_si = s_i
         #         t.update(1)
         tqdm.pandas()
         # err_si = S.apply(apply_err_si_calc)
         err_si = S.progress_apply(apply_err_si_calc)
         self.min_si = S[err_si.argmin()]
         # pd.DataFrame(S).apply(apply_err_si_calc)

    '''''
    # slow fit..
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
            print(f'min_err_si: {min_err_si}')
            self.intercept_ = -self.min_si / self.separable_cost.cost_factor

    # fast fit:
    # def fit(self, X: pd.DataFrame, y: pd.DataFrame):
    #     def apply_err_si_calc(s_i):
    #         def apply_cost_with_thresh(x):
    #             return 1 if x @ self.separable_cost.a * self.separable_cost.cost_factor + 2 >= thresh else -1
    #
    #         thresh = s_i - 2
    #         err_si = np.sum(y != np.apply_along_axis(apply_cost_with_thresh, 1, X)) / len(X)
    #         return err_si
    #
    #     X = X.to_numpy()
    #     S = X @ self.separable_cost.a * self.separable_cost.cost_factor + 2
    #     apply_err_si_calc_func = np.vectorize(apply_err_si_calc)
    #     err_si = apply_err_si_calc_func(S)
    #     self.min_si = S[err_si.argmin()]
    #     print(f'err min si: {err_si.min()}')
    #     self.intercept_ = -self.min_si / self.separable_cost.cost_factor


    # def dump(self, path_to_dump):
    #     with open(path_to_dump, 'wb') as output_file:
    #         pickle.dump(self, output_file, pickle.HIGHEST_PROTOCOL)

    # @classmethod
    # def load_model(cls, model_path):
    #     with open(model_path, 'rb') as model_file:
    #         return pickle.load(model_file)









