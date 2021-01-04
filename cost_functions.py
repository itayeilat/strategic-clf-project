from abc import ABC, abstractmethod
import numpy as np
import cvxpy as cp

class CostFunction(ABC):
    @abstractmethod
    def __call__(self, z: np.array, x: np.array):
        '''

        :param z: Feature vector that player might want to have
        :param x: Feature that player now have.
        :return: the cost that player pays to become z
        '''
        pass

    def maximize_features_against_binary_model(self, x: np.array, trained_model):
        '''

        :param x: current vector features.
        :param trained_model: binary model that is trained and player want to get positive score on it.
        :return: vector features  that has minimum cost and get positive score on trained_model.
        '''
        pass


class SeparableCost(CostFunction):
    @abstractmethod
    def apply_cost1(self, x: np.array):
        pass

    @abstractmethod
    def apply_cost2(self, x: np.array):
        pass


class WeightedLinearCostFunction(SeparableCost):
    def __init__(self, weighted_vector: np.array, cost_factor=6):
        self.a = weighted_vector
        self.cost_factor = cost_factor

    def __call__(self, z: np.array, x: np.array):
        return max(self.a.T @ (z - x), 0)

    def maximize_features_against_binary_model(self, x: np.array, trained_model, tolerance=1e-9):
        x_tag = cp.Variable(len(x))

        func_to_solve = cp.Minimize(cp.maximum(self.a.T @ (x_tag - x), 0))
        constrains = [x_tag @ trained_model.coef_[0] >= -trained_model.intercept_ + tolerance]
        prob = cp.Problem(func_to_solve, constrains)
        prob.solve()
        cost_result = cp.maximum(self.a.T @(x_tag.value - x), 0)
        if x_tag is None:
            print("couldn't solve this problem")
            return
        if trained_model.predict(x_tag.value.reshape(1, -1))[0] == 1 and cost_result.value < 2:
            return x_tag.value
        else:
            return x

    def apply_cost1(self, x: np.array):
        return self.cost_factor * self.a.T @ x

    def apply_cost2(self, x: np.array):
        return self.cost_factor * self.a.T @ x


def check_result(trained_model, new_x, cost):
    return trained_model.predict(new_x.value.reshape(1, -1))[0] == 1 and cost.value < 2


class MixWeightedLinearSumSquareCostFunction(CostFunction):
    def __init__(self, weighted_vector: np.array, epsilon=0.3, cost_factor=7):
        self.a = weighted_vector
        self.epsilon = epsilon
        # some values for statistic and debugging:
        self.num_changed = 0  # the number of example that his changed because of solving the minimization problem
        self.num_examples = 0
        self.num_above_trash = 0
        self.trash = 0.0005
        self.cost_factor = cost_factor
        self.max_cost, self.max_separable_cost = -np.inf, -np.inf
        import pickle
        model_loan_returned_path = 'models/loan_returned_model.sav'
        self.f = pickle.load(open(model_loan_returned_path, 'rb'))
        self.num_changed_on_f_hat_not_f = 0
        self.cost_left_avg = 0
        self.sub_f_res_f_hat_res = 0
        # self.num_could_improved_on_f_not_f_hat = 0

    def __call__(self, z: np.array, x: np.array):
        return max((1 - self.epsilon) * self.a.T @ (z - x) + self.epsilon * np.sum((z - x) ** 2), 0)

    def solve_problem(self, model, x, tol):
        x_t = cp.Variable(len(x))
        func_to_solve = cp.Minimize(
            self.cost_factor * (cp.maximum((1 - self.epsilon) * self.a.T @ (x_t - x), 0) + self.epsilon *
                                cp.sum((x_t - x) ** 2)))
        constrains = [x_t @ model.coef_[0] >= -model.intercept_ + tol]
        if len(x) > 5:
            constrains.append(x_t[5] >= x[5])  # credit history can't get lower.
        if len(x) > 4:
            constrains.append(x_t[4] >= x[4])  # total number of inquiries can't get lower.
        prob = cp.Problem(func_to_solve, constrains)
        prob.solve()
        if x_t is None:
            print("couldn't solve this problem")
            return
        cost = cp.maximum((1 - self.epsilon) * self.a.T @ (x_t - x), 0) + self.epsilon * cp.sum(
            (x_t - x) ** 2)  # it look like cvxpy has bug that why I calculate again..
        cost *= self.cost_factor
        return x_t, cost

    def update_statistic_and_return_correct_x(self, x: np.ndarray, x_tag: cp.Variable, cost_result, trained_model):
        self.max_cost = max(self.max_cost, cost_result.value)
        self.max_separable_cost = max(self.max_separable_cost, (1 - self.epsilon) * self.a.T @ (x_tag.value - x))
        self.num_examples += 1
        if trained_model.predict(x_tag.value.reshape(1, -1))[0] == 1 and cost_result.value < 2:
            self.num_changed += 1
            if self.f.predict(x_tag.value.reshape(1, -1))[0] == -1:
                self.num_changed_on_f_hat_not_f += 1
                self.cost_left_avg += 2 - cost_result.value
                self.sub_f_res_f_hat_res += (x_tag.value @ self.f.coef_[0] + self.f.intercept_ - (
                            x_tag.value @ trained_model.coef_[0] + trained_model.intercept_))
            if self.a.T @ (x_tag.value - x) > self.trash:
                self.num_above_trash += 1
            return x_tag.value
        else:

            return x

    def smart_maximize_features_against_binary_model(self, x: np.array, trained_model):

        x_tag, cost_result = self.solve_problem(trained_model, x, 0.1)
        if not check_result(trained_model, x_tag, cost_result):
            x_tag, cost_result = self.solve_problem(trained_model, x, 0.01)
            if not check_result(trained_model, x_tag, cost_result):
                x_tag, cost_result = self.solve_problem(trained_model, x, 0.001)
                if not check_result(trained_model, x_tag, cost_result):
                    x_tag, cost_result = self.solve_problem(trained_model, x, 0.00000001)

        return self.update_statistic_and_return_correct_x(x, x_tag, cost_result, trained_model)


    def maximize_features_against_binary_model(self, x: np.array, trained_model, tolerance=0.00001, smart=False):
        if smart:
            return self.smart_maximize_features_against_binary_model(x, trained_model)
        x_tag, cost_result = self.solve_problem(trained_model, x, tolerance)
        return self.update_statistic_and_return_correct_x(x, x_tag, cost_result, trained_model)

    def get_statistic_on_num_change(self):
        calc_percent = lambda x: 100 * x /self.num_examples
        print(
            f'number of examples that has changed: {self.num_changed} and the percent is {calc_percent(self.num_changed)}'
            f'the number of examples above {self.trash} is : {self.num_above_trash} which are {calc_percent(self.num_above_trash)}% '
            f'max cost func is:{self.max_cost} and the max separable cost is: {self.max_separable_cost} \n'
        )
        if self.num_changed_on_f_hat_not_f != 0:
            print(
                f'num_changed_on_f_hat_not_f: {self.num_changed_on_f_hat_not_f}\n'
                f'cost left avg according to num changed on f_hat but not f: {self.cost_left_avg / self.num_changed_on_f_hat_not_f}\n'
                f'avg sub f and f_hat: {self.sub_f_res_f_hat_res/ self.num_changed_on_f_hat_not_f}'
            )
            # print(f'the number that could improved on f but not f_hat {self.num_could_improved_on_f_not_f_hat}')


# class SumSquareCostFunction(CostFunction):
#     def __init__(self, cost_factor):
#         self.cost_factor = cost_factor
#
#     def __call__(self, z: np.array, x: np.array):
#         return np.sum((z - x) ** 2)
#
#     def maximize_features_against_binary_model(self, x: np.array, trained_model, tolerance=1e+9):
#         z = cp.Variable(len(x))
#         func_to_solve = cp.Minimize(self.cost_factor * cp.sum_squares(z - x))
#         constrains = [z @ trained_model.coef_[0] >= -trained_model.intercept_ + tolerance]
#         prob = cp.Problem(func_to_solve, constrains)
#         result = prob.solve()
#         if z is None:
#             print("couldn't solve this problem")
#             return
#         if trained_model.predict(z.value.reshape(1, -1))[0] == 1 and result < 2:
#             return z.value
#         else:
#             return x
