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
        :return: vector features  that has minimum cost and get positive score.
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
    def __init__(self, weighted_vector: np.array, cost_factor=10):
        self.a = weighted_vector
        self.cost_factor = cost_factor

    def __call__(self, z: np.array, x: np.array):
        return max(self.a.T @ (z - x), 0)

    def maximize_features_against_binary_model(self, x: np.array, trained_model, tolerance=1e-9):
        z = cp.Variable(len(x))

        func_to_solve = cp.Minimize(cp.maximum(self.a.T @ (z - x), 0))
        constrains = [z @ trained_model.coef_[0] >= -trained_model.intercept_ + tolerance]
        prob = cp.Problem(func_to_solve, constrains)
        result = prob.solve()

        cost_result = cp.maximum(self.a.T @(z.value - x), 0) # it look like cvxpy ha s bug that why I calculate again..
        if z is None:
            print("couldn't solve this problem")
            return
        if trained_model.predict(z.value.reshape(1, -1))[0] == 1 and cost_result.value < 2:
            return z.value
        else:
            return x

    def apply_cost1(self, x: np.array):
        return self.cost_factor * self.a.T @ x

    def apply_cost2(self, x: np.array):
        return self.cost_factor * self.a.T @ x


class MixWeightedLinearSumSquareCostFunction(CostFunction):
    def __init__(self, weighted_vector: np.array, epsilon=0.3, cost_factor=10):
        self.a = weighted_vector
        self.epsilon = epsilon
        self.num_changed = 0 # the number of example that his changed because of solving the minimization problem
        self.num_examples = 0
        self.num_above_trash = 0
        self.trash = 0.0005
        self.cost_factor = cost_factor
        self.max_cost, self.max_separable_cost = -np.inf, -np.inf

    def __call__(self, z: np.array, x: np.array):
        return max((1 - self.epsilon) * self.a.T @ (z - x) + self.epsilon * np.sum((z - x) ** 2), 0)

    def maximize_features_against_binary_model(self, x: np.array, trained_model, tolerance=0.01):
        x_tag = cp.Variable(len(x))
        func_to_solve = cp.Minimize(self.cost_factor * (cp.maximum((1 - self.epsilon) * self.a.T @ (x_tag - x), 0) + self.epsilon *
                                               cp.sum((x_tag - x) ** 2)))

        constrains = [x_tag @ trained_model.coef_[0] >= -trained_model.intercept_ + tolerance]
        if len(x) > 5:
            constrains.append(x_tag[5] >= x[5]) #credit history can't get lower.
        if len(x) > 4:
            constrains.append(x_tag[4] >= x[4])  # total number of inquiries can't get lower.
        prob = cp.Problem(func_to_solve, constrains)
        result = prob.solve()

        cost_result = cp.maximum((1 - self.epsilon) * self.a.T @ (x_tag - x), 0) + self.epsilon * cp.sum((x_tag - x) ** 2) # it look like cvxpy has bug that why I calculate again..
        cost_result *= self.cost_factor
        self.max_cost = max(self.max_cost, cost_result.value)
        self.max_separable_cost = max(self.max_separable_cost, (1 - self.epsilon) * self.a.T @ (x_tag.value - x))
        if x_tag is None:
            print("couldn't solve this problem")
            return x
        self.num_examples += 1
        if trained_model.predict(x_tag.value.reshape(1, -1))[0] == 1 and cost_result.value < 2:
            if self.a.T @ (x_tag.value - x) > self.trash:
                self.num_above_trash += 1
            self.num_changed += 1
            return x_tag.value

        else:
            return x



    def get_statistic_on_num_change(self):
        calc_percent = lambda x: 100 * x /self.num_examples
        print(
            f'number of examples that has changed: {self.num_changed} and the percent is {calc_percent(self.num_changed)}'
            f'the number of examples above {self.trash} is : {self.num_above_trash} which are {calc_percent(self.num_above_trash)}% '
            f'max cost func is:{self.max_cost} and the max separable cost is: {self.max_separable_cost}')


        #return self.num_changed, 100 * self.num_changed / self.num_examples


class SumSquareCostFunction(CostFunction):
    def __init__(self, cost_factor):
        self.cost_factor = cost_factor

    def __call__(self, z: np.array, x: np.array):
        return np.sum((z - x) ** 2)

    def maximize_features_against_binary_model(self, x: np.array, trained_model, tolerance=1e+9):
        z = cp.Variable(len(x))
        func_to_solve = cp.Minimize(self.cost_factor * cp.sum_squares(z - x))
        constrains = [z @ trained_model.coef_[0] >= -trained_model.intercept_ + tolerance]
        prob = cp.Problem(func_to_solve, constrains)
        result = prob.solve()
        if z is None:
            print("couldn't solve this problem")
            return
        if trained_model.predict(z.value.reshape(1, -1))[0] == 1 and result < 2:
            return z.value
        else:
            return x
