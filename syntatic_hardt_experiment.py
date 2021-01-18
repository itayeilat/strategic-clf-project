import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from model import HardtAlgo
from cost_functions import WeightedLinearCostFunction, MixWeightedLinearSumSquareCostFunction
from strategic_players import strategic_modify_using_known_clf, strategic_modify_learn_from_friends
from create_synthetic_data import create_member_friends_dict
from utills_and_consts import evaluate_model_on_test_set, result_folder_path, plot_graph


def from_numpy_to_panda_df(data):
    data = pd.DataFrame(data=data[0:, 0:], index=[i for i in range(data.shape[0])],
                 columns=['f' + str(i) for i in range(data.shape[1])])
    return data


def create_dataset(data_size, pos_data_ratio, means_neg: np.array, means_pos: np.array, covariances_neg: np.array,
                   covariances_pos: np.array):
    assert len(means_neg) == len(means_pos)
    assert covariances_neg.shape == covariances_pos.shape
    assert covariances_pos.shape[0] == len(means_pos)
    pos_size, neg_size = int(data_size * pos_data_ratio), int(data_size * (1-pos_data_ratio))
    data_pos = np.random.multivariate_normal(mean=means_pos, cov=covariances_pos, size=pos_size)
    data_pos = from_numpy_to_panda_df(data_pos)
    data_neg = np.random.multivariate_normal(mean=means_neg, cov=covariances_neg, size=neg_size)
    data_neg = from_numpy_to_panda_df(data_neg)
    data = pd.concat([data_pos, data_neg], ignore_index=True)
    data.insert(len(data.columns), 'label', np.concatenate((np.ones(pos_size), -np.ones(neg_size))))
    data.insert(len(data.columns), 'MemberKey',
                ['Mem' + str(i) for i in range(len(data))], True)

    return data


def plot_hist(data, title: str = '', save_path=None):
    data_pos = data[data['label'] == 1]
    data_neg = data[data['label'] == -1]
    plt.hist(data_pos['f0'], 20, label='positive data', alpha=0.7)
    plt.hist(data_neg['f0'], 20, label='negetive data', alpha=0.7)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.savefig(save_path)
    plt.show()



base_folder = os.path.join(result_folder_path, 'oneD_synthetic_hardt_exp')
os.makedirs(base_folder, exist_ok=True)
histograms_folder = os.path.join(base_folder, 'histograms')
os.makedirs(histograms_folder, exist_ok=True)
mean_pos, mean_neg = 0, 4
variances_list = [2, 1, 0.5, 0.2, 0.1]


friends_list = [4, 6, 8, 10, 20]
result_friends_changes = list()
full_info_acc = list()
si_list = list()

for var in variances_list:
    result_friends_changes.append(list())
    np.random.seed(4)
    test_df = create_dataset(200, 0.5, np.array([mean_pos]), np.array([mean_neg]), np.array([[var]]), np.array([[var]]))
    plot_hist(test_df, title=f'original data no movements var: {var}', save_path=os.path.join(histograms_folder, f'original data no movements_{var}.png'))
    train_df = create_dataset(800, 0.5, np.array([mean_pos]), np.array([mean_neg]), np.array([[var]]), np.array([[var]]))
    cost_factor = 1
    cost = WeightedLinearCostFunction(np.array([1]), cost_factor)
    hardt_algo = HardtAlgo(cost)
    hardt_algo.fit(pd.DataFrame(pd.DataFrame(train_df['f0'])), train_df['label'])
    test_label = hardt_algo(pd.DataFrame(test_df['f0']))
    acc = np.sum(test_label == test_df['label']) / len(test_df['label'])
    print(f'acc on hardt {acc}')
    full_info_acc.append(acc)
    print(f'the si* we get: {hardt_algo.min_si}')
    si_list.append(hardt_algo.min_si)
    real_cost_func = MixWeightedLinearSumSquareCostFunction(np.array([1]), epsilon=0.2, cost_factor=cost_factor)

    modify_data = strategic_modify_using_known_clf(test_df, hardt_algo, ['f0'], real_cost_func)
    plot_hist(modify_data, title=f'known hardt model cov:{var}', save_path=os.path.join(histograms_folder, f'full_information_variance{var}.png'))
    acc = np.sum(hardt_algo(pd.DataFrame(modify_data['f0'])) == test_df['label']) / len(test_df['label'])
    print(f'hardt acc on modified: {acc}')
    hardet_train_label = hardt_algo(pd.DataFrame(train_df['f0']))

    for num_friend in friends_list:
        member_dict = create_member_friends_dict(num_friend, hardet_train_label, test_df, force_to_crate=True)
        modify_learn_from_friends, f_hat_acc, avg_l2_f_dist, avg_angle, num_example_moved = strategic_modify_learn_from_friends(None, test_df,

                                                                                                                                                 train_df,
                                                                                                                                                 hardet_train_label,
                                                                                                                                                 ['f0'], real_cost_func,

                                                                                                                                                 member_dict=member_dict,
                                                                                                                                                 f_vec=np.append(hardt_algo.coef_[0], hardt_algo.intercept_),
                                                                                                                                                 dir_name_for_result='',
                                                                                                                                                 title_for_visualization=f'',
                                                                                                                                                 visualization=False
                                                                                                                                                 )
        plot_hist(modify_learn_from_friends, title=f'learn model from {num_friend} friends var: {var}',
                  save_path=os.path.join(histograms_folder, f'f^_{num_friend}_friends_var_{var}.png'))
        acc = evaluate_model_on_test_set(modify_learn_from_friends, hardt_algo, ['f0'], 'label')
        print(f'acc on {num_friend} and var:{var} friends is: {acc}')
        result_friends_changes[-1].append(acc)

friends_list_duplicated = [friends_list for _ in range(len(result_friends_changes))]
graphs_labels_list = [f'var = {var}' for var in variances_list]
hardt_hat_acc_path = os.path.join(base_folder, 'acc_hardt_hat.png')
plot_graph('accuracy vs number of friend to learn', x_label='number of friends', y_label='accuracy',
           x_data_list=friends_list_duplicated, y_data_list=result_friends_changes, saving_path=hardt_hat_acc_path,
           graph_label_list=graphs_labels_list, symlog_scale=False)

plot_graph('si* vs variance', x_label='variance', y_label='si*', x_data_list=[variances_list], y_data_list=[si_list],
           saving_path=os.path.join(base_folder, 'si_vs_variance.png'),symlog_scale=False)

plot_graph('full info acc vs variance', 'variance', 'accuracy', x_data_list=[variances_list], y_data_list=[full_info_acc], saving_path=os.path.join(base_folder, 'acc_vs_variance.png'),symlog_scale=False)






