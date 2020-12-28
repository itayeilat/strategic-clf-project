import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import json
import matplotlib.pyplot as plt
from train_member_clf import *
from startegic_players import *
from cost_functions import *
from create_synthetic_data import main_create_synhetic_data, create_member_friends_dict
import random


def evaluate_on_modify(train_path, test_path, model_to_train, feature_list_to_predict, target_label='LoanStatus'):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    model_to_train.fit(train_df[feature_list_to_predict], train_df[target_label])
    will_loan_returned_pred = model_to_train.predict(test_df[feature_list_to_predict])
    return sum(will_loan_returned_pred == test_df[target_label]) / len(will_loan_returned_pred)


def run_strategic_full_info():
    dir_name_for_saving_visualize = 'result/full_information_strategic'
    create_strategic_data_sets_using_known_clf(dir_name_for_saving_visualize)

    feature_list_to_use = six_most_significant_features
    hardt_algo = HardtAlgo(WeightedLinearCostFunction(a[:len(feature_list_to_use)]))

    test_df = pd.read_csv(synthetic_test_path)
    train_df = pd.read_csv(synthetic_train_path)
    modify_test_df = pd.read_csv(modify_full_information_test_synthetic_path)
    hardt_algo.fit(train_df[feature_list_to_use], train_df['LoanStatus'])

    pred_loan_status = hardt_algo(modify_test_df[feature_list_to_use])

    acc = np.sum(pred_loan_status == modify_test_df['LoanStatus']) / len(test_df)
    print(f'acc on modify test: {acc}')

    pred_loan_status = hardt_algo(test_df[feature_list_to_use])

    acc = np.sum(pred_loan_status == test_df['LoanStatus']) / len(test_df)
    print(f'acc on not modify test: {acc}')


def strategic_random_friends_info(train_hadart=False, cost_factor=5, epsilon=0.3):
    hardt_model_path = models_folder_path + f'/Hardt_cost_factor={cost_factor}'
    feature_list_to_use = six_most_significant_features
    if train_hadart or os.path.exists(hardt_model_path) is False:
        print(f'training Hardt model')
        train_df = pd.read_csv(synthetic_train_path)
        hardt_algo = HardtAlgo(WeightedLinearCostFunction(a[:len(feature_list_to_use)], cost_factor))
        hardt_algo.fit(train_df[feature_list_to_use], train_df['LoanStatus'])
        hardt_algo.dump(hardt_model_path)
    else:
        hardt_algo = HardtAlgo.load_model(hardt_model_path)

    # main_create_synhetic_data(create_new_sample_set=True, train_gmm=False)

    f = load_sklearn_model(model_loan_returned_path)
    fake_test_df = pd.read_csv(synthetic_test_path)[feature_list_to_use]
    test_pred_loans_status_f = f.predict(fake_test_df)
    # test_pred_loans_status_hardt = hardt_algo(fake_test_df)
    dict_result = dict()
    dict_result['number_of_friends_to_learn_list'] = [4, 6, 10, 50, 100, 200, 500, 1000, 2000, 4000, 7000, 10000]
    # dict_result['number_of_friends_to_learn_list'] = [4, 10000]
    dict_result['hardt_friends_acc_list'] = []
    dict_result['linear_model_friends_acc_list'] = []
    dict_result['num_improved_list_f'] = []
    dict_result['num_degrade_list_f'] = []
    dict_result['num_improved_hardt_list'] = []
    dict_result['num_degrade_hardt_list'] = []

    parent_folder_path = os.path.join(result_folder_path, 'changed_samples_by_gaming_random_friends')
    os.makedirs(parent_folder_path, exist_ok=True)
    base_output_path = os.path.join(parent_folder_path, f'cost_factor={cost_factor}_epsilon={epsilon}')
    friends_dict_dir_path = os.path.join(parent_folder_path, 'friends_dict')
    os.makedirs(friends_dict_dir_path, exist_ok=True)

    for num_friend in dict_result['number_of_friends_to_learn_list']:
        print(num_friend)
        member_friend_dict_path = os.path.join(friends_dict_dir_path, f'member_friends_{num_friend}friends.json')
        member_dict = create_member_friends_dict(num_friend, synthetic_set_to_sample_from_path,
                                                 synthetic_test_path, member_friend_dict_path)  # todo: consider save dictionary and make sure we have seed inside function
        cost_func_for_gaming = MixWeightedLinearSumSquareCostFunction(a[:len(feature_list_to_use)], epsilon=epsilon,
                                                                      cost_factor=cost_factor)
        friends_modify_strategic_data = strategic_modify_learn_from_friends(synthetic_test_path,
                                                                            synthetic_set_to_sample_from_path,
                                                                            feature_list_to_use, cost_func_for_gaming,
                                                                            target_label='LoanStatus',
                                                                            member_dict=member_dict,
                                                                            title_for_visualization=f'synthetic test learned {num_friend}',
                                                                            dir_name_for_saving_visualize=base_output_path)

        modify_pred_loan_status = f.predict(friends_modify_strategic_data[feature_list_to_use])
        dict_result['num_improved_list_f'].append(np.sum(modify_pred_loan_status > test_pred_loans_status_f).item())
        dict_result['num_degrade_list_f'].append(np.sum(modify_pred_loan_status < test_pred_loans_status_f).item())

        f_acc = np.sum(modify_pred_loan_status == friends_modify_strategic_data['LoanStatus']).item() / len(
            friends_modify_strategic_data)
        dict_result['linear_model_friends_acc_list'].append(f_acc)
        print(f_acc)

        hardt_pred_loan_status = hardt_algo(friends_modify_strategic_data[feature_list_to_use])
        dict_result['num_improved_hardt_list'].append(
            np.sum(hardt_pred_loan_status > test_pred_loans_status_f).item())

        dict_result['num_degrade_hardt_list'].append(
            np.sum(hardt_pred_loan_status < test_pred_loans_status_f).item())

        hardt_acc = np.sum(hardt_pred_loan_status == friends_modify_strategic_data['LoanStatus']).item() / len(
            friends_modify_strategic_data)
        print(hardt_acc)
        dict_result['hardt_friends_acc_list'].append(hardt_acc)

    with open(os.path.join(base_output_path, 'dict_result.json'), 'w') as json_file:
        json.dump(dict_result, json_file, indent=4)

    plt.title('accuracy vs number of random friend to learn')
    plt.xlabel('number of friends')
    plt.xscale('symlog')
    plt.ylabel('accuracy')
    plt.plot(dict_result['number_of_friends_to_learn_list'], dict_result['linear_model_friends_acc_list'], '-b',
             label='f linear model')
    plt.plot(dict_result['number_of_friends_to_learn_list'], dict_result['hardt_friends_acc_list'], '-r',
             label='Hardt model')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(base_output_path, 'accuracy_vs_num_friends.png'))
    plt.show()
    plt.close()

    plt.title('number players that improved on model vs number of random friend to learn')
    plt.xlabel('number of friends')
    plt.xscale('symlog')
    plt.ylabel('num improved')
    plt.plot(dict_result['number_of_friends_to_learn_list'], dict_result['num_improved_list_f'], '-b',
             label='f linear model')
    plt.plot(dict_result['number_of_friends_to_learn_list'], dict_result['num_improved_hardt_list'], '-r',
             label='Hardt model')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(base_output_path, 'num_improved_vs_num_friends.png'))
    plt.show()
    plt.close()

    plt.title('number players that degrade on model vs number of random friend to learn')
    plt.xlabel('number of friends')
    plt.xscale('symlog')
    plt.ylabel('num degrade')
    plt.plot(dict_result['number_of_friends_to_learn_list'], dict_result['num_degrade_list_f'], '-b',
             label='f linear model')
    plt.plot(dict_result['number_of_friends_to_learn_list'], dict_result['num_degrade_hardt_list'], '-r',
             label='Hardt model')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(base_output_path, 'num_degrade_vs_num_friends.png'))
    plt.show()
    plt.close()


if __name__ == '__main__':
    # main_create_synhetic_data()
    # strategic_random_friends_info()
    strategic_random_friends_info(train_hadart=False, cost_factor=10, epsilon=0.3)
    strategic_random_friends_info(train_hadart=False, cost_factor=5, epsilon=0.3)
    strategic_random_friends_info(train_hadart=False, cost_factor=3, epsilon=0.3)
    strategic_random_friends_info(train_hadart=False, cost_factor=0.5, epsilon=0.3)
    #run_strategic_full_info()
