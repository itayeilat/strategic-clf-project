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

def get_hardt_model(cost_factor, train_path, train_hardt=False):
    hardt_model_path = models_folder_path + f'/Hardt_cost_factor={cost_factor}'
    feature_list_to_use = six_most_significant_features
    if train_hardt or os.path.exists(hardt_model_path) is False:
        print(f'training Hardt model')
        train_df = pd.read_csv(train_path)
        hardt_algo = HardtAlgo(WeightedLinearCostFunction(a[:len(feature_list_to_use)], cost_factor))
        hardt_algo.fit(train_df[feature_list_to_use], train_df['LoanStatus'])
        hardt_algo.dump(hardt_model_path)
    else:
        hardt_algo = HardtAlgo.load_model(hardt_model_path)
    return hardt_algo

def evaluate_on_modify(train_path, test_path, model_to_train, feature_list_to_predict, target_label='LoanStatus'):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    model_to_train.fit(train_df[feature_list_to_predict], train_df[target_label])
    will_loan_returned_pred = model_to_train.predict(test_df[feature_list_to_predict])
    return sum(will_loan_returned_pred == test_df[target_label]) / len(will_loan_returned_pred)


def run_strategic_full_info(train_hardt=False, cost_factor=7, epsilon=0.3):
    dir_name_for_saving_visualize = 'result/full_information_strategic'
    os.makedirs(dir_name_for_saving_visualize, exist_ok=True)
    modify_test_df = create_strategic_data_sets_using_known_clf(dir_name_for_saving_visualize, cost_factor=cost_factor, epsilon=epsilon)

    feature_list_to_use = six_most_significant_features
    hardt_algo = get_hardt_model(cost_factor, train_path=real_train_val_f_star_loan_status_path, train_hardt=train_hardt)
    # hardt_algo = HardtAlgo(WeightedLinearCostFunction(a[:len(feature_list_to_use)], cost_factor=cost_factor))

    test_df = pd.read_csv(real_test_f_star_loan_status_path)



    pred_loan_status = hardt_algo(modify_test_df[feature_list_to_use])

    acc = np.sum(pred_loan_status == modify_test_df['LoanStatus']) / len(test_df)
    print(f'acc on modify test: {acc}')

    pred_loan_status = hardt_algo(test_df[feature_list_to_use])

    acc = np.sum(pred_loan_status == test_df['LoanStatus']) / len(test_df)
    print(f'acc on not modify test: {acc}')


def strategic_random_friends_info(train_hadart=True, cost_factor=10, epsilon=0.3):
    # main_create_synhetic_data(create_new_sample_set=True, train_gmm=False, fake_data_set_size=40000)
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



    f = load_sklearn_model(model_loan_returned_path)
    f_weights, f_inter = f.coef_[0], f.intercept_
    test_df = pd.read_csv(real_test_f_star_loan_status_path)
    test_pred_loans_status_f = f.predict(test_df[feature_list_to_use])
    # test_pred_loans_status_hardt = hardt_algo(fake_test_df)
    dict_result = dict()
    dict_result['number_of_friends_to_learn_list'] = [4, 6, 10, 50, 100, 200, 500, 1000, 2000, 4000, 7000, 10000, 15000]
    # dict_result['number_of_friends_to_learn_list'] = [4, 6, 10, 50, 100, 200, 500, 1000, 2000, 4000, 7000, 8500]
    dict_result['hardt_friends_acc_list'] = []
    dict_result['linear_model_friends_acc_list'] = []
    dict_result['num_improved_list_f'] = []
    dict_result['num_degrade_list_f'] = []
    dict_result['num_improved_hardt_list'] = []
    dict_result['num_degrade_hardt_list'] = []
    dict_result['avg_acc_f_hat'] = []
    dict_result['l2 dist'] = []
    dict_result['angel_f_f_hat'] = []

    parent_folder_path = os.path.join(result_folder_path, 'changed_samples_by_gaming_random_friends_losns_status_new_sy')
    os.makedirs(parent_folder_path, exist_ok=True)
    base_output_path = os.path.join(parent_folder_path, f'cost_factor={cost_factor}_epsilon={epsilon}f_hat_more_smaples')
    friends_dict_dir_path = os.path.join(parent_folder_path, 'friends_dict')
    os.makedirs(friends_dict_dir_path, exist_ok=True)

    for num_friend in dict_result['number_of_friends_to_learn_list']:
        print(num_friend)
        member_friend_dict_path = os.path.join(friends_dict_dir_path, f'member_friends_{num_friend}friends.json')
        member_dict = create_member_friends_dict(num_friend, real_train_val_f_star_loan_status_path,
                                                  real_test_f_star_loan_status_path, member_friend_dict_path, force_to_crate=False)

        # member_dict = create_member_friends_dict(num_friend, synthetic_train_val_path,
        #                                          synthetic_test_path, member_friend_dict_path, force_to_crate=True)

        cost_func_for_gaming = MixWeightedLinearSumSquareCostFunction(a[:len(feature_list_to_use)], epsilon=epsilon,
                                                                      cost_factor=cost_factor)
        friends_modify_strategic_data, f_hat_acc, avg_l2_f_dist, avg_angle = strategic_modify_learn_from_friends(real_test_f_star_loan_status_path,
                                                                            real_train_val_f_star_loan_status_path,
                                                                            # synthetic_train_val_path,
                                                                            feature_list_to_use, cost_func_for_gaming,
                                                                            target_label='LoanStatus',
                                                                            # target_label='LoanStatusByModelF',
                                                                            member_dict=member_dict,
                                                                            f_weights=f_weights, f_inter=f_inter,
                                                                            title_for_visualization=f'real test learned {num_friend}',
                                                                            dir_name_for_saving_visualize=base_output_path)
        dict_result['angel_f_f_hat'].append(avg_angle)
        dict_result['l2 dist'].append(avg_l2_f_dist)
        dict_result['avg_acc_f_hat'].append(f_hat_acc)
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

    # plt.plot(dict_result['number_of_friends_to_learn_list'], dict_result['num_improved_hardt_list'], '-r',
    #          label='Hardt model')

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
    # plt.plot(dict_result['number_of_friends_to_learn_list'], dict_result['num_degrade_hardt_list'], '-r',
    #          label='Hardt model')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(base_output_path, 'num_degrade_vs_num_friends.png'))
    plt.show()
    plt.close()

    plt.title('f_hat_avg_acc vs number of random friend to learn')
    plt.xlabel('number of friends')
    plt.xscale('symlog')
    plt.ylabel('avg acc')
    plt.plot(dict_result['number_of_friends_to_learn_list'], dict_result['avg_acc_f_hat'], '-b')
    plt.savefig(os.path.join(base_output_path, 'f_hat_avg_acc_vs_num_friends.png'))
    plt.show()
    plt.close()

    plt.title('f_hat dist from f vs number of random friend to learn')
    plt.xlabel('number of friends')
    plt.xscale('symlog')
    plt.ylabel('dist l2')
    plt.plot(dict_result['number_of_friends_to_learn_list'], dict_result['l2 dist'], '-b')
    plt.savefig(os.path.join(base_output_path, 'f_hat_dist_f_vs_num_friends.png'))
    plt.show()
    plt.close()

    plt.title('angle between f_hat and f vs number of random friend to learn')
    plt.xlabel('number of friends')
    plt.xscale('symlog')
    plt.ylabel('angle')
    plt.plot(dict_result['number_of_friends_to_learn_list'], dict_result['angel_f_f_hat'], '-b')
    plt.savefig(os.path.join(base_output_path, 'angle_between_f_hat_and_f_vs_num_friends.png'))
    plt.show()
    plt.close()


if __name__ == '__main__':
    # main_create_synhetic_data()
    # run_strategic_full_info(train_hardt=False, cost_factor=6, epsilon=0.3)
    # strategic_random_friends_info()
    print(10)
    #
    # from sklearn.linear_model import LogisticRegression
    # train_val_df = pd.read_csv(real_train_val_f_loan_status_path, nrows=100)
    # test_df = pd.read_csv(real_test_f_loan_status_path)
    # f = load_sklearn_model(model_loan_returned_path)
    # # f_hat = LinearSVC(penalty='l2', random_state=42, max_iter=1000).fit(train_val_df[six_most_significant_features], f.predict(train_val_df[six_most_significant_features]))
    # f_hat = LogisticRegression(penalty='l2', random_state=42, max_iter=1000).fit(train_val_df[six_most_significant_features], f.predict(train_val_df[six_most_significant_features]))
    # acc = np.sum(f_hat.predict(test_df[six_most_significant_features]) == test_df['LoanStatus']) / len(test_df)
    #
    # print(acc)

    strategic_random_friends_info(train_hadart=False, cost_factor=6, epsilon=0.3)
    # print(5)
    # strategic_random_friends_info(train_hadart=False, cost_factor=5, epsilon=0.3)
    # print(3)
    # strategic_random_friends_info(train_hadart=False, cost_factor=3, epsilon=0.3)
    # print(0.5)
    # strategic_random_friends_info(train_hadart=False, cost_factor=0.5, epsilon=0.3)
    # strategic_random_friends_info(train_hadart=False, cost_factor=0.5, epsilon=0.1)
