import json
import matplotlib.pyplot as plt
from startegic_players import *
from create_synthetic_data import main_create_synthetic_data, create_member_friends_dict
from model import *


def get_hardt_model(cost_factor, train_path, force_train_hardt=False):
    hardt_model_path = os.path.join(models_folder_path, f'Hardt_cost_factor={cost_factor}')
    feature_list_to_use = six_most_significant_features
    if force_train_hardt or os.path.exists(hardt_model_path) is False:
        print(f'training Hardt model')
        train_df = pd.read_csv(train_path)
        hardt_algo = HardtAlgo(WeightedLinearCostFunction(a[:len(feature_list_to_use)], cost_factor))

        hardt_algo.fit(train_df[feature_list_to_use], train_df['LoanStatus'])
        save_model(hardt_algo, hardt_model_path)
    else:
        hardt_algo = load_model(hardt_model_path)
    return hardt_algo


def run_strategic_full_info(train_hardt=False, cost_factor=7, epsilon=0.3):
    dir_name_for_saving_visualize = os.path.join(result_folder_path, 'full_information_strategic')
    os.makedirs(dir_name_for_saving_visualize, exist_ok=True)
    modify_test_df = create_strategic_data_sets_using_known_clf(dir_name_for_saving_visualize, cost_factor=cost_factor, epsilon=epsilon)

    feature_list_to_use = six_most_significant_features
    # hardt_algo = get_hardt_model(cost_factor, train_path=real_train_val_f_star_loan_status_path, force_train_hardt=train_hardt)
    hardt_algo = get_hardt_model(cost_factor, train_path=real_train_val_f_star_loan_status_path,
                                 force_train_hardt=train_hardt)
    test_df = pd.read_csv(real_test_f_star_loan_status_path)
    acc = evaluate_model_on_test_set(modify_test_df, hardt_algo, feature_list_to_use)
    print(f'acc on modify test: {acc}')
    acc = evaluate_model_on_test_set(test_df, hardt_algo, feature_list_to_use)
    print(f'acc on not modify test: {acc}')


def strategic_random_friends_info(train_hadart=True, cost_factor=10, epsilon=0.3):
    def init_dict_result():
        dict_result = dict()
        dict_result['number_of_friends_to_learn_list'] = [4, 6, 10, 50, 100, 200, 500, 1000, 2000, 4000, 7000, 10000,
                                                          15000]
        dict_result['hardt_friends_acc_list'] = []
        dict_result['linear_model_friends_acc_list'] = []
        dict_result['num_improved_list_f'] = []
        dict_result['num_degrade_list_f'] = []
        dict_result['num_improved_hardt_list'] = []
        dict_result['num_degrade_hardt_list'] = []
        dict_result['avg_acc_f_hat'] = []
        dict_result['l2 dist'] = []
        dict_result['angel_f_f_hat'] = []
        return dict_result

    def get_datasets_and_f_grade(f_model, train_path, test_path):
        #todo: this function might be outside strategic_random_friends_info
        test_f_star = pd.read_csv(test_path)
        train_f_star = pd.read_csv(train_path)
        train_f_loan_status = f_model.predict(train_f_star[feature_list_to_use])
        test_f_loan_status = f_model.predict(test_f_star[feature_list_to_use])
        return train_f_star, test_f_star, train_f_loan_status, test_f_loan_status

    def create_paths_and_dirs_for_random_friends_experiment():
        path_to_parent_folder = os.path.join(result_folder_path,
                                          'changed_samples_by_gaming_random_friends_losns_status')
        os.makedirs(path_to_parent_folder, exist_ok=True)
        path_to_base_output = os.path.join(path_to_parent_folder,
                                        f'cost_factor={cost_factor}_epsilon={epsilon}')
        path_to_friends_dict_dir = os.path.join(path_to_parent_folder, 'friends_dict')
        os.makedirs(path_to_friends_dict_dir, exist_ok=True)
        return path_to_parent_folder, path_to_base_output, path_to_friends_dict_dir

    def update_dict_result():
        dict_result['angel_f_f_hat'].append(avg_angle)
        dict_result['l2 dist'].append(avg_l2_f_dist)
        dict_result['avg_acc_f_hat'].append(f_hat_acc)
        f_pred_on_modify = f.predict(friends_modify_strategic_data[feature_list_to_use])
        dict_result['num_improved_list_f'].append(np.sum(f_pred_on_modify > test_pred_loans_status_f).item())
        dict_result['num_degrade_list_f'].append(np.sum(f_pred_on_modify < test_pred_loans_status_f).item())

        f_acc = np.sum(f_pred_on_modify == friends_modify_strategic_data['LoanStatus']).item() / len(
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

    def plot_graph(title: str, x_label: str, y_label: str, x_data_list: list, y_data_list: list, saving_path: str, graph_label_list=None):
        plt.title(title)
        plt.xlabel(x_label)
        plt.xscale('symlog')
        plt.ylabel(y_label)
        color_list = ['-b', '-r']
        if graph_label_list is not None:
            for i in range(len(x_data_list)):
                plt.plot(x_data_list[i], y_data_list[i], color_list[i], label=graph_label_list[i])
            plt.legend(loc="upper right")
        else:
            for i in range(len(x_data_list)):
                plt.plot(x_data_list[i], y_data_list[i], color_list[i])
        plt.savefig(saving_path)
        plt.show()

    def plot_dict_result_graph():
        x_data_list = [dict_result['number_of_friends_to_learn_list'], dict_result['number_of_friends_to_learn_list']]
        y_data_list = [dict_result['linear_model_friends_acc_list'], dict_result['hardt_friends_acc_list']]
        saving_path = os.path.join(base_output_path, 'accuracy_vs_num_friends.png')
        plot_graph(title='accuracy vs number of random friends to learn',
                   x_label='accuracy vs number of random friend to learn',
                   y_label='accuracy', x_data_list=x_data_list, y_data_list=y_data_list,
                   graph_label_list=['f linear model', 'Hardt model'], saving_path=saving_path)

        x_data_list = [dict_result['number_of_friends_to_learn_list']]
        y_data_list = [dict_result['num_improved_list_f']]
        saving_path = os.path.join(base_output_path, 'num_improved_vs_num_friends.png')
        plot_graph(title='number players that improved on model vs number of random friends to learn',
                   x_label='number of friends',
                   y_label='num improved', x_data_list=x_data_list, y_data_list=y_data_list, saving_path=saving_path)

        y_data_list = [dict_result['num_degrade_list_f']]
        saving_path = os.path.join(base_output_path, 'num_degrade_vs_num_friends.png')
        plot_graph(title='number players that degrade on model vs number of random friends to learn',
                   x_label='number of friends',
                   y_label='num degrade', x_data_list=x_data_list, y_data_list=y_data_list, saving_path=saving_path)

        y_data_list = [dict_result['avg_acc_f_hat']]
        saving_path = os.path.join(base_output_path, 'f_hat_avg_acc_vs_num_friends.png')
        plot_graph(title='f_hat_avg_acc vs number of random friends to learn',
                   x_label='number of friends',
                   y_label='avg acc', x_data_list=x_data_list, y_data_list=y_data_list, saving_path=saving_path)

        y_data_list = [dict_result['l2 dist']]
        saving_path = os.path.join(base_output_path, 'f_hat_dist_f_vs_num_friends.png')
        plot_graph(title='f_hat dist from f vs number of random friends to learn',
                   x_label='number of friends',
                   y_label='dist l2', x_data_list=x_data_list, y_data_list=y_data_list, saving_path=saving_path)

        y_data_list = [dict_result['angel_f_f_hat']]
        saving_path = os.path.join(base_output_path, 'angle_between_f_hat_and_f_vs_num_friends.png')
        plot_graph(title='angle between f_hat and f vs number of random friends to learn',
                   x_label='number of friends',
                   y_label='angle', x_data_list=x_data_list, y_data_list=y_data_list, saving_path=saving_path)


    hardt_algo = get_hardt_model(cost_factor, real_train_f_star_loan_status_path, train_hadart)
    feature_list_to_use = six_most_significant_features
    f = load_model(model_loan_returned_path)
    f_vec = np.append(f.coef_[0], f.intercept_)
    # test_pred_loans_status_hardt = hardt_algo(fake_test_df)
    dict_result = init_dict_result()

    real_train_val_f_star_df, real_test_f_star_df, real_train_val_f_loan_status, test_pred_loans_status_f = \
        get_datasets_and_f_grade(f, real_train_val_f_star_loan_status_path, real_test_f_star_loan_status_path)

    parent_folder_path, base_output_path, friends_dict_dir_path = create_paths_and_dirs_for_random_friends_experiment()

    for num_friend in dict_result['number_of_friends_to_learn_list']:
        print(num_friend)
        member_friend_dict_path = os.path.join(friends_dict_dir_path, f'member_friends_{num_friend}friends.json')
        member_dict = create_member_friends_dict(num_friend, real_train_val_f_loan_status,
                                                  real_test_f_star_df, member_friend_dict_path, force_to_crate=False) #todo: change it to false

        cost_func_for_gaming = MixWeightedLinearSumSquareCostFunction(a, epsilon=epsilon, cost_factor=cost_factor)
        friends_modify_strategic_data, f_hat_acc, avg_l2_f_dist, avg_angle = strategic_modify_learn_from_friends(real_test_f_star_df,
                                                                            test_pred_loans_status_f,
                                                                            real_train_val_f_star_df,
                                                                            real_train_val_f_loan_status,
                                                                            feature_list_to_use, cost_func_for_gaming,
                                                                            target_label='LoanStatus',
                                                                            member_dict=member_dict,
                                                                            f_vec=f_vec,
                                                                            dir_name_for_saving_visualize=base_output_path,
                                                                            title_for_visualization=f'real test learned {num_friend}'
                                                                            )
        update_dict_result()

    with open(os.path.join(base_output_path, 'dict_result.json'), 'w') as json_file:
        json.dump(dict_result, json_file, indent=4)
    plot_dict_result_graph()

def create_main_folders():
    os.makedirs(result_folder_path, exist_ok=True)
    os.makedirs(models_folder_path, exist_ok=True)

if __name__ == '__main__':
    cost_factor = 10
    # create_main_folders()
    # main_create_synthetic_data()
    run_strategic_full_info(train_hardt=False, cost_factor=cost_factor, epsilon=0.3)
    print(10)
    strategic_random_friends_info(train_hadart=False, cost_factor=cost_factor, epsilon=0.3)