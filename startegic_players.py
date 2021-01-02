import pandas as pd
import os
import numpy as np
from train_member_clf import load_sklearn_model
from cost_functions import *
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import LinearSVC
import random
from sklearn.linear_model import LinearRegression
from create_synthetic_data import get_f_star_loan_status_real_train_val_test_df


import pickle
from utills_and_consts import *
import matplotlib.pyplot as plt


def visualize_projected_changed_df(before_change_df_path, after_change_df, features_to_project, title, f_weights, f_inter, label='LoanStatus',
                                   num_point_to_plot=100, dir_for_projection_images: str = '2D_projection_images',
                                   to_save=True, dir_name_for_saving_visualize=None):
    def apply_transform_for_2D(df: pd.DataFrame):
        array_list = list()
        # for item in f_weights[:int(len(f_weights) / 2)]:
        #     array_list.append([item, 0])
        # for item in f_weights[int(len(f_weights) / 2):]:
        #     array_list.append([0, item])
        # transform_matrix = np.vstack(array_list)
        axis_x_indecies = [0, 1, 2]
        axis_y_indecies = [3, 4, 5]
        for i in range(len(f_weights)):
            if i in axis_x_indecies:
                array_list.append([f_weights[i], 0])
            else:
                array_list.append([0, f_weights[i]])

        # for i in axis_x_indecies:
        #     array_list.append([f_weights[i], 0])
        # for i in axis_y_indecies:
        #     array_list.append([0, f_weights[i]])
        transform_matrix = np.vstack(array_list)
        return df @ transform_matrix

    os.makedirs(dir_name_for_saving_visualize, exist_ok=True)
    dir_for_projection_images = os.path.join(dir_name_for_saving_visualize, dir_for_projection_images)
    fig, (ax_before, ax_after) = plt.subplots(1, 2)
    df_before = pd.read_csv(before_change_df_path)
    df_before_loan_status, df_before = df_before[label], df_before[features_to_project]
    df_after = after_change_df[features_to_project]
    fig.suptitle(title)
    ax_before.set_title('before')
    ax_after.set_title('after')


    projected_df_before, projected_df_after = apply_transform_for_2D(df_before), apply_transform_for_2D(df_after)


    # projected_df_after = projected_df_after[(df_before != df_after).any(1)]
    # projected_df_before = projected_df_before[(df_before != df_after).any(1)]
    np.random.seed(4)
    ax_before.scatter(projected_df_before[0][:num_point_to_plot], projected_df_before[1][:num_point_to_plot], color='green')
    ax_after.scatter(projected_df_after[0][:num_point_to_plot], projected_df_after[1][:num_point_to_plot], color='orange')
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.scatter(projected_df_before[0][:num_point_to_plot], projected_df_before[1][:num_point_to_plot], s=10)
    ax.scatter(projected_df_after[0][:num_point_to_plot], projected_df_after[1][:num_point_to_plot], s=10)
    for i, (before_row_tup, after_row_tup) in enumerate(zip(projected_df_before.iterrows(), projected_df_after.iterrows())):
        before_row, after_row = before_row_tup[1], after_row_tup[1]
        plt.arrow(before_row[0], before_row[1], after_row[0] - before_row[0], after_row[1] - before_row[1],
                  shape='full', color='black', length_includes_head=True,
                  zorder=0, head_length=0.1, head_width=0.05)
        if i > num_point_to_plot:
            break

    left_bound, right_bound = -1, 3
    bottom_bound, up_bound = 1, 3
    t = np.arange(left_bound, right_bound, 0.2)
    plt.plot(t, -t - f_inter, color='blue')
    plt.xlim([left_bound, right_bound])
    plt.ylim([bottom_bound, up_bound])
    plt.title(title)
    if to_save:
        saving_path = os.path.join(dir_for_projection_images, title + '.png')
        os.makedirs(dir_for_projection_images, exist_ok=True)
        plt.savefig(saving_path)
    plt.show()


def evaluate_on_modify(test_path, trained_model_path, feature_list_to_predict, target_label='LoanStatus'):
    f = load_sklearn_model(trained_model_path)
    f_to_keep = feature_list_to_predict + [target_label]
    test_df = pd.read_csv(test_path)[f_to_keep]

    will_loan_returned_pred = f.predict(test_df[feature_list_to_predict])

    return sum(will_loan_returned_pred == test_df[target_label]) / len(will_loan_returned_pred)


def train_loan_return_model(list_features_for_pred, binary_trained_model_path, target_label='LoanStatus'):
    train_df, val_df, test_df, train_val_df = get_f_star_loan_status_real_train_val_test_df()
    # linear_model = LR(penalty='none', random_state=42) # it was chosen empirically with validation set
    linear_model = LinearSVC(penalty='l2', random_state=42)

    linear_model.fit(train_val_df[list_features_for_pred], train_val_df[target_label])
    y_test_pred = linear_model.predict(test_df[list_features_for_pred])
    acc = np.sum(y_test_pred == test_df[target_label]) / len(y_test_pred)
    print(f'acc on pre2009 test:{acc}')
    pickle.dump(linear_model, open(binary_trained_model_path, 'wb'))


def strategic_modify_using_known_clf(orig_df_path: str, f, feature_list,
                                     cost_func: CostFunction, out_path=None,
                                     target_label='LoanStatus'):
    orig_df = pd.read_csv(orig_df_path)

    modify_data = orig_df[feature_list].copy()

    with tqdm(total=len(orig_df)) as t:
        for (index, ex), label in zip(orig_df[feature_list].iterrows(), orig_df[target_label]):
            x = np.array(ex)
            if f.predict(x.reshape(1, -1))[0] == -1:
                z = cost_func.maximize_features_against_binary_model(x, f)
                modify_data.loc[index] = z
            t.update(1)
    for col_name in filter(lambda c: c not in modify_data.columns, orig_df.columns):
        modify_data.insert(len(modify_data.columns), col_name, orig_df[col_name], True)

    if out_path is not None:
        modify_data.to_csv(out_path)
    return modify_data

#todo: create one function for each line here..
def create_data_with_f_loan_status(f: LinearSVC, force_to_create=True):
    if force_to_create or os.path.exists(real_train_f_loan_status_path):
        train_df = pd.read_csv(real_train_f_star_loan_status_path)
        train_df['LoanStatus'] = f.predict(train_df[six_most_significant_features])
        train_df.to_csv(real_train_f_loan_status_path)

    if force_to_create or os.path.exists(real_val_f_loan_status_path):
        val_df = pd.read_csv(real_val_f_star_loan_status_path)
        val_df['LoanStatus'] = f.predict(val_df[six_most_significant_features])
        val_df.to_csv(real_val_f_loan_status_path)

    if force_to_create or os.path.exists(real_test_f_loan_status_path):
        test_df = pd.read_csv(real_test_f_star_loan_status_path)
        test_df['LoanStatus'] = f.predict(test_df[six_most_significant_features])
        test_df.to_csv(real_test_f_loan_status_path)

    if force_to_create or os.path.exists(real_train_val_f_loan_status_path):
        train_val_df = pd.read_csv(real_train_val_f_star_loan_status_path)
        train_val_df['LoanStatus'] = f.predict(train_val_df[six_most_significant_features])
        train_val_df.to_csv(real_train_val_f_loan_status_path)


def create_strategic_data_sets_using_known_clf(dir_name_for_saving_visualize, cost_factor, epsilon, retrain_model_loan_return=False, save_visualize_projected_changed=True):

    binary_trained_model_path = model_loan_returned_path
    features_to_use = six_most_significant_features
    a_vec = a[:len(features_to_use)]
    if retrain_model_loan_return or os.path.exists(binary_trained_model_path) is False:
        train_loan_return_model(features_to_use, binary_trained_model_path)

    f = load_sklearn_model(binary_trained_model_path)
    create_data_with_f_loan_status(f)
    f_weights, f_inter = f.coef_[0], f.intercept_


    weighted_linear_cost = MixWeightedLinearSumSquareCostFunction(a_vec, cost_factor=cost_factor, epsilon=epsilon)
    modify_full_information_test = strategic_modify_using_known_clf(real_test_f_star_loan_status_path, f, features_to_use,
                                                                    weighted_linear_cost, modify_full_information_real_test_path)
    visualize_projected_changed_df(real_test_f_star_loan_status_path, modify_full_information_test, features_to_use, 'real test',
                                   to_save=save_visualize_projected_changed, dir_name_for_saving_visualize=dir_name_for_saving_visualize, f_weights=f_weights, f_inter=f_inter)

    acc_fake_test_modify = evaluate_on_modify(test_path=modify_full_information_real_test_path,
                                              trained_model_path=binary_trained_model_path,
                                              feature_list_to_predict=six_most_significant_features)
    print(f'the accuracy on the test set when it trained on not modify train {acc_fake_test_modify}')
    print(f'angle: {np.arccos(a @ f.coef_[0] / (np.linalg.norm(a) * np.linalg.norm(f.coef_[0]))) * 180/np.pi}')
    weighted_linear_cost.get_statistic_on_num_change()
    return modify_full_information_test





def for_each_member_row(row_member, friends_and_member_df: pd.DataFrame, features_to_learn: list,
                        cost_func: CostFunction, target_label: str, filter_by_creation_date=False):
    row_listing_creation_date = row_member['ListingCreationDate']
    if filter_by_creation_date:
        rows_to_learn_df = friends_and_member_df[friends_and_member_df['ListingCreationDate'] < row_listing_creation_date]
    else:
        rows_to_learn_df = friends_and_member_df

    num_changed = 0
    if len(rows_to_learn_df) > 0 and len(rows_to_learn_df[target_label].value_counts()) == 2:
        num_changed += 1
        linear_model = LR()
        linear_model.fit(rows_to_learn_df[features_to_learn], rows_to_learn_df[target_label])
        x = np.array(row_member[features_to_learn])
        z = cost_func.maximize_features_against_binary_model(x, linear_model)
        row_member[features_to_learn] = z


    return row_member, num_changed



def strategic_modify_learn_from_friends(orig_df_path: str, sample_from_df_path: str, feature_to_learn_list, cost_func: CostFunction, target_label,
                                        member_dict: dict, f_weights, f_inter, dir_name_for_saving_visualize: str = None, title_for_visualization: str = None):
    f = load_sklearn_model(model_loan_returned_path)
    counter = 0 #todo del..
    sum_acc = 0
    orig_df = pd.read_csv(orig_df_path)
    friends_df = pd.read_csv(sample_from_df_path)
    modify_data = orig_df[feature_to_learn_list].copy()
    sum_l2_norm = 0
    sum_angle_f_hat_f = 0
    with tqdm(total=len(orig_df)) as t:
        for (index, ex), member_key in zip(orig_df[feature_to_learn_list].iterrows(), orig_df['MemberKey']):
            member_friend_keys = set(friends_df.iloc[member_dict[member_key]["friends with credit data"], :]['MemberKey'])

            friends_and_member_df = friends_df[friends_df['MemberKey'].isin(member_friend_keys)]
            x = np.array(ex)
            f_hat = LinearSVC(penalty='l2', random_state=42, max_iter=10000).fit(friends_and_member_df[feature_to_learn_list],
                                                                                 f.predict(friends_and_member_df[feature_to_learn_list]))

            sum_acc += np.sum(f_hat.predict(orig_df[feature_to_learn_list]) == f.predict(orig_df[feature_to_learn_list])) / len(orig_df)
            v1, v2 = np.append(f_hat.coef_[0], f_hat.intercept_), np.append(f.coef_[0], f.intercept_)
            sum_l2_norm += np.linalg.norm(v1 - v2)

            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            sum_angle_f_hat_f += np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / np.pi

            # f_hat = LR(penalty='none', random_state=42).fit(friends_and_member_df[feature_to_learn_list], friends_and_member_df[target_label])

            # f_hat = LR(penalty='l2', random_state=42).fit(friends_and_member_df[feature_to_learn_list],
            #                                                   f.predict(friends_and_member_df[feature_to_learn_list]))
            # f_hat = LinearRegression().fit(friends_and_member_df[feature_to_learn_list],
            #                                                   f.predict(friends_and_member_df[feature_to_learn_list]))

            ##################################
            # fake_train_df, fake_val_df = pd.read_csv(synthetic_train_path), pd.read_csv(synthetic_val_path)
            # fake_test_df = pd.read_csv(synthetic_test_path)[feature_to_learn_list]
            # fake_test_predictions = f.predict(fake_test_df)
            # linear_model = LR(penalty='none', random_state=42)  # it was chosen empirically with validation set
            # fake_train_val = pd.concat([fake_train_df, fake_val_df])
            # linear_model.fit(fake_train_val[feature_to_learn_list], fake_train_val[target_label])
            # acc_linear_model = np.sum(linear_model.predict(fake_test_df) == fake_test_predictions) / len(fake_test_predictions)
            # acc_f_hat = np.sum(f_hat.predict(fake_test_df) == fake_test_predictions) / len(fake_test_predictions)

            ##################################

            if f_hat.predict(x.reshape(1, -1))[0] == -1:
                counter += 1
                z = cost_func.maximize_features_against_binary_model(x, f_hat)
                modify_data.loc[index] = z

            t.update(1)
    print(f'counter is: {counter}')
    for col_name in filter(lambda c: c not in modify_data.columns, orig_df.columns):
        modify_data.insert(len(modify_data.columns), col_name, orig_df[col_name], True)

    cost_func.get_statistic_on_num_change()
    visualize_projected_changed_df(real_test_f_star_loan_status_path, modify_data, feature_to_learn_list, title_for_visualization,
                                   dir_name_for_saving_visualize=dir_name_for_saving_visualize, f_weights=f_weights, f_inter=f_inter)
    return modify_data, sum_acc/len(orig_df), sum_l2_norm/(len(orig_df)), sum_angle_f_hat_f/len(orig_df)


# def strategic_modify_learn_from_friends(orig_df_path: str, sample_from_df_path: str, feature_to_learn_list, cost_func: CostFunction, target_label,
#                                         member_dict: dict, out_path: str = None, title_for_visualization: str = None):
#     orig_df = pd.read_csv(orig_df_path)
#     friends_df = pd.read_csv(sample_from_df_path)
#     modify_data = orig_df[feature_to_learn_list].copy()
#     f_star = load_sklearn_model(model_loan_returned_path)
#     num_with_neg_pred, num_changed_to_f_star = 0, 0
#     num_that_make_f_star_better = 0
#     num_changed_features = 0
#     with tqdm(total=len(orig_df)) as t:
#         for (index, ex), member_key, label in zip(orig_df[feature_to_learn_list].iterrows(), orig_df['MemberKey'], orig_df[target_label]):
#             member_friend_keys = set(friends_df.iloc[member_dict[member_key]["friends with credit data"], :]['MemberKey'])
#
#             friends_and_member_df = friends_df[friends_df['MemberKey'].isin(member_friend_keys)]
#             x = np.array(ex)
#             f = LR(penalty='none').fit(friends_and_member_df[feature_to_learn_list], friends_and_member_df[target_label])
#             if f.predict(x.reshape(1, -1))[0] == -1:
#                 num_with_neg_pred += 1
#                 z = cost_func.maximize_features_against_binary_model(x, f)
#                 if f_star.predict(x.reshape(1, -1))[0] != f_star.predict(z.reshape(1, -1))[0]:
#                     if label == f_star.predict(z.reshape(1, -1))[0]:
#                         num_that_make_f_star_better += 1
#                     num_changed_to_f_star += 1
#                 if (x != z).any():
#                     num_changed_features += 1
#                 modify_data.loc[index] = z
#             t.update(1)
#     for col_name in filter(lambda c: c not in modify_data.columns, orig_df.columns):
#         modify_data.insert(len(modify_data.columns), col_name, orig_df[col_name], True)
#
#     print(f'the number that thought that they are neg is: {num_with_neg_pred} and the number that changed f_star is {num_changed_to_f_star}'
#           f'and number that made f better is {num_that_make_f_star_better} number that changed features is: {num_changed_features}')
#     cost_func.get_statistic_on_num_change()
#     visualize_projected_changed_df(fake_test_path, modify_data, feature_to_learn_list, title_for_visualization)
#     return modify_data




