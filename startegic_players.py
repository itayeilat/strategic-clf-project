import pandas as pd
import numpy as np
from train_member_clf import load_sklearn_model
from cost_functions import *
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LinearRegression

import pickle
from utills_and_consts import *
import matplotlib.pyplot as plt


def visualize_projected_changed_df(before_change_df_path, after_change_df, features_to_project, title, label='LoanStatus',
                                   num_point_to_plot=100, base_dir_path: str = 'result/changed_samples_by_gaming/2D_projection_images',
                                   to_save=True):
    def apply_transform_for_2D(df: pd.DataFrame):
        transform_matrix = np.array([[1.5, 0], [1.5, 0], [2, 0], [0, -5], [0, -1], [1, 1]])
        return df @ transform_matrix



    fig, (ax_before, ax_after) = plt.subplots(1, 2)
    df_before = pd.read_csv(before_change_df_path)
    df_before_loan_status, df_before = df_before[label], df_before[features_to_project]
    # df_after = pd.read_csv(after_change_df_path)[features_to_project]
    df_after = after_change_df[features_to_project]
    fig.suptitle(title)
    ax_before.set_title('before')
    ax_after.set_title('after')



    projected_df_before, projected_df_after = apply_transform_for_2D(df_before), apply_transform_for_2D(df_after)

    # color_list = np.array(list(map(lambda eq: 'tab:green' if eq else 'tab:blue', (df_before == df_after).any(1))))
    #color_list[df_before_loan_status == -1] = 'tab:red'
    # ax_before.scatter(projected_df_before[:, 0], projected_df_before[:, 1], c=color_list.tolist())
    # ax_after.scatter(projected_df_after[:, 0], projected_df_after[:, 1], c=color_list.tolist())

    # ax_before.scatter(projected_df_before[0], projected_df_before[1], c=color_list.tolist())
    # ax_after.scatter(projected_df_after[0], projected_df_after[1], c=color_list.tolist())

    projected_df_after = projected_df_after[(df_before != df_after).any(1)]
    projected_df_before = projected_df_before[(df_before != df_after).any(1)]

    ax_before.scatter(projected_df_before[0][:num_point_to_plot], projected_df_before[1][:num_point_to_plot])
    ax_after.scatter(projected_df_after[0][:num_point_to_plot], projected_df_after[1][:num_point_to_plot])
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.scatter(projected_df_before[0][:num_point_to_plot], projected_df_before[1][:num_point_to_plot], s=10)
    ax.scatter(projected_df_after[0][:num_point_to_plot], projected_df_after[1][:num_point_to_plot], s=10)
    for i, (before_row_tup, after_row_tup) in enumerate(zip(projected_df_before.iterrows(), projected_df_after.iterrows())):
        before_row, after_row = before_row_tup[1], after_row_tup[1]
        plt.arrow(before_row[0], before_row[1], after_row[0] - before_row[0], after_row[1] - before_row[1],
                  shape='full', color='black', length_includes_head=True,
                  zorder=0, head_length=0.1, head_width=0.2)
        if i > num_point_to_plot:
            break
    plt.title(title)
    if to_save:
        saving_path = base_dir_path + '/' + title + '.png'
        plt.savefig(saving_path)
    plt.show()







def evaluate_on_modify(test_path, trained_model_path, feature_list_to_predict, target_label='LoanStatus'):
    f = load_sklearn_model(trained_model_path)
    f_to_keep = feature_list_to_predict + [target_label]
    test_df = pd.read_csv(test_path)[f_to_keep]

    will_loan_returned_pred = f.predict(test_df[feature_list_to_predict])

    return sum(will_loan_returned_pred == test_df[target_label]) / len(will_loan_returned_pred)


def train_loan_return_model(list_features_for_pred, binary_trained_model_path, target_label='LoanStatus'):
    def round_loan_numeric_to_classified(predicted_value):
        return 1 if predicted_value >= 0 else -1

    fake_train_df, fake_val_df, fake_test_df = pd.read_csv(fake_train_path), pd.read_csv(fake_val_path), pd.read_csv(fake_test_path)
    linear_model = LR(penalty='none') # it was chosen empirically with validation set
    fake_train_val = pd.concat([fake_train_df, fake_val_df])
    linear_model.fit(fake_train_val[list_features_for_pred], fake_train_val[target_label])
    y_test_pred = linear_model.predict(fake_test_df[list_features_for_pred])
    acc = np.sum(y_test_pred == fake_test_df[target_label]) / len(y_test_pred)
    print(f'acc on synthetic test:{acc}')
    pickle.dump(linear_model, open(binary_trained_model_path, 'wb'))


def strategic_modify_using_known_clf(orig_df_path: str, binary_trained_model_path, feature_list,
                                     cost_func: CostFunction, out_path=None,
                                     target_label='LoanStatus'):
    orig_df = pd.read_csv(orig_df_path)
    f = load_sklearn_model(binary_trained_model_path)
    modify_data = orig_df[feature_list].copy()

    with tqdm(total=len(orig_df)) as t:
        for (index, ex), label in zip(orig_df[feature_list].iterrows(), orig_df[target_label]):
            x = np.array(ex)
            #if label == -1: #this maybe should be f here..
            if f.predict(x.reshape(1, -1))[0] == -1:
                z = cost_func.maximize_features_against_binary_model(x, f)
                modify_data.loc[index] = z
            t.update(1)
    for col_name in filter(lambda c: c not in modify_data.columns, orig_df.columns):
        modify_data.insert(len(modify_data.columns), col_name, orig_df[col_name], True)

    if out_path is not None:
        modify_data.to_csv(out_path)
    return modify_data


def create_strategic_data_sets_using_known_clf(retrain_model_loan_return=True):
    binary_trained_model_path = 'data/loan_returned_model.sav'
    features_to_use = six_most_significant_features
    a_vec = a[:len(features_to_use)]
    if retrain_model_loan_return:
        train_loan_return_model(features_to_use, binary_trained_model_path)

    weighted_linear_cost = MixWeightedLinearSumSquareCostFunction(a_vec)
    modify_full_information_test = strategic_modify_using_known_clf(fake_test_path, binary_trained_model_path, features_to_use,
                                     weighted_linear_cost, modify_full_information_test_fake_path)
    visualize_projected_changed_df(fake_test_path, modify_full_information_test, features_to_use, 'fake test')

    acc_fake_test_modify = evaluate_on_modify(test_path=modify_full_information_test_fake_path,
                                              trained_model_path=binary_trained_model_path,
                                              feature_list_to_predict=six_most_significant_features)
    print(f'the accuracy on the test set when it trained on not modify train {acc_fake_test_modify}')
    weighted_linear_cost.get_statistic_on_num_change()
    weighted_linear_cost.get_statistic_on_num_change()






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
                                        member_dict: dict, out_path: str = None, title_for_visualization: str = None):
    orig_df = pd.read_csv(orig_df_path)
    friends_df = pd.read_csv(sample_from_df_path)
    modify_data = orig_df[feature_to_learn_list].copy()
    with tqdm(total=len(orig_df)) as t:
        for (index, ex), member_key in zip(orig_df[feature_to_learn_list].iterrows(), orig_df['MemberKey']):
            member_friend_keys = set(friends_df.iloc[member_dict[member_key]["friends with credit data"], :]['MemberKey'])

            friends_and_member_df = friends_df[friends_df['MemberKey'].isin(member_friend_keys)]
            x = np.array(ex)
            f = LR(penalty='none').fit(friends_and_member_df[feature_to_learn_list], friends_and_member_df[target_label])
            if f.predict(x.reshape(1, -1))[0] == -1:
                z = cost_func.maximize_features_against_binary_model(x, f)
                modify_data.loc[index] = z
            t.update(1)
    for col_name in filter(lambda c: c not in modify_data.columns, orig_df.columns):
        modify_data.insert(len(modify_data.columns), col_name, orig_df[col_name], True)

    cost_func.get_statistic_on_num_change()
    visualize_projected_changed_df(fake_test_path, modify_data, feature_to_learn_list, title_for_visualization)
    return modify_data


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




