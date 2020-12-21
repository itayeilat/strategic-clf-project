import pandas as pd
import json
from tqdm import tqdm
import os
import sklearn
import pickle
import numpy as np
from cost_functions import *
from model import *


def get_feature_list_and_keeping_list():
    # list_to_keep = ['CreditGrade', 'AvailableBankcardCredit', 'LoanOriginalAmount', 'TradesNeverDelinquent(percentage)',
    #                 'BankcardUtilization', 'TotalInquiries', 'CreditHistoryLength', 'IsBorrowerHomeowner',
    #                 'DebtToIncomeRatio', 'ListingCreationDate', 'LoanKey', 'MemberKey']

    # feature_list = ['AvailableBankcardCredit', 'LoanOriginalAmount', 'TradesNeverDelinquent(percentage)',
    #                 'BankcardUtilization', 'TotalInquiries', 'CreditHistoryLength', 'IsBorrowerHomeowner',
    #                 'DebtToIncomeRatio'
    #                 ]

    feature_list = [ 'TotalTrades', 'TotalInquiries',
        'AvailableBankcardCredit', 'BankcardUtilization', 'AmountDelinquent',
        'IncomeRange', 'LoanOriginalAmount',
        'MonthlyLoanPayment', 'StatedMonthlyIncome', 'DebtToIncomeRatio',
        'TradesNeverDelinquent(percentage)', 'TradesOpenedLast6Months',
        'RevolvingCreditBalance', 'CurrentlyInGroup',
        'IsBorrowerHomeowner',
        'PublicRecordsLast10Years', 'PublicRecordsLast12Months',
        'CurrentCreditLines',
        'OpenCreditLines',
        'OpenRevolvingAccounts',
        'CreditHistoryLength',
        ]


    list_to_keep = ['CreditGrade', 'TotalTrades', 'TotalInquiries',
                    'AvailableBankcardCredit', 'BankcardUtilization', 'AmountDelinquent',
                    'IncomeRange', 'LoanOriginalAmount',
                    'MonthlyLoanPayment', 'StatedMonthlyIncome', 'DebtToIncomeRatio',
                    'TradesNeverDelinquent(percentage)', 'TradesOpenedLast6Months',
                    'RevolvingCreditBalance', 'CurrentlyInGroup',
                    'IsBorrowerHomeowner',
                    'PublicRecordsLast10Years', 'PublicRecordsLast12Months',
                    'CurrentCreditLines', 'OpenCreditLines',
                    'OpenRevolvingAccounts',
                    'CreditHistoryLength',
                    'MemberKey', 'LoanKey', 'ListingCreationDate'
                    ]


    return list_to_keep, feature_list

def create_complete_kaggle_data(paths_list: list, list_features: list, map_creditGrade=True):
    assert len(paths_list) > 0
    # return pd.read_csv('data2/train_pre2009.csv')
    df = pd.concat([pd.read_csv(path)[list_features] for path in paths_list])
    if map_creditGrade:
        df['CreditGrade'] = df['CreditGrade'].map({
            3: 3, 4: 2, 2: 4, 0: 6, 6: 0, 1: 5, 5: 1, 7: -1
        })
    return df


def get_member_friend_dict(path: str):
    """
    :param path: the path to the json file to load the dictionary
    :return: dictionary where keys are member keys and values are dict with one key: "friends with credit data" which is
     list pf keys
    """
    with open(path, 'r') as f:
        return json.load(f)


def convert_date_month_to_numeric(df: pd.DataFrame, feature_name: str):
    def convert_date_month(date: str, base_year: int =1900):
        date = date.split(' ')[0]
        year, month, day = tuple(date.split('-'))
        return (int(year) - base_year) * 12 + int(month) + int(day) / 31

    df[feature_name] = df[feature_name].apply(convert_date_month)
    return df


def load_sklearn_model(path: str):
    return pickle.load(open(path, 'rb'))


def get_X_y_data(data: pd.DataFrame, list_to_learn: list, target_feature: str):
    X = data[list_to_learn]
    y = data[target_feature]
    return X, y


def for_each_model_cost_func(model_init_func, cost_func: CostFunction, rows_to_learn_df: pd.DataFrame,
                             features_to_learn: list, target_feature, row_member, trained_model):
    poly = PolynomialFeatures(2, include_bias=True)
    model_result = dict()
    model = model_init_func(cost_func)
    model_result['model name'] = type(model).__name__
    model_result['cost function'] = type(cost_func).__name__
    model_result['original features'] = row_member[features_to_learn].to_dict()
    if len(rows_to_learn_df) > 0:
        model.train(*get_X_y_data(rows_to_learn_df, features_to_learn, target_feature))
        new_y, ok_check = model.get_strategic_new_feature(model_result['original features'])
        model_result['ok check'] = ok_check
        model_result['learned'] = True
    else:
        model_result['learned'] = False
        model_result['ok check'] = True
        new_y = model_result['original features']
    model_result['new y'] = new_y
    model_result['new CreditGrade'] = trained_model.predict(poly.fit_transform(pd.DataFrame.from_dict(new_y, orient='index').T)).item()
    return model_result


def for_each_member_row(row_member, friends_and_member_df: pd.DataFrame, trained_model, features_to_learn: list,
                        model_cost_func_list: list, target_feature: str, member_key):
    poly = PolynomialFeatures(2, include_bias=True)
    loan_dict = dict()
    loan_dict['LoanKey'], loan_dict['real CreditGrade'] = row_member['LoanKey'], row_member['CreditGrade']
    loan_dict['model grade'] = trained_model.predict(poly.fit_transform(pd.DataFrame(row_member[features_to_learn]).T)).item() # todo: change it without poly
    row_listing_creation_date = row_member['ListingCreationDate']
    rows_to_learn_df = friends_and_member_df[friends_and_member_df['ListingCreationDate'] < row_listing_creation_date]
    # rows_to_learn_df = friends_and_member_df
    loan_dict['Number Loans To Learn From'] = len(rows_to_learn_df)
    loan_dict['model results'] = list()
    for model_init_func, cost_func in model_cost_func_list:
        if len(rows_to_learn_df) > 4:
            print(f'key: {member_key} learned: {len(rows_to_learn_df)}')
        model_result = for_each_model_cost_func(model_init_func, cost_func, rows_to_learn_df, features_to_learn,
                                                target_feature, row_member, trained_model)

        loan_dict['model results'].append(model_result)
    return loan_dict


def train_member_friends(data: pd.DataFrame, member_dict: dict, model_cost_func_list: list, result_path: str,
                         member_criterion_func, feature_to_learn: list, target_feature: str, trained_model):
    members_stands_criterion_func = [member_key for member_key in member_dict.keys() if member_criterion_func(member_key, member_dict)]
    output_dict = dict()
    bar = tqdm(total=len(members_stands_criterion_func))
    for member_key in members_stands_criterion_func:
        output_dict[member_key] = dict()
        member_friend = member_dict[member_key]["friends with credit data"]
        output_dict[member_key]['Number Friends'] = len(member_friend)
        member_df = data[data['MemberKey'] == member_key]
        friends_and_member_df = data[data['MemberKey'].isin(set(member_friend).union({member_key}))]
        output_dict[member_key]['Number of Member\'s Loans'] = len(member_df)
        output_dict[member_key]['Number loans to learn from '] = len(friends_and_member_df)
        output_dict[member_key]['Loans Data'] = list()
        for i, (index, row) in enumerate(member_df.iterrows()):
            loan_dict = for_each_member_row(row, friends_and_member_df, trained_model, feature_to_learn,
                                            model_cost_func_list, target_feature, member_key)
            output_dict[member_key]['Loans Data'].append(loan_dict)

        bar.update(1)

    with open(result_path, 'w+') as f:
        json.dump(output_dict, f, indent=4)


def true_criterion(member_key, member_dict):
    # return member_key == 'D1E133954706125454654D8'
    # return member_key == '4BAD3370907307803C76E13'
    return True


def main_tarin_member():
    list_to_keep, feature_list = get_feature_list_and_keeping_list()

    data = create_complete_kaggle_data(['data2/train_pre2009.csv', 'data2/val_pre2009.csv', 'data2/test_pre2009.csv'],
                                       list_to_keep)
    data = convert_date_month_to_numeric(data, 'ListingCreationDate')

    member_dict = get_member_friend_dict('data2/mem_common_kaggle_friends_dict.json')

    poly_svr = load_sklearn_model('data2/linear_poly_svr.sav')

    weighted_list = [('BankcardUtilization', -1), ('TotalInquiries', -1), ('DebtToIncomeRatio', -1)]
    exp_cost_func = WeightedExponentCostFunction(feature_list, weighted_list)
    sum_cost_func = WeightedSumCostFunction(feature_list, weighted_list)
    train_member_friends(data, member_dict,
                         model_cost_func_list=[(SVRPoly2Model, sum_cost_func), (LinearModel, exp_cost_func)],
                         result_path='results/resPoly.json', member_criterion_func=true_criterion,
                         feature_to_learn=feature_list,
                         target_feature='CreditGrade', trained_model=poly_svr)


if __name__ == '__main__':
    main_tarin_member()
