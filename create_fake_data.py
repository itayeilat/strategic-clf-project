import pandas as pd
import pickle
from collections import Counter
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as LR
from train_member_clf import *
import lightgbm as lgb
from tqdm import tqdm
import random
from model import *
from cost_functions import *
from utills_and_consts import *


def learn_lgb_model(out_path):
    # lgb_params = [{'metric': [['l2', 'auc']], 'learning_rate': [0.005, 0.01, 0.05], 'feature_fraction': [0.6, 0.9],
    #                'bagging_fraction': [0.5, 0.7, 0.9], 'bagging_freq': [10, 100], 'max_depth': [4, 8, 16],
    #                'num_leaves': [64, 128, 256], 'max_bin': [128, 256, 512, 1024], "num_iterations": [1000, 10000],
    #                'n_estimators': [100, 1000]}]

    lgb_params = [{'metric': [['l2', 'auc']], 'learning_rate': [0.01], 'feature_fraction': [0.6],
                   'bagging_fraction': [0.9], 'bagging_freq': [10], 'max_depth': [4],
                   'num_leaves': [64], 'max_bin': [512], "num_iterations": [10000],
                   'n_estimators': [100]}]

    train_df = pd.read_csv(orig_train_path)
    val_df = pd.read_csv(orig_val_path)
    test_df = pd.read_csv(orig_test_path)
    target_label = 'CreditGrade'

    y_train = train_df['CreditGrade']
    X_train = train_df[feature_list_for_pred]


    clf = GridSearchCV(lgb.LGBMRegressor(), lgb_params, cv=5, scoring='r2')
    clf = clf.fit(X_train, y_train)
    best_params = clf.best_params_
    print(f'best params: {best_params}')
    lgb_model = lgb.LGBMRegressor(**best_params)

    r2 = evaluate(lgb_model, train_df, val_df, target_label, feature_list_for_pred)
    print(f'r2 on validation: {r2}')
    train_val_df = pd.concat([train_df, val_df])
    r2 = evaluate(lgb_model, train_val_df, test_df, target_label, feature_list_for_pred)
    print(f'r2 on test: {r2}')
    all_data = pd.concat([train_df, val_df, test_df])
    lgb_model.fit(all_data[feature_list_for_pred], all_data[target_label])


    pickle.dump(lgb_model, open(out_path, 'wb'))


def check_gmm_model(feature_list, orig_train_path, orig_val_path):
    """

    :param feature_list: the features we use when we create the gmm model
    :prints: the function prints aic and bic measure
    """
    train_df = pd.read_csv(orig_train_path)
    val_df = pd.read_csv(orig_val_path)
    gmm_model = GaussianMixture(n_components=16)
    gmm_model.fit(train_df[feature_list])
    print(f'aic on train: {gmm_model.aic(train_df[feature_list])} and bic on train: '
          f'{gmm_model.bic(train_df[feature_list])}')
    print(f'aic on val: {gmm_model.aic(val_df[feature_list])} and bic on val: '
          f'{gmm_model.bic(val_df[feature_list])}')


def generate_dataset(feature_list, trained_model, fake_data_set_size=10000,
                     create_key_features=False, check_gmm=False):
    def get_dist_index(weights_array: np.array):
        return np.random.choice(len(weights_array), fake_data_set_size, p=weights_array)

    def split_df(df: pd.DataFrame):
        train_df, test_val = train_test_split(df, test_size=0.3, random_state=42)
        test_df, val_df = train_test_split(test_val, test_size=0.5, random_state=42)
        return train_df, val_df, test_df


    if check_gmm:
        check_gmm_model(feature_list, orig_train_path, orig_val_path)
    path_list_to_data = [orig_train_path, orig_val_path, orig_test_path]
    orig_data = pd.concat([pd.read_csv(path)[feature_list] for path in path_list_to_data])

    np.random.seed(8)
    gmm_model = GaussianMixture(n_components=8)
    gmm_model.fit(orig_data[feature_list])
    new_data_sets_array_list = list()
    gaussian_index_arr = get_dist_index(gmm_model.weights_)
    for index, rep in Counter(gaussian_index_arr).items():
        new_data_sets_array_list.append(
            np.random.multivariate_normal(mean=gmm_model.means_[index], cov=gmm_model.covariances_[index], size=rep))
    fake_data_set = np.vstack(new_data_sets_array_list)
    fake_data_set = pd.DataFrame(data=fake_data_set, columns=orig_data.columns)

    credit_grade_list = trained_model.predict(fake_data_set[feature_list_for_pred])
    fake_data_set.insert(len(fake_data_set.columns), 'CreditGrade', credit_grade_list, True)
    loan_return_status_list = fake_data_set['CreditGrade'].apply(apply_transform_creditgrade_loan_returned)
    fake_data_set.insert(len(fake_data_set.columns), 'LoanStatus', loan_return_status_list, True)

    if create_key_features:
        fake_data_set.insert(len(fake_data_set.columns), 'MemberKey',
                             ['fakeMem' + str(i) for i in range(fake_data_set_size)], True)
        fake_data_set.insert(len(fake_data_set.columns), 'LoanKey',
                             ['fakeLoan' + str(i) for i in range(fake_data_set_size)], True)

    fake_train_df, fake_val_df, fake_test_df = split_df(fake_data_set)
    fake_train_df.to_csv(fake_train_path)
    fake_val_df.to_csv(fake_val_path)
    fake_test_df.to_csv(fake_test_path)


def apply_transform_creditgrade_loan_returned(credit_grade):
    guss = random.uniform(0, 1)
    loan_tresh = max(0, -0.2942 * credit_grade + 1.4592)
    return -1 if guss < loan_tresh else 1












def main_create_fake_data():

    path_to_CG_model = 'lgb_model_CG.sav'
    train_model = False
    gmm_feature_list = feature_list_for_pred + ['ListingCreationDate']
    if train_model:
        learn_lgb_model(path_to_CG_model)
    cg_trained_model = load_sklearn_model(path_to_CG_model)

    generate_dataset(gmm_feature_list, cg_trained_model, fake_data_set_size=10000, create_key_features=True, check_gmm=True)

    '''
    ####### from here shouldnt be in this file..
    train_loan_return_model(fake_train_path, 'data2/fake_val1.csv', feature_list_for_pred, 'data2/loan_returned_model_LR1.sav')

    a = np.array([1, 1, 1, -1, -1, 1, 1, -1])
    strategic_modify_data(fake_train_path, 'data2/loan_returned_model_LR1.sav', feature_list_for_pred,
                          WeightedLinearCostFunction(a), modify_fake_train_path)
    strategic_modify_data(fake_test_path, 'data2/loan_returned_model_LR1.sav', feature_list_for_pred,
                          WeightedLinearCostFunction(a), modify_fake_test_path)

    acc_fake_test_modify = evaluate_on_modify(train_path=fake_train_path, test_path=modify_fake_test_path,
                                         model_to_train=LR(penalty='none'),
                                         feature_list_to_predict=feature_list_for_pred)
    print(acc_fake_test_modify)
    acc_train_and_test_modify = evaluate_on_modify(train_path=modify_fake_train_path,
                                                   test_path=modify_fake_test_path,
                                                   model_to_train=LR(penalty='none'),
                                                   feature_list_to_predict=feature_list_for_pred)
    print(acc_train_and_test_modify)


    a = np.array([1, 1, 1, -1, -1, 1, 1, -1])
    algo1 = Algo1(WeightedLinearCostFunction(a))

    test_df = pd.read_csv(fake_test_path)
    train_df = pd.read_csv(fake_train_path)
    modify_test_df = pd.read_csv(modify_fake_test_path)

    algo1.fit(train_df[feature_list_for_pred], train_df['LoanStatus'])
    pred_loan_status = algo1(test_df[feature_list_for_pred])

    acc = np.sum(pred_loan_status == test_df['LoanStatus']) / len(test_df)
    print(acc)

    pred_loan_status = algo1(modify_test_df[feature_list_for_pred])

    acc = np.sum(pred_loan_status == modify_test_df['LoanStatus']) / len(test_df)
    print(acc)
    '''








def modify_real_data():
    orig_train_path = 'data/train_pre2009.csv'
    orig_val_path = 'data/val_pre2009.csv'
    orig_test_path = 'data/test_pre2009.csv'
    modify_real_train_path = 'data/modify_real_train.csv'
    modify_real_test_path = 'data/modify_real_test.csv'
    model_loan_returned_path = 'data/loan_returned_model_LR_real_data.sav'
    list_to_keep, feature_list_for_pred = get_feature_list_and_keeping_list()
    train_loan_return_model(orig_train_path, orig_val_path, feature_list_for_pred, model_loan_returned_path)

    a = np.array([1, 1, 1, -1, -1, 1, 1, -1])

    strategic_modify_data(orig_train_path, model_loan_returned_path, feature_list_for_pred, WeightedLinearCostFunction(a), modify_real_train_path)

    strategic_modify_data(orig_test_path, model_loan_returned_path, feature_list_for_pred, WeightedLinearCostFunction(a), modify_real_test_path)

    acc_test_modify = evaluate_on_modify(train_path=orig_train_path, test_path=modify_real_test_path,
                                         model_to_train=LR(penalty='none'),
                                         feature_list_to_predict=feature_list_for_pred)
    print(acc_test_modify)
    acc_train_and_test_modify = evaluate_on_modify(train_path=modify_real_train_path,
                                                   test_path=modify_real_test_path,
                                                   model_to_train=LR(penalty='none'),
                                                   feature_list_to_predict=feature_list_for_pred)
    print(acc_train_and_test_modify)

    a = np.array([1, 1, 1, -1, -1, 1, 1, -1])
    algo1 = Algo1(WeightedLinearCostFunction(a))
    test_df = pd.read_csv(orig_test_path)
    train_df = pd.read_csv(orig_train_path)
    modify_test_df = pd.read_csv(modify_real_test_path)


    algo1.fit(train_df[feature_list_for_pred], train_df['LoanStatus'])
    pred_loan_status = algo1(test_df[feature_list_for_pred])

    acc = np.sum(pred_loan_status == test_df['LoanStatus']) / len(test_df)
    print(acc)

    pred_loan_status = algo1(modify_test_df[feature_list_for_pred])

    acc = np.sum(pred_loan_status == modify_test_df['LoanStatus']) / len(modify_test_df)
    print(acc)



#from startegic_players import strategic_modify_data

def check_for_now():
    fake_train_path = 'data2/fake_train1.csv'
    fake_test_path = 'data2/fake_test1.csv'
    modify_fake_train_path = 'data2/modify_fake_train1.csv'
    modify_fake_test_path = 'data2/modify_fake_test1.csv'
    list_to_keep, feature_list_for_pred = get_feature_list_and_keeping_list()
    gmm_feature_list = feature_list_for_pred + ['ListingCreationDate']
    # poly_svr = load_sklearn_model('data2/linear_poly_svr.sav')
    # generate_dataset(['data2/train_pre2009.csv', 'data2/val_pre2009.csv', 'data2/test_pre2009.csv'], gmm_feature_list,
    #                  poly_svr, feature_list_for_pred, fake_data_set_size=10000, create_key_features=True,
    #                  check_gmm=False)
    # train_loan_return_model(fake_train_path, 'data2/fake_val1.csv', feature_list_for_pred,
    #                         'data2/loan_returned_model_LR1.sav')
    #
    # stratigic_modify_data(fake_train_path, 'data2/loan_returned_model_LR1.sav', feature_list_for_pred,
    #                       out_path=modify_fake_train_path)
    # stratigic_modify_data(fake_test_path, 'data2/loan_returned_model_LR1.sav', feature_list_for_pred,
    #                       out_path=modify_fake_test_path)


    a = np.array([1, 1, 1, -1, -1, 1, 1, -1])
    algo1 = Algo1(WeightedLinearCostFunction(a))
    train_df, test_df = pd.read_csv(fake_train_path), pd.read_csv(modify_fake_test_path)
    algo1.fit(train_df[feature_list_for_pred], train_df['LoanStatus'])
    pred_loan_status = algo1(test_df[feature_list_for_pred])
    # acc = evaluate_with_gaming_algorithm(fake_train_path, modify_fake_test_path, feature_list_for_pred)
    # print(acc)
    acc = np.sum(pred_loan_status == test_df['LoanStatus']) / len(test_df)
    print(acc)




if __name__ == '__main__':
    main_create_fake_data()



