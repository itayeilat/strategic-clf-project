import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from train_member_clf import *
from startegic_players import *
from cost_functions import *
from create_fake_data import main_create_fake_data


def evaluate_on_modify(train_path, test_path, model_to_train, feature_list_to_predict, target_label='LoanStatus'):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    model_to_train.fit(train_df[feature_list_to_predict], train_df[target_label])
    will_loan_returned_pred = model_to_train.predict(test_df[feature_list_to_predict])
    return sum(will_loan_returned_pred == test_df[target_label]) / len(will_loan_returned_pred)


def run_strategic():
    orig_train_path = 'data2/train_pre2009.csv'
    orig_val_path = 'data2/val_pre2009.csv'
    orig_test_path = 'data2/test_pre2009.csv'
    modify_with_friends_real_train_path = 'data2/modify_with_friends_real_train.csv'
    modify_with_friends_real_test_path = 'data2/modify_with_friends_real_test.csv'
    model_loan_returned_path = 'data2/loan_returned_model_LR_real_data.sav'
    mem_dict_path = 'data2/mem_common_kaggle_friends_dict.json'
    target_label = 'LoanStatus'

    with open(mem_dict_path, 'r') as f:
        mem_dict = json.load(f)

    list_to_keep, feature_list_for_pred = get_feature_list_and_keeping_list()

    a = np.array([1, 1, 1, -1, -1, 1, 1, -1])

    strategic_modify_learn_from_friends(orig_train_path, feature_list_for_pred, WeightedLinearCostFunction(a),
                                        target_label,
                                        mem_dict, modify_with_friends_real_train_path)

    strategic_modify_learn_from_friends(orig_test_path, feature_list_for_pred, WeightedLinearCostFunction(a),
                                        target_label,
                                        mem_dict, modify_with_friends_real_test_path)

    acc_fake_test_modify = evaluate_on_modify(train_path=orig_train_path, test_path=modify_with_friends_real_test_path,
                                              model_to_train=LR(penalty='none'),
                                              feature_list_to_predict=feature_list_for_pred)
    print(acc_fake_test_modify)
    acc_train_and_test_modify = evaluate_on_modify(train_path=modify_with_friends_real_train_path,
                                                   test_path=modify_with_friends_real_test_path,
                                                   model_to_train=LR(penalty='none'),
                                                   feature_list_to_predict=feature_list_for_pred)
    print(acc_train_and_test_modify)

    a_tag = a + np.random.normal(size=len(a))
    hardt_algo = HardtAlgo(WeightedLinearCostFunction(a_tag))

    test_df = pd.read_csv(orig_test_path)
    train_df = pd.read_csv(orig_train_path)
    modify_test_df = pd.read_csv(modify_with_friends_real_test_path)

    hardt_algo.fit(train_df[feature_list_for_pred], train_df['LoanStatus'])
    pred_loan_status = hardt_algo(test_df[feature_list_for_pred])

    acc = np.sum(pred_loan_status == test_df['LoanStatus']) / len(test_df)
    print(acc)

    pred_loan_status = hardt_algo(modify_test_df[feature_list_for_pred])

    acc = np.sum(pred_loan_status == modify_test_df['LoanStatus']) / len(test_df)
    print(acc)






if __name__ == '__main__':
   #main_create_fake_data()
   create_strategic_data_sets_using_known_clf()

   feature_list_to_use = six_most_significant_features
   hardt_algo = HardtAlgo(WeightedLinearCostFunction(a[:len(feature_list_to_use)]))

   test_df = pd.read_csv(fake_test_path)
   train_df = pd.read_csv(fake_train_path)
   modify_test_df = pd.read_csv(modify_full_information_test_fake_path)
   hardt_algo.fit(train_df[feature_list_to_use], train_df['LoanStatus'])

   pred_loan_status = hardt_algo(modify_test_df[feature_list_to_use])

   acc = np.sum(pred_loan_status == modify_test_df['LoanStatus']) / len(test_df)
   print(f'acc on modify test: {acc}')

   pred_loan_status = hardt_algo(test_df[feature_list_to_use])

   acc = np.sum(pred_loan_status == test_df['LoanStatus']) / len(test_df)
   print(f'acc on not modify test: {acc}')




