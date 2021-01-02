import numpy as np
def evaluate(model, training_df, testing_df, label, list_for_pred):
    model.fit(training_df[list_for_pred], training_df[label])
    y_true = testing_df[label]
    y_pred = model.predict(testing_df[list_for_pred])
    r2 = 1 - ((y_pred - y_true) ** 2).sum() / ((y_true.mean() - y_true) ** 2).sum()
    return r2


#todo: change orig-> real
orig_train_path = 'data/train_pre2009.csv'
orig_val_path = 'data/val_pre2009.csv'
orig_test_path = 'data/test_pre2009.csv'
orig_train_val_path = 'data/train_val_pre2009.csv'

real_train_f_star_loan_status_path = 'data/train_pre2009_f_star_loan_status.csv'
real_val_f_star_loan_status_path = 'data/val_pre2009_f_star_loan_status.csv'
real_test_f_star_loan_status_path = 'data/test_pre2009_f_star_loan_status.csv'
real_train_val_f_star_loan_status_path = 'data/train_val_pre2009_f_star_loan_status.csv'

real_train_f_loan_status_path = 'data/train_pre2009_f_loan_status.csv'
real_val_f_loan_status_path = 'data/val_pre2009_f_loan_status.csv'
real_test_f_loan_status_path = 'data/test_pre2009_f_loan_status.csv'
real_train_val_f_loan_status_path = 'data/train_val_pre2009_f_loan_status.csv'

synthetic_all_data = 'data/synthetic_dataset.csv'
synthetic_train_path = 'data/synthetic_train.csv'
synthetic_val_path = 'data/synthetic_val.csv'
synthetic_train_val_path = 'data/synthetic_train_val.csv'
synthetic_test_path = 'data/synthetic_test.csv'
#synthetic_set_to_sample_from_path = 'data/synthetic_set_to_sample_from_set.csv'
synthetic_set_to_sample_from_path = 'data/synthetic_set_to_sample_from_set2.csv'
modify_full_information_train_synthetic_path = 'data/modify_full_information_synthetic_train.csv'
modify_full_information_val_synthetic_path = 'data/modify_full_information_synthetic_val.csv'
modify_full_information_test_synthetic_path = 'data/modify_full_information_synthetic_test.csv'

modify_full_information_real_test_path = 'data/modify_full_information_real_test.csv'

model_loan_returned_path = 'models/loan_returned_model.sav'
# hardt_model_path = 'models/Hardt_model.sav'
models_folder_path = 'models'
result_folder_path = 'result'


# a = np.array([0.5, 0.5, 1, -2, -0.5, 0.5]) #the a we use for logistic regression
# a = np.array([0.1, 0.1, 1, -2, -0.1, 0.1])
a = np.array([0.5, 0.5, 1.5, -3, -0.5, 0.5])
# a = np.array([0.21, 0.179, 1.3, -2.4, -0.14, 0.148]) # this is svm f
# a = np.array([0.57323717, 0.7151804, 3.18898121, -5.54032394, -0.38288026, 0.54179837]) #the a that is f


feature_list_for_pred = ['TotalTrades', 'TotalInquiries',
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
                        'CreditHistoryLength'
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

six_most_significant_features = ['AvailableBankcardCredit', 'LoanOriginalAmount', 'TradesNeverDelinquent(percentage)',
                                'BankcardUtilization', 'TotalInquiries', 'CreditHistoryLength']

eight_most_significant_features = six_most_significant_features + ['IsBorrowerHomeowner', 'DebtToIncomeRatio']

# run_name = 'result/changed_samples_by_gaming'
run_name = 'result/changed_samples_by_gaming_cost_factor=10'