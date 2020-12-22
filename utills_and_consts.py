import numpy as np
def evaluate(model, training_df, testing_df, label, list_for_pred):
    model.fit(training_df[list_for_pred], training_df[label])
    y_true = testing_df[label]
    y_pred = model.predict(testing_df[list_for_pred])
    r2 = 1 - ((y_pred - y_true) ** 2).sum() / ((y_true.mean() - y_true) ** 2).sum()
    return r2



orig_train_path = 'data/train_pre2009.csv'
orig_val_path = 'data/val_pre2009.csv'
orig_test_path = 'data/test_pre2009.csv'
fake_all_data = 'data/fake_dataset.csv'
fake_train_path = 'data/fake_train.csv'
fake_val_path = 'data/fake_val.csv'
fake_test_path = 'data/fake_test.csv'
fake_set_to_sample_from_path = 'data/fake_set_to_sample_from_set.csv'
modify_full_information_train_fake_path = 'data/modify_full_information_fake_train.csv'
modify_full_information_val_fake_path = 'data/modify_full_information_fake_val.csv'
modify_full_information_test_fake_path = 'data/modify_full_information_fake_test.csv'

model_loan_returned_path = 'models/loan_returned_model.sav'
hardt_model_path = 'models/Hardt_model.sav'

#a = 10 * np.array([10000, 1, 100000, -10000000, -1, 1000000000, 1, -1])
# a = np.array([1, 1, 1, -1, -1, 1, 1, -1])
# a = np.array([0.6573, 0.664, 3.72, -6.15, -0.404, 0.56])
a = np.array([0.5, 0.5, 1, -2, -0.5, 0.5])

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
