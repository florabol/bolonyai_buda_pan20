# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import time

# In[2]:


data_v1 = pd.read_csv('/Users/Flora/Downloads/drive-download-20200530T201246Z-001/clean_en_data_v1.tsv', delimiter='\t',
                      encoding='utf-8')
data_v2 = pd.read_csv('/Users/Flora/Downloads/drive-download-20200530T201246Z-001/clean_en_data_v2.tsv', delimiter='\t',
                      encoding='utf-8')

# In[ ]:


# load data for cross validation (sampling per user not per tweet to model reality)
X_total = pd.read_csv('/Users/Flora/Downloads/drive-download-20200530T201213Z-001/clean_en_data_v1.tsv', delimiter='\t',
                      encoding='utf-8')
X_all = X_total['Tweets']
y_all = X_total['spreader']

# initializing parameters and df to store results
grid = ParameterGrid({"min_max": [(1, 1), (1, 2), (2, 2)],
                      "min_freq": [3, 4, 5, 6, 7, 8, 9, 10],
                      "B": [100, 300, 400],
                      "min_n": [5, 6, 7, 8, 9, 10]})

rf_results_v1 = pd.DataFrame()
params_list = []
[params_list.append(params) for params in grid]
rf_results_v1["params"] = params_list
rf_results = []

# same split for each round for comparability
cv = StratifiedKFold(5, shuffle=True, random_state=0)

i = 0
for params in grid:
    i = i + 1
    if i % 10 == 0:
        print(i)
        print(time.strftime("%m/%d/%Y %H:%M:%S"))
    min_max, min_freq, B, min_n = params['min_max'], params['min_freq'], params['B'], params['min_n']
    vectorizer = TfidfVectorizer(ngram_range=min_max, min_df=min_freq, sublinear_tf=True)
    X = vectorizer.fit_transform(data_v1['Tweets'])
    y = data_v1['spreader']
    rf_clf = RandomForestClassifier(n_estimators=B, min_samples_leaf=min_n, criterion='gini', random_state=0)
    fold_acc = []  # list to store tweetwise predictions in each fold
    for train_index, test_index in cv.split(X_all, y_all):
        preds = pd.DataFrame()  # df to store results

        # sampling is done on aggregate data, indices needed for individual tweets
        train_indeces = []
        test_indeces = []
        for k in train_index:
            train_indeces.append([*range(k * 100, (k + 1) * 100)])
        for j in test_index:
            test_indeces.append([*range(j * 100, (j + 1) * 100)])
        train = [item for sublist in train_indeces for item in sublist]
        test = [item for sublist in test_indeces for item in sublist]

        # define train and test data within fold
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        rf_clf.fit(X_train, y_train)  # fit logreg
        preds['test'] = test
        preds['y_truth'] = y[test].values
        preds['y_pred'] = rf_clf.predict(X_test)  # predict for test data
        preds['ID'] = data_v1['ID'][test].values

        # aggregate results per person
        results_per_person = round(preds.groupby(['ID'])['y_pred'].sum() / 100)
        truth_per_person = round(preds.groupby(['ID'])['y_truth'].sum() / 100)
        compare = pd.merge(results_per_person, truth_per_person, on='ID')

        # calculate accuracy per person in each fold
        accuracy = 1 - sum(abs(compare['y_pred'] - compare['y_truth'])) / len(compare)
        fold_acc.append(accuracy)  # fold_acc: list of test accuracy for each fold (5 values)
    rf_results.append(fold_acc)  # svm_results: list of lists of 5 accuracies per fold

rf_results_v1["result"] = rf_results
rf_results_v1['score'] = rf_results_v1.apply(lambda row: sum(row.result) / 5, axis=1)

rf_results_v1.to_csv('/Users/Flora/Downloads/rf_results_v1.tsv', sep='\t', index=False)

# In[ ]:


rf_results_v2 = pd.DataFrame()
params_list = []
[params_list.append(params) for params in grid]
rf_results_v2["params"] = params_list
rf_results = []

# same split for each round for comparability
cv = StratifiedKFold(5, shuffle=True, random_state=0)

i = 0
for params in grid:
    i = i + 1
    if i % 50 == 0:
        print(i)
        print(time.strftime("%m/%d/%Y %H:%M:%S"))
    min_max, min_freq, B, min_n = params['min_max'], params['min_freq'], params['B'], params['min_n']
    vectorizer = TfidfVectorizer(ngram_range=min_max, min_df=min_freq, sublinear_tf=True)
    X = vectorizer.fit_transform(data_v1['Tweets'])
    y = data_v2['spreader']
    rf_clf = RandomForestClassifier(n_estimators=B, min_samples_leaf=min_n, criterion='gini', random_state=0,
                                    oob_score=True)
    fold_acc = []  # list to store tweetwise predictions in each fold
    for train_index, test_index in cv.split(X_all, y_all):
        preds = pd.DataFrame()  # df to store results

        # sampling is done on aggregate data, indices needed for individual tweets
        train_indeces = []
        test_indeces = []
        for k in train_index:
            train_indeces.append([*range(k * 100, (k + 1) * 100)])
        for j in test_index:
            test_indeces.append([*range(j * 100, (j + 1) * 100)])
        train = [item for sublist in train_indeces for item in sublist]
        test = [item for sublist in test_indeces for item in sublist]

        # define train and test data within fold
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        rf_clf.fit(X_train, y_train)  # fit logreg
        preds['test'] = test
        preds['y_truth'] = y[test].values
        preds['y_pred'] = rf_clf.predict(X_test)  # predict for test data
        preds['ID'] = data_v1['ID'][test].values

        # aggregate results per person
        results_per_person = round(preds.groupby(['ID'])['y_pred'].sum() / 100)
        truth_per_person = round(preds.groupby(['ID'])['y_truth'].sum() / 100)
        compare = pd.merge(results_per_person, truth_per_person, on='ID')

        # calculate accuracy per person in each fold
        accuracy = 1 - sum(abs(compare['y_pred'] - compare['y_truth'])) / len(compare)
        fold_acc.append(accuracy)  # fold_acc: list of test accuracy for each fold (5 values)
    rf_results.append(fold_acc)  # svm_results: list of lists of 5 accuracies per fold

rf_results_v2["result"] = rf_results
rf_results_v2['score'] = rf_results_v2.apply(lambda row: sum(row.result) / 5, axis=1)
rf_results_v2.to_csv('/Users/Flora/Downloads/rf_results_v2.tsv', sep='\t', index=False)

