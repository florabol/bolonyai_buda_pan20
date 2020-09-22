import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

#kivettem a helyet, hogy paraméterként állítható legyen
loc_data = '/Users/Flora/Downloads/drive-download-20200530T201213Z-001/'

data_es_v1=pd.read_csv(loc_data + 'clean_es_data_v1.tsv', delimiter='\t',
                      encoding='utf-8')
data_es_v2=pd.read_csv(loc_data + 'clean_es_data_v2.tsv', delimiter='\t',
                      encoding='utf-8')
data_es_tw_cons=pd.read_csv(loc_data + 'es_data_tweet_consist.tsv', delimiter='\t',
                      		encoding='utf-8')

# Ezek minden modell esetében a legjobb paraméterek
xgb_pl = Pipeline([('vect', TfidfVectorizer(min_df=8, ngram_range=(1,2), use_idf=True, smooth_idf=True, sublinear_tf=True)),
                ('xgb', xgb.XGBClassifier(colsample_bytree= 0.7, eta= 0.3, max_depth= 6, n_estimators= 200, subsample= 0.6))])

lr_pl=Pipeline([('vect', TfidfVectorizer(min_df=9, ngram_range=(2,2), use_idf=True, smooth_idf=True, sublinear_tf=True)),
                ('lr', LogisticRegression(C=100, penalty='l2', solver='liblinear', fit_intercept=False, verbose=0))])

rf_pl=Pipeline([('vect', TfidfVectorizer(min_df=3, ngram_range=(1,2), use_idf=True, smooth_idf=True, sublinear_tf=True)),
                ('rf', RandomForestClassifier(n_estimators=100, min_samples_leaf=8, criterion='gini'))])

svm_pl=Pipeline([('vect', TfidfVectorizer(ngram_range=(2,2), min_df=8, sublinear_tf=True, use_idf=True, smooth_idf=True)),
                ('rf', SVC(C=10, kernel='linear', verbose=False, probability=True))])

xgb_twc = xgb.XGBClassifier(colsample_bynode = 0.8,
                            colsample_bytree = 0.8,
                            gamma = 4,
                            learning_rate = 0.3,
                            max_depth = 3,
                            min_child_weight = 5,
                            n_estimators = 100,
                            reg_alpha = 0.3,
                            subsample = 0.8)

cv = StratifiedKFold(5, shuffle=True)

X=data_es_v1["Tweets"]
y=data_es_v1['spreader']

results = []
for train_index, test_index in cv.split(X, y):
    preds = pd.DataFrame()
    y_train, y_test = y[train_index], y[test_index]
    preds['y_truth'] = y[test_index].values
    #LR
    X_train, X_test = data_es_v1["Tweets"][train_index], data_es_v1["Tweets"][test_index]
    lr_pl.fit(X_train,y_train)
    preds["lr"] = lr_pl.predict_proba(X_test)[:,1]

    #SVM
    X_train, X_test = data_es_v1["Tweets"][train_index], data_es_v1["Tweets"][test_index]
    svm_pl.fit(X_train,y_train)
    preds["svm"] = svm_pl.predict_proba(X_test)[:,1]

    #RF
    X_train, X_test = data_es_v1["Tweets"][train_index], data_es_v1["Tweets"][test_index]
    rf_pl.fit(X_train,y_train)
    preds["rf"] = rf_pl.predict_proba(X_test)[:,1]

    #XGB
    X_train, X_test = data_es_v1["Tweets"][train_index], data_es_v1["Tweets"][test_index]
    xgb_pl.fit(X_train,y_train)
    preds["xgb"] = xgb_pl.predict_proba(X_test)[:,1]

    #XGB on tweets
    X_train, X_test = data_es_tw_cons.iloc[list(train_index), 2: ], data_es_tw_cons.iloc[list(test_index), 2: ]
    xgb_twc.fit(X_train,y_train)
    preds["xgb_tw"] = xgb_twc.predict_proba(X_test)[:,1]


    results.append(preds)

result_es=pd.concat(results)
result_es_v2=pd.concat(results)

# find best aggregation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier

# MEAN
print(classification_report(result_es["y_truth"], result_es.iloc[:,1:].mean(axis=1)>0.5))
print(confusion_matrix(result_es["y_truth"], result_es.iloc[:,1:].mean(axis=1)>0.5))

print(classification_report(result_es_v2["y_truth"], result_es_v2.iloc[:,1:].mean(axis=1)>0.5))
print(confusion_matrix(result_es_v2["y_truth"], result_es_v2.iloc[:,1:].mean(axis=1)>0.5))

# MAJORITY
print(classification_report(result_es["y_truth"], (result_es.iloc[:,1:]>0.5).mean(1)>0.5))
print(confusion_matrix(result_es["y_truth"], (result_es.iloc[:,1:]>0.5).mean(1)>0.5))

print(classification_report(result_es_v2["y_truth"], (result_es_v2.iloc[:,1:]>0.5).mean(1)>0.5))
print(confusion_matrix(result_es_v2["y_truth"], (result_es_v2.iloc[:,1:]>0.5).mean(1)>0.5))

# LOGREG
acc_scorer = make_scorer(accuracy_score)
scoring = {'Accuracy': acc_scorer}
params = {'solver' : ['saga'],
        'penalty' : ['elasticnet'],
        'C' : [0, 0.1, 0.2, 0.4, 0.7, 0.9, 1, 1.2, 1.5, 2],
        'l1_ratio' : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

logreg_clf = LogisticRegression()
logreg = GridSearchCV(logreg_clf,
                  param_grid=params,
                  scoring=scoring,
                  refit = "Accuracy",
                  return_train_score=True,
                  cv=5,
                  verbose=1,
                  n_jobs = -1)
logreg.fit(result_es.iloc[:,1:], result_es["y_truth"])

print(logreg.best_score_, logreg.best_params_, "\n",
      logreg.best_estimator_, "\n",
      logreg.best_estimator_.coef_, logreg.best_estimator_.intercept_)

print(classification_report(result_es["y_truth"],
                            logreg.predict(result_es.iloc[:,1:])))
print(confusion_matrix(result_es["y_truth"],
                            logreg.predict(result_es.iloc[:,1:])))

print(classification_report(result_es_v2["y_truth"],
                            logreg.predict(result_es_v2.iloc[:,1:])))
print(confusion_matrix(result_es_v2["y_truth"],
                            logreg.predict(result_es_v2.iloc[:,1:])))

# LINREG
params = {'alpha' : [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 15]}
linreg_clf = RidgeClassifier()
linreg = GridSearchCV(linreg_clf,
                  param_grid=params,
                  scoring=scoring,
                  refit = "Accuracy",
                  return_train_score=True,
                  cv=5,
                  verbose=1,
                  n_jobs = -1)
linreg.fit(result_es.iloc[:,1:], result_es["y_truth"])
print(linreg.best_score_, linreg.best_params_, "\n",
      linreg.best_estimator_, "\n",
      linreg.best_estimator_.coef_, linreg.best_estimator_.intercept_)

print(classification_report(result_es["y_truth"],
                            linreg.predict(result_es.iloc[:,1:])))

print(classification_report(result_es_v2["y_truth"],
                            linreg.predict(result_es_v2.iloc[:,1:])))

# save out logistic regression for aggregation
pickle.dump(logreg, open('/Users/Flora/Downloads/es_aggregate_lr.sav', 'wb'))

model=pickle.load(open('/Users/Flora/Downloads/es_aggregate_lr-2.sav', 'rb'))
model.best_estimator_.coef_