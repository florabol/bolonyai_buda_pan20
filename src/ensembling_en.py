import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

data_en_v1=pd.read_csv('/Users/Flora/Downloads/drive-download-20200530T201213Z-001/clean_en_data_v1.tsv', delimiter='\t',
                      encoding='utf-8')
data_en_v2=pd.read_csv('/Users/Flora/Downloads/drive-download-20200530T201213Z-001/clean_en_data_v2.tsv', delimiter='\t',
                      encoding='utf-8')

# Ezek minden modell esetében a legjobb paraméterek
xgb_pl = Pipeline([('vect', TfidfVectorizer(min_df=8, ngram_range=(1,2), use_idf=True, smooth_idf=True, sublinear_tf=True)),
                ('xgb', xgb.XGBClassifier(random_state=0, colsample_bytree= 0.6, eta= 0.01, max_depth= 6, n_estimators= 300, subsample= 0.8))])

lr_pl=Pipeline([('vect', TfidfVectorizer(min_df=9, ngram_range=(1,2), use_idf=True, smooth_idf=True, sublinear_tf=True)),
                ('lr', LogisticRegression(C=100, penalty='l2', solver='liblinear', fit_intercept=False, verbose=0, random_state=5))])

rf_pl=Pipeline([('vect', TfidfVectorizer(min_df=9, ngram_range=(1,2), use_idf=True, smooth_idf=True, sublinear_tf=True)),
                ('rf', RandomForestClassifier(n_estimators=300, min_samples_leaf=3, criterion='gini', random_state=0))])

svm_pl=Pipeline([('vect', TfidfVectorizer(ngram_range=(1,2), min_df=5, sublinear_tf=True, use_idf=True, smooth_idf=True)),
                ('rf', SVC(C=100, kernel='linear',  random_state=5, verbose=False))])

cv = StratifiedKFold(5, shuffle=True, random_state=3)

X=data_en_v1["Tweets"]
y=data_en_v1['spreader']

results = []
for train_index, test_index in cv.split(X, y):
    preds = pd.DataFrame()
    y_train, y_test = y[train_index], y[test_index]
    preds['y_truth'] = y[test_index].values
    #LR
    X_train, X_test = data_en_v1["Tweets"][train_index], data_en_v1["Tweets"][test_index]
    lr_pl.fit(X_train,y_train)
    preds["lr"] = lr_pl.predict_proba(X_test)[:,1]

    #SVM
    X_train, X_test = data_en_v1["Tweets"][train_index], data_en_v1["Tweets"][test_index]
    svm_pl.fit(X_train,y_train)
    preds["svm"] = svm_pl.predict(X_test)

    #RF
    X_train, X_test = data_en_v2["Tweets"][train_index], data_en_v2["Tweets"][test_index]
    rf_pl.fit(X_train,y_train)
    preds["rf"] = rf_pl.predict_proba(X_test)[:,1]

    #XGB
    X_train, X_test = data_en_v2["Tweets"][train_index], data_en_v2["Tweets"][test_index]
    xgb_pl.fit(X_train,y_train)
    preds["xgb"] = xgb_pl.predict_proba(X_test)[:,1]

    results.append(preds)

result_en=pd.concat(results)

import joblib
xgb_tweetcons_es = joblib.load("/Users/Flora/Downloads/tweetconsistence_xgboost_es")