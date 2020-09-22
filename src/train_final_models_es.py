import pickle
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

data_v1_es=pd.read_csv('/Users/Flora/Downloads/drive-download-20200530T201213Z-001/clean_es_data_v1.tsv', delimiter='\t',
                      encoding='utf-8')
data_v2_es=pd.read_csv('/Users/Flora/Downloads/drive-download-20200530T201213Z-001/clean_es_data_v2.tsv', delimiter='\t',
                      encoding='utf-8')

#LR
lr_vectorizer_v1=TfidfVectorizer(min_df=9, ngram_range=(2,2), use_idf=True, smooth_idf=True, sublinear_tf=True)
lr_X_v1=lr_vectorizer_v1.fit_transform(data_v1_es["Tweets"])
pickle.dump(lr_vectorizer_v1, open('/Users/Flora/Downloads/lr_vectorizer_v1_es.pickle', 'wb'))
lr_v1=LogisticRegression(C=100, penalty='l2', solver='liblinear', fit_intercept=False, verbose=0)
lr_v1.fit(lr_X_v1,data_v1_es['spreader'])
pickle.dump(lr_v1, open('/Users/Flora/Downloads/lr_v1_es.sav', 'wb'))

#SVM
svm_vectorizer_v1=TfidfVectorizer(ngram_range=(2,2), min_df=8, sublinear_tf=True, use_idf=True, smooth_idf=True)
svm_X_v1=svm_vectorizer_v1.fit_transform(data_v1_es["Tweets"])
pickle.dump(svm_vectorizer_v1, open('/Users/Flora/Downloads/svm_vectorizer_v1_es.pickle', 'wb'))
svm_v1=svm.SVC(C=10, kernel='linear', verbose=False, probability=True)
svm_v1.fit(svm_X_v1, data_v1_es['spreader'])
pickle.dump(svm_v1, open('/Users/Flora/Downloads/svm_v1_es.sav', 'wb'))

# RF
rf_vectorizer_v1=TfidfVectorizer(min_df=3, ngram_range=(1,2), use_idf=True, smooth_idf=True, sublinear_tf=True)
rf_X_v1=rf_vectorizer_v1.fit_transform(data_v1_es['Tweets'])
pickle.dump(rf_vectorizer_v1, open('/Users/Flora/Downloads/rf_vectorizer_v1_es.pickle', 'wb'))
rf_v1=RandomForestClassifier(n_estimators=100, min_samples_leaf=8, criterion='gini')
rf_v1.fit(rf_X_v1, data_v1_es["spreader"])
pickle.dump(rf_v1, open('/Users/Flora/Downloads/rf_v1_es.sav', 'wb'))

# XGB
vect_xgb_es_v1 = TfidfVectorizer(min_df=8, ngram_range=(1,2), use_idf=True, smooth_idf=True, sublinear_tf=True)
xgb_X_v1=vect_xgb_es_v1.fit_transform(data_v1_es['Tweets'])
pickle.dump(vect_xgb_es_v1, open('/Users/Flora/Downloads/vect_xgb_es_v1.pickle', 'wb'))
xgb_es_v1 = xgb.XGBClassifier(colsample_bytree= 0.7, eta= 0.3, max_depth= 6, n_estimators= 200, subsample= 0.6)
xgb_es_v1.fit(xgb_X_v1,data_v1_es['spreader'])
pickle.dump(xgb_es_v1, open('/Users/Flora/Downloads/xgb_es_v1.sav', 'wb'))

# Fitting the models
# Fitting best LR
# v1 {v1, 'lr__C': 1000, 'vect__min_df': 6, 'vect__ngram_range': (1, 2)}


# Fitting best SVM
# v1 {v1, 'svm__C': 100 vect__min_df': 5 vect__ngram_range': (1 2)}


# Fitting best RF
# {v2, 'B': 300 min_freq': 10 min_max': (1 2) min_n': 9}


# Fitting best XGB

