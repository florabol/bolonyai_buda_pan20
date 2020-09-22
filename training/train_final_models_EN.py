import pickle
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

data_v1=pd.read_csv('/Users/Flora/Downloads/drive-download-20200530T201213Z-001/clean_en_data_v1.tsv', delimiter='\t',
                      encoding='utf-8')
data_v2=pd.read_csv('/Users/Flora/Downloads/drive-download-20200530T201213Z-001/clean_en_data_v2.tsv', delimiter='\t',
                      encoding='utf-8')

#LR
lr_vectorizer_v1=TfidfVectorizer(ngram_range=(1,2), min_df=6, sublinear_tf=True, use_idf=True, smooth_idf=True)
lr_X_v1=lr_vectorizer_v1.fit_transform(data_v1["Tweets"])
pickle.dump(lr_vectorizer_v1, open('/content/drive/My Drive/PAN20/vectorizers/lr_vectorizer_v1.pickle', 'wb'))

#SVM
svm_vectorizer_v1=TfidfVectorizer(ngram_range=(1,2), min_df=5, sublinear_tf=True, use_idf=True, smooth_idf=True)
svm_X_v1=svm_vectorizer_v1.fit_transform(data_v1["Tweets"])
pickle.dump(svm_vectorizer_v1, open('/Users/Flora/Downloads/svm_vectorizer_v1.pickle', 'wb'))

# RF
rf_vectorizer_v1=TfidfVectorizer(ngram_range=(1,2), min_df=9)
rf_X_v1=rf_vectorizer_v1.fit_transform(data_v2['Tweets'])
pickle.dump(rf_vectorizer_v1, open('/content/drive/My Drive/PAN20/vectorizers/rf_vectorizer_v1.pickle', 'wb'))

# XGB
vect_xgb_en_v1 = TfidfVectorizer(min_df=8, ngram_range=(1,2), use_idf=True, smooth_idf=True, sublinear_tf=True)
xgb_X_v1=vect_xgb_en_v1.fit_transform(data_v1['Tweets'])
pickle.dump(vect_xgb_en_v1, open('/Users/Flora/Downloads/vect_xgb_en_v1.pickle', 'wb'))

# Fitting the models
# Fitting best LR
# v1 {v1, 'lr__C': 1000, 'vect__min_df': 6, 'vect__ngram_range': (1, 2)}
lr_v1=LogisticRegression(C=1000, penalty='l2', solver='liblinear', fit_intercept=False, verbose=0, random_state=5)
lr_v1.fit(lr_X_v1,data_v1['spreader'])
pickle.dump(lr_v1, open('/content/drive/My Drive/PAN20/models/lr_v1.sav', 'wb'))

# Fitting best SVM
# v1 {v1, 'svm__C': 100 vect__min_df': 5 vect__ngram_range': (1 2)}
svm_v1=svm.SVC(C=100, kernel='linear',  probability=True, verbose=False)
svm_v1.fit(svm_X_v1, data_v1['spreader'])
pickle.dump(svm_v1, open('/Users/Flora/Downloads/svm_v1.sav', 'wb'))

# Fitting best RF
# {v2, 'B': 300 min_freq': 10 min_max': (1 2) min_n': 9}
rf_v1=RandomForestClassifier(n_estimators=300, min_samples_leaf=9, criterion='gini', random_state=0, oob_score=True)
rf_v1.fit(rf_X_v1, data_v2["spreader"])
pickle.dump(rf_v1, open('/content/drive/My Drive/PAN20/models/rf_v2.sav', 'wb'))

# Fitting best XGB
xgb_en_v1 = xgb.XGBClassifier(colsample_bytree= 0.6, eta= 0.01, max_depth= 6, n_estimators= 300, subsample= 0.8)
xgb_en_v1.fit(xgb_X_v1,data_v1['spreader'])
pickle.dump(xgb_en_v1, open('/Users/Flora/Downloads/xgb_en_v1.sav', 'wb'))
