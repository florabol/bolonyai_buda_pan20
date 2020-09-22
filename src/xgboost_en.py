import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import pickle
import xgboost as xgb

data_v1=pd.read_csv('/Users/Flora/Downloads/drive-download-20200530T201213Z-001/clean_en_data_v1.tsv', delimiter='\t',
                      encoding='utf-8')
data_v2=pd.read_csv('/Users/Flora/Downloads/drive-download-20200530T201213Z-001/clean_en_data_v2.tsv', delimiter='\t',
                      encoding='utf-8')

pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                ('xgb', xgb.XGBClassifier(random_state=0))])

parameters={"vect__ngram_range": [(1,2)],
            "vect__min_df":[7,8,9],
            "xgb__eta":[.01,.1,.3],
            "xgb__n_estimators":[200,300],
            "xgb__max_depth": [3,4,5,6],
            "xgb__subsample": [.6,.7,.8],
            "xgb__colsample_bytree":[.5,.6,.7]}

grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1)
print(time.strftime("%m/%d/%Y %H:%M:%S"))
xgb_v12=grid_search.fit(data_v1["Tweets"],data_v1['spreader'])
print(time.strftime("%m/%d/%Y %H:%M:%S"))
filename='/Users/Flora/Downloads/xgb_v11_gs.pickle' #saving out gridsearch
pickle.dump(xgb_v12, open(filename, 'wb'))
v13_df=pd.DataFrame()
v13_df['params']=xgb_v12.cv_results_['params']
v13_df['scores']=xgb_v12.cv_results_['mean_test_score']
v13_df.to_csv('/Users/Flora/Downloads/xgb_tfidf_results_v13.tsv', sep='\t', index=False)

grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1)
print(time.strftime("%m/%d/%Y %H:%M:%S"))
xgb_v21=grid_search.fit(data_v2["Tweets"],data_v2['spreader'])
print(time.strftime("%m/%d/%Y %H:%M:%S"))
filename='/Users/Flora/Downloads/xgb_v21_gs.pickle' #saving out gridsearch
pickle.dump(xgb_v21, open(filename, 'wb'))
v23_df=pd.DataFrame()
v23_df['params']=xgb_v21.cv_results_['params']
v23_df['scores']=xgb_v21.cv_results_['mean_test_score']
v23_df.to_csv('/Users/Flora/Downloads/xgb_tfidf_results_v23.tsv', sep='\t', index=False)

# train and save best model
vect_xgb_en_v1 = TfidfVectorizer(min_df=8, ngram_range=(1,2), use_idf=True, smooth_idf=True, sublinear_tf=True)
xgb_X_v1=vect_xgb_en_v1.fit_transform(data_v1['Tweets'])
xgb_en_v1 = xgb.XGBClassifier(random_state=0, colsample_bytree= 0.6, eta= 0.01, max_depth= 6, n_estimators= 300, subsample= 0.8)
xgb_en_v1.fit(xgb_X_v1,data_v1['spreader'])
pickle.dump(vect_xgb_en_v1, open('/Users/Flora/Downloads/vect_xgb_en_v1.pickle', 'wb'))
pickle.dump(xgb_en_v1, open('/Users/Flora/Downloads/xgb_en_v1.sav', 'wb'))
