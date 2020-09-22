import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import pickle
import xgboost as xgb

data_v1=pd.read_csv('/Users/Flora/Downloads/drive-download-20200530T201213Z-001/clean_es_data_v1.tsv', delimiter='\t',
                      encoding='utf-8')
data_v2=pd.read_csv('/Users/Flora/Downloads/drive-download-20200530T201213Z-001/clean_es_data_v2.tsv', delimiter='\t',
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
filename='/Users/Flora/Downloads/xgb_v12_gs_es.pickle' #saving out gridsearch
pickle.dump(xgb_v12, open(filename, 'wb'))
v12_df=pd.DataFrame()
v12_df['params']=xgb_v12.cv_results_['params']
v12_df['scores']=xgb_v12.cv_results_['mean_test_score']
v12_df.to_csv('/Users/Flora/Downloads/xgb_tfidf_results_v12_es.tsv', sep='\t', index=False)

grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1)
print(time.strftime("%m/%d/%Y %H:%M:%S"))
xgb_v22=grid_search.fit(data_v2["Tweets"],data_v2['spreader'])
print(time.strftime("%m/%d/%Y %H:%M:%S"))
filename='/Users/Flora/Downloads/xgb_v22_gs.pickle_es' #saving out gridsearch
pickle.dump(xgb_v22, open(filename, 'wb'))
v22_df=pd.DataFrame()
v22_df['params']=xgb_v22.cv_results_['params']
v22_df['scores']=xgb_v22.cv_results_['mean_test_score']
v22_df.to_csv('/Users/Flora/Downloads/xgb_tfidf_results_v22_es.tsv', sep='\t', index=False)

