#Random Forest grid search

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
import time

data_v1=pd.read_csv('/content/drive/My Drive/PAN20/data/clean_es_data_v1.tsv',delimiter='\t', encoding='utf-8')
data_v2=pd.read_csv('/content/drive/My Drive/PAN20/data/clean_es_data_v2.tsv',delimiter='\t', encoding='utf-8')

grid=ParameterGrid({"min_max":[(1,1), (1,2), (2,2)],
                    "min_freq": [3,4,5,6,7,8,9,10],
                    "B":[100,300,400],
                    "min_n":[5,6,7,8,9,10]})

rf_results_v1=pd.DataFrame()
params_list=[]
[params_list.append(params) for params in grid]

rf_results_v1["params"]=params_list
rf_results_1=[]
i=0
for params in grid:
    i=i+1
    min_max, min_freq, B, min_n = params['min_max'], params['min_freq'], params['B'], params['min_n']
    vectorizer=TfidfVectorizer(ngram_range=min_max,min_df=min_freq,sublinear_tf=True)
    X_vectorized=vectorizer.fit_transform(data_v1['Tweets'])
    rf_clf=RandomForestClassifier(n_estimators=B, min_samples_leaf=min_n, criterion='gini',random_state=0, oob_score=True)
    rf_clf.fit(X_vectorized, data_v1["spreader"])
    oob=rf_clf.oob_score_
    rf_results_1.append(oob)
    if i%10==0:
        print(i)
        print(time.strftime("%m/%d/%Y %H:%M:%S"))
rf_results_v1['results']=rf_results_1
rf_results_v1.to_csv('/content/drive/My Drive/PAN20/training_results/ES/rf_tfidf_results_v1.tsv', sep='\t', index=False)


rf_results_v2=pd.DataFrame()
rf_results_v2["params"]=params_list
rf_results_v2=[]
i=0
for params in grid:
    i=i+1
    min_max, min_freq, B, min_n = params['min_max'], params['min_freq'], params['B'], params['min_n']
    vectorizer=TfidfVectorizer(ngram_range=min_max,min_df=min_freq,sublinear_tf=True, token_pattern=r'[^\s]+')
    X_vectorized=vectorizer.fit_transform(data_v2['Tweets'])
    rf_clf=RandomForestClassifier(n_estimators=B, min_samples_leaf=min_n, criterion='gini',random_state=0, oob_score=True)
    rf_clf.fit(X_vectorized, data_v2["spreader"])
    oob=rf_clf.oob_score_
    rf_results_v2.append(oob)
    if i%10==0:
        print(i)
        print(time.strftime("%m/%d/%Y %H:%M:%S"))
rf_results_v2.to_csv('/content/drive/My Drive/PAN20/training_results/ES/rf_tfidf_results_v2.tsv', sep='\t', index=False)

rf_results_v2_df=pd.DataFrame()
rf_results_v2_df["params"]=params_list
rf_results_v2_df["results"]=rf_results_v2
rf_results_v2_df.to_csv('/content/drive/My Drive/PAN20/training_results/ES/rf_tfidf_results_v2.tsv', sep='\t', index=False)