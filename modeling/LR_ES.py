#Cross-validating logistic regression

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time

data_v1=pd.read_csv('/content/drive/My Drive/PAN20/data/clean_es_data_v1.tsv',delimiter='\t', encoding='utf-8')
data_v2=pd.read_csv('/content/drive/My Drive/PAN20/data/clean_es_data_v2.tsv',delimiter='\t', encoding='utf-8')

data_v1.head()

# as pipeline
# tfidf-v1
pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                ('lr', LogisticRegression(penalty='l2', solver='liblinear',verbose=0, random_state=5))])
parameters={"vect__ngram_range": [(1,1),(1,2),(2,2)],
            "vect__min_df":[3,4,5,6,7,8,9,10],
            "lr__C":[.1,1,10,100,1000]}
grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1)
print(time.strftime("%m/%d/%Y %H:%M:%S"))
lr_tfidf_v1=grid_search.fit(data_v1["Tweets"],data_v1['spreader'])
print(time.strftime("%m/%d/%Y %H:%M:%S"))
filename='/content/drive/My Drive/PAN20/training_results/ES/normalized_v1_gs.pickle' #saving out gridsearch
pickle.dump(lr_tfidf_v1, open(filename, 'wb'))
v1_df=pd.DataFrame()
v1_df['params']=lr_tfidf_v1.cv_results_['params']
v1_df['scores']=lr_tfidf_v1.cv_results_['mean_test_score']
v1_df.to_csv('/content/drive/My Drive/PAN20/training_results/ES/lr_tfidf_results_v1.tsv', sep='\t', index=False)

#tfidf-v2
pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True, token_pattern=r'[^\s]+')),
                ('lr', LogisticRegression(penalty='l2', solver='liblinear',verbose=0, random_state=5))])
parameters={"vect__ngram_range": [(1,1),(1,2),(2,2)],
            "vect__min_df":[3,4,5,6,7,8,9,10],
            "lr__C":[.1,1,10,100,1000]}
grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1)
print(time.strftime("%m/%d/%Y %H:%M:%S"))
lr_tfidf_v1=grid_search.fit(data_v2["Tweets"],data_v2['spreader'])
print(time.strftime("%m/%d/%Y %H:%M:%S"))
filename='/content/drive/My Drive/PAN20/training_results/ES/normalized_v2_gs.pickle'
pickle.dump(lr_tfidf_v1, open(filename, 'wb'))
v2_df=pd.DataFrame()
v2_df['params']=lr_tfidf_v1.cv_results_['params']
v2_df['scores']=lr_tfidf_v1.cv_results_['mean_test_score']
v2_df.to_csv('/content/drive/My Drive/PAN20/training_results/ES/lr_tfidf_results_v2.tsv', sep='\t', index=False)