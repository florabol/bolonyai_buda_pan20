#Linear SVM cross-validation and grid search

import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time

data_v1=pd.read_csv('/content/drive/My Drive/PAN20/data/clean_es_data_v1.tsv',delimiter='\t', encoding='utf-8')
data_v2=pd.read_csv('/content/drive/My Drive/PAN20/data/clean_es_data_v2.tsv',delimiter='\t', encoding='utf-8')

# tfidf-v1
pipeline = Pipeline([('vect', TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                ('svm', svm.SVC(kernel='linear',  random_state=5, verbose=False))])
parameters={"vect__ngram_range": [(1,1),(1,2),(2,2)],
            "vect__min_df":[3,4,5,6,7,8,9,10],
            "svm__C":[1,10,100,1000]}
grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1)
print(time.strftime("%m/%d/%Y %H:%M:%S"))
svm_tfidf_v1=grid_search.fit(data_v1["Tweets"],data_v1['spreader'])
filename='/content/drive/My Drive/PAN20/training_results/ES/svm_tfidf_1_cs.pickle'
pickle.dump(svm_tfidf_v1, open(filename, 'wb'))
print(time.strftime("%m/%d/%Y %H:%M:%S"))
v1_df=pd.DataFrame()
v1_df['params']=svm_tfidf_v1.cv_results_['params']
v1_df['scores']=svm_tfidf_v1.cv_results_['mean_test_score']
v1_df.to_csv('/content/drive/My Drive/PAN20/training_results/ES/svm_tfidf_results_v1.tsv', sep='\t', index=False)


# tfidf-v2
pipeline = Pipeline([('vect', TfidfVectorizer(sublinear_tf=True, token_pattern=r'[^\s]+')),
                ('svm', svm.SVC(kernel='linear',  random_state=5, verbose=False))])
parameters={"vect__ngram_range": [(1,1),(1,2),(2,2)],
            "vect__min_df":[3,4,5,6,7,8,9,10],
            "svm__C":[1,10,100,1000]}
grid_search=GridSearchCV(pipeline, parameters, cv=StratifiedKFold(5,shuffle=True, random_state=0),n_jobs=-1)
print(time.strftime("%m/%d/%Y %H:%M:%S"))
svm_tfidf_v2=grid_search.fit(data_v2["Tweets"],data_v2['spreader'])
filename='/content/drive/My Drive/PAN20/training_results/ES/svm_tfidf_2_cs.pickle'
pickle.dump(svm_tfidf_v2, open(filename, 'wb'))
print(time.strftime("%m/%d/%Y %H:%M:%S"))
v2_df=pd.DataFrame()
v2_df['params']=svm_tfidf_v2.cv_results_['params']
v2_df['scores']=svm_tfidf_v2.cv_results_['mean_test_score']
v2_df.to_csv('/content/drive/My Drive/PAN20/training_results/ES/svm_tfidf_results_v2.tsv', sep='\t', index=False)