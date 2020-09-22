import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import xgboost as xgb

#EN model
en_data_v2=pd.read_csv('/Users/Flora/Downloads/drive-download-20200530T201213Z-001/clean_en_data_v2.tsv', delimiter='\t',
                      encoding='utf-8')
vect_xgb_en=TfidfVectorizer(min_df=8, ngram_range=(1,2), use_idf=True, smooth_idf=True, sublinear_tf=True)
xgb_en=xgb.XGBClassifier(random_state=0, max_depth=3, eta=0.1, n_estimators=200)
xgb_X_en=vect_xgb_en.fit_transform(en_data_v2["Tweets"])
pickle.dump(vect_xgb_en, open('/Users/Flora/Downloads/vect_xgb_en.pickle', 'wb'))
xgb_en.fit(xgb_X_en,en_data_v2['spreader'])
pickle.dump(xgb_en, open('/Users/Flora/Downloads/xgb_en.sav', 'wb'))

#ES model
es_data_v2=pd.read_csv('/Users/Flora/Downloads/drive-download-20200530T201213Z-001/clean_es_data_v2.tsv', delimiter='\t',
                      encoding='utf-8')
vect_xgb_es=TfidfVectorizer(min_df=8, ngram_range=(1,2), use_idf=True, smooth_idf=True, sublinear_tf=True)
xgb_es=xgb.XGBClassifier(random_state=0, max_depth=5, eta=0.1, n_estimators=400)
xgb_X_es=vect_xgb_es.fit_transform(es_data_v2["Tweets"])
pickle.dump(vect_xgb_es, open('/Users/Flora/Downloads/vect_xgb_es.pickle', 'wb'))
xgb_es.fit(xgb_X_es,es_data_v2['spreader'])
pickle.dump(xgb_es, open('/Users/Flora/Downloads/xgb_es.sav', 'wb'))
