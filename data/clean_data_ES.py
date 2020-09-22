import re
import pandas as pd
from emoji import UNICODE_EMOJI

es_data=pd.read_csv('/content/drive/My Drive/PAN20/data/es_data.tsv',delimiter='\t', encoding='utf-8')
feed_list=es_data["Tweets"].tolist()
def cleaning_v1(tweet_lista):
    cleaned_feed_v1=[]
    for feed in tweet_lista:
        feed = feed.lower()
        feed = re.sub('[^0-9a-z #@]', "", feed)
        feed = re.sub('[\n]', " ", feed)
        cleaned_feed_v1.append(feed)
    return cleaned_feed_v1

es_data["Tweets"]=cleaning_v1(feed_list)
es_data.to_csv('/content/drive/My Drive/PAN20/data/clean_es_data_v1.tsv', sep='\t', index=False)

def is_emoji(s):
    return s in UNICODE_EMOJI
def emoji_space(text):
    return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()

def cleaning_v2(tweet_lista):
    cleaned_feed_v2=[]
    for feed in tweet_lista:
        feed = feed.lower()
        feed = emoji_space(feed)
        feed = re.sub('[,.\'\"\‘\’\”\“]', '', feed)
        feed = re.sub(r'([a-z\'-\’]+)', r'\1 ', feed)
        feed = re.sub(r'(?<![?!:;/])([:\'\";.,?()/!])(?= )','',feed)
        feed = re.sub('[\n]', ' ', feed)
        feed = ' '.join(feed.split())
        cleaned_feed_v2.append(feed)
    return cleaned_feed_v2

es_data["Tweets"]=cleaning_v2(feed_list)
es_data.to_csv('/content/drive/My Drive/PAN20/data/clean_es_data_v2.tsv', sep='\t', index=False)