from pandas._libs.reshape import explode
from pymongo import MongoClient
import pandas as pd
import tokenize
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt') ##TODO Add to Package __init__() script


class Sentiment:
    """Sentiment is a class to perform Sentiment Analysis of the text.

      In order to get the National Identity, we need to know if the post/comment/sub-comment
      has a positive or a negative sentiment. Lexicons from R, get_sentiments('bing', 'nrc', 'loughran')
      are being used.
      ##TODO: Dictionary should be downloaded when the package is installed

      - **parameters**, **types**, **return** and **return types**::

            :param arg1: description
            :param arg2: description
            :type arg1: type description
            :type arg1: type description
            :return: return description
            :rtype: the return type description

      """

    def __init__(self, data, ):
        self.data = data
        pass

    def sentiment_post(self):

        pass



client = MongoClient('localhost', 27017)
db = client['03_NationalIdentity_Combined']
common_post = db.common_post_Combined
post = pd.DataFrame(list(common_post.find({})))
# post = pd.DataFrame(list(common_post.find({})))
common_comment = db.common_comment_Combined
# comment = pd.DataFrame(list(common_comment.find({})))
common_sub_comment = db.common_subcomment_Combined
# sub_comment = pd.DataFrame(list(common_sub_comment.find({})))

# Sentiment Dictionary
client = MongoClient('localhost', 27017)
db = client['lexicon']
bing_Collection = db.bing_Collection
loughran_Collection = db.loughran_Collection
nrc_Collection = db.nrc_Collection
bing = pd.DataFrame(list(bing_Collection.find({})))
loughran = pd.DataFrame(list(loughran_Collection.find({})))
nrc = pd.DataFrame(list(nrc_Collection.find({})))
sentiment_Dictionary = pd.concat([bing, loughran, nrc])
sentiment_Dictionary = sentiment_Dictionary.drop(['_id'], axis=1)


text = post['onlyText']
print(type(text))

def tokenn(x):
    return word_tokenize(x)

ds = text.apply(tokenn)
ds = ds.rename('word')
df2 = pd.concat([ds, text], axis = 1)
print(df2.head())

df2 = df2.explode('word')
df2['word'] = df2['word'].str.strip()
df2['word'] = df2['word'].astype('string')
sentiment_Dictionary['word'] = sentiment_Dictionary['word'].astype('string')
print(df2.dtypes, sentiment_Dictionary.dtypes)


jn = df2.merge(sentiment_Dictionary, on = 'word', how = 'inner', suffixes = ['', '_1'])
del df2
sentimentFrame = jn.groupby(by=['onlyText', 'sentiment'], as_index=False).count() # Columns and the aggregation if agg() used
# r = sentimentFrame.groupby(by=['onlyText', 'sentiment'])['word'].max()
del jn
sentimentFrame['onlyText'] = sentimentFrame['onlyText'].str.strip()
sentimentFrame = sentimentFrame.sort_values('word').drop_duplicates(['onlyText'], keep='last')
# sentimentFrame = sentimentFrame.sort_values('word').drop_duplicates(['onlyText'], keep='last')
# counts = counts.drop(['index'], axis = 1)
post['onlyText'] = post['onlyText'].str.strip()
data_sentiment_post = post.merge(sentimentFrame, on = 'onlyText', how='left', suffixes=['_post',''])
# Done till here
# data_sentiment_post < - left_join(data_sentiment, sentimentFrame, by="onlyText")
del sentimentFrame
data_sentiment_test_noDupl = post.groupby(['Id', 'source_Type', 'username', 'text', 'likes', 'timestamp',
       'hashtags', 'owner_id', 'url', 'TagList', 'onlyText', 'data_Type'])['_id'].count().reset_index(name='count')

# Filter Data
filter_1 = data_sentiment_test_noDupl['count']==1
data_sentiment_test_noDupl = data_sentiment_test_noDupl[filter_1]

# Memory Issues
test = data_sentiment_test_noDupl.merge(data_sentiment_post, on='onlyText', how='left', suffixes=['_noDup',''])
del data_sentiment_test_noDupl