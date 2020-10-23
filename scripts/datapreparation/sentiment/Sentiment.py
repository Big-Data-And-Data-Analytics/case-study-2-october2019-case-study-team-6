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
comment = pd.DataFrame(list(common_comment.find({})))
common_sub_comment = db.common_subcomment_Combined
sub_comment = pd.DataFrame(list(common_sub_comment.find({})))

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
counts = jn.groupby(by=['onlyText', 'sentiment'], as_index=False).count() # Columns and the aggregation if agg() used
counts['onlyText'] = counts['onlyText'].str.strip()
counts = counts.sort_values('word').drop_duplicates(['onlyText'],keep='last')
# counts = counts.drop(['index'], axis = 1)
text = text.str.strip()
data_sentiment_post = counts.merge(text, on = 'onlyText', how='right', suffixes=['_post',''])
# sentiment_frame
# sentiment_frame['word'].to_csv('C:/Users/shubh/Documents/sentiment_frame.csv')
# sentiment_Dictionary['word'].to_csv('C:/Users/shubh/Documents/sentiment_dict.csv')
#
#
# sent_dict = pd.read_csv('C:/Users/shubh/Documents/sentiment_dict.csv')
# sent_frame = pd.read_csv('C:/Users/shubh/Documents/sentiment_frame.csv')
#
# sent_frame.word.astype(str)
# sent_dict.word.astype(str)
#
# r = sent_frame.merge(sent_dict, on = 'word', suffixes=['','_1'], how='inner')










