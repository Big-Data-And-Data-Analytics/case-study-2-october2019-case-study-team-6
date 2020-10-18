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
post = pd.DataFrame(list(common_post.find({}).limit(10)))
common_comment = db.common_comment_Combined
comment = pd.DataFrame(list(common_comment.find({})))
common_sub_comment = db.common_subcomment_Combined
sub_comment = pd.DataFrame(list(common_sub_comment.find({})))

def sentiment_post(text):

    for i in text:
        # print(f"{i}")
        sentiment_dict = {}
        for eachToken in word_tokenize(i):
            sentiment_dict['onlyText'] = i
            sentiment_dict['word'] = eachToken
            print(pd.DataFrame(sentiment_dict))

sentiment_post(post['text'])

print(post.columns)
print(post.head())