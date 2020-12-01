from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scripts.mongoConnection import getCollection, insertCollection


class Sentiment:
    """
    Sentiment Class is used to retrieve sentiment [positive, negative & neutral] from the onlyText.

    If new data has to be prepared for sentiment analysis, then initialize the Class with two more parameters
    the class has no return result, it will store the data to the appropriate data_type collections or if passed
    :param col_name: passed otherwise


    :param1 new_data: Pass a dataframe with 'onlyText' column to get sentiment for the same
    :param2 col_name: Pass the collection name to which it should be loaded, depending on the data type
    :type1 new_data: pandas.DataFrame
    :type2 col_name: str

    :Example:

    - Example:
        Sentiment(df, 'sentiment_post_Collection')

    .. note:: df is the Dataframe should be replicating the structure of the combined data step depending on the
    data_type [post, comment or subcomment].

    """
    def __init__(self):
        self.analyser = SentimentIntensityAnalyzer()


    def sentiment_analyzer_scores(self, sentence):
        """
        This method returns sentiment value for each sentence i.e. text passed to it.

        :param sentence:
        :type sentence: str
        :return: str
        :return: sentiment
        :rtype: str

        """
        score = self.analyser.polarity_scores(sentence)
        sentiment = score['compound']
        if sentiment >= 0.05:
            return 'positive'
        elif -0.05 < sentiment < 0.05:
            return 'neutral'
        else:
            return 'negative'

    def apply_load_sentiment(self, data, col_name):
        """
        This method retrieves sentiment from sentiment_analyser_scores and loads into appropriate database collection

        :param data: Dataframe for which onlyText sentiment is to be retrieved
        :param col_name: Collection name to load the result Dataframe into, respective to the data_type
        :type data: pandas.DataFrame
        :type col_name: str
        """

        data['onlyText'] = data['onlyText'].str.strip()
        data['sentiment'] = data['onlyText'].apply(self.sentiment_analyzer_scores)
        insertCollection('04_NationalIdentity_Sentiment', col_name, data)




if __name__ == '__main__':

    post = getCollection('03_NationalIdentity_Combined', 'common_post_Combined')
    comment = getCollection('03_NationalIdentity_Combined', 'common_comment_Combined')
    sub_comment = getCollection('03_NationalIdentity_Combined', 'common_subcomment_Combined')

    sentiment = Sentiment()
    sentiment.apply_load_sentiment(post, 'sentiment_post_Collection3')
    sentiment.apply_load_sentiment(comment, 'sentiment_comment_Collection2')
    sentiment.apply_load_sentiment(sub_comment, 'sentiment_subcomment_Collection2')

