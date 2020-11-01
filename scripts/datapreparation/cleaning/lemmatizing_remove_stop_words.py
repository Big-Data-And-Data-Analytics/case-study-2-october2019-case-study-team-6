
import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scripts.mongoConnection import getCollection, insertCollection


class TextCleaning:
    """TextCleaning class represents a class for cleaning a dataframe column onlyText
    where stop words will be removed and text will be lemmatized

    """

    def __init__(self):
        """lemmatizer and stop words for the instance of TextCleaning class is initialized
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def remove_stop_words(self, listOfWords):
        """removes stop words from the list of words, lemmatizes the words and returns the final output

        :param listOfWords: list of words
        :type listOfWords: List

        :return: list of words after lemmatization and removing stop words
        :rtype: List
        """

        filtered_list = [w for w in listOfWords if not w in self.stop_words]
        lemmatized_output = ' '.join([self.lemmatizer.lemmatize(w) for w in filtered_list])
        return lemmatized_output

    def get_clean_df(self, df, collection):
        """cleans onlyText column of the given dataframe and inserts that dataframe in the '07_PreProcessing' database
        

        :param1 df: Dataframe to Clean
        :param2 collection: Collection name where the cleaned dataframe will be inserted

        :type1 df: Pandas Dataframe
        :type2 collection: String
        """

        df = df.dropna(subset=['onlyText'])
        df['onlyText'] = df['onlyText'].str.lower()
        df['onlyText'] = df['onlyText'].replace(r'@', '', regex=True).replace(
            r'http\S+', '', regex=True).replace(
            r'http \S+', '', regex=True).replace(
            r'www\S+', '', regex=True)
        df['onlyText'] = df['onlyText'].str.replace('[^\w\s]','')
        df['onlyText'] = df['onlyText'].str.replace('\s\s+',' ')
        df['onlyText'] = df['onlyText'].str.strip()
        df['onlyText'] = df['onlyText'].apply(word_tokenize)
        df['onlyText'] = df['onlyText'].apply(self.remove_stop_words)
        del df['_id']

        insertCollection('07_PreProcessing', collection, df)

if __name__ == '__main__':

    textCleaning = TextCleaning()

    df_post = getCollection('06_NationalIdentity_Translated', 'ni_post_translated')
    textCleaning.get_clean_df(df_post, 'ni_post_preprocessed')

    df_comment = getCollection('06_NationalIdentity_Translated', 'ni_comment_translated')
    textCleaning.get_clean_df(df_comment, 'ni_comment_preprocessed')

    df_subcomment = getCollection('06_NationalIdentity_Translated', 'ni_subcomment_translated')
    textCleaning.get_clean_df(df_subcomment, 'ni_subcomment_preprocessed')

