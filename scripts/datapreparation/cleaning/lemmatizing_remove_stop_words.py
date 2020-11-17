
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

    def get_clean_df(self, df):
        """cleans onlyText column of the given dataframe and inserts that dataframe in the '07_PreProcessing' database
        
        :param1 df: Dataframe to Clean
        :type1 df: Pandas Dataframe
        """

        df.dropna(subset=['onlyText'], inplace=True)
        df['onlyText'] = df['onlyText'].str.lower()
        df['onlyText'].replace(r'@', '', regex=True, inplace=True)
        df['onlyText'].replace(r'http\S+', '', regex=True, inplace=True)
        df['onlyText'].replace(r'http \S+', '', regex=True, inplace=True)
        df['onlyText'].replace(r'www\S+', '', regex=True, inplace=True)
        df['onlyText'].replace(r'[^\w\s]','', regex=True, inplace=True)
        df['onlyText'].replace(r'\s\s+',' ', regex=True, inplace=True)
        df['onlyText'] = df['onlyText'].str.strip()
        df['onlyText'] = df['onlyText'].apply(word_tokenize)
        df['onlyText'] = df['onlyText'].apply(self.remove_stop_words)
        del df['_id']

        return df

if __name__ == '__main__':

    textCleaning = TextCleaning()

    df_post = getCollection('06_NationalIdentity_Translated', 'ni_post_translated')
    df_post = textCleaning.get_clean_df(df_post)
    insertCollection('07_PreProcessing', 'ni_post_preprocessed', df_post)

    df_comment = getCollection('06_NationalIdentity_Translated', 'ni_comment_translated')
    df_comment = textCleaning.get_clean_df(df_comment)
    insertCollection('07_PreProcessing', 'ni_comment_preprocessed', df_comment)

    df_subcomment = getCollection('06_NationalIdentity_Translated', 'ni_subcomment_translated')
    df_subcomment = textCleaning.get_clean_df(df_subcomment)
    insertCollection('07_PreProcessing', 'ni_subcomment_preprocessed', df_subcomment)


