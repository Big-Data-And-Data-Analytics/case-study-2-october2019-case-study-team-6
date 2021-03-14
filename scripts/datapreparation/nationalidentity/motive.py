import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from datapreparation.mongoConnection import getCollection, insertCollection


class IdentityMotiveTagging:
    """IdentityMotiveTagging class represents a class for finding the six identity motives in the text
    namely meaning, belonging, efficacy, continuity, distinctiveness and self esteem

    """
    def __init__(self):
        """Initializes lists for all the identity motives where synonyms for those motives will be stored
        """
        self.meaning = []
        self.efficacy = []
        self.belonging = []
        self.continuity = []
        self.distinctiveness = []
        self.selfEsteem = []
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
        lemmatized_output = ' | '.join([self.lemmatizer.lemmatize(w) for w in filtered_list])
        return lemmatized_output


    def tagging(self, df):
        """tagging is used to check if any words from the list of the six identity motives are present in the column
        of the df, to tag the text if the motive is found and finally insert to the '08_PreTrain' database.

        :param df: DataFrame in which the identity motives are to be searched
        :type df: Pandas Dataframe

        :return: Dataframe with identity motive tagged
        :rtype: Pandas DataFrame
        """
        df = df.loc[:, ['Id', 'source_Type', 'data_Type', 'onlyText', 'sentiments', 'country', 'countryem']]
        df.loc[df['onlyText'].str.contains(rf'\b{self.meaning}\b',regex=True, case=False), 'meaning'] = 1
        df.loc[df['onlyText'].str.contains(rf'\b{self.belonging}\b',regex=True, case=False), 'belonging'] = 1
        df.loc[df['onlyText'].str.contains(rf'\b{self.continuity}\b',regex=True, case=False), 'continuity'] = 1
        df.loc[df['onlyText'].str.contains(rf'\b{self.distinctiveness}\b',regex=True, case=False), 'distinctiveness'] = 1
        df.loc[df['onlyText'].str.contains(rf'\b{self.efficacy}\b',regex=True, case=False), 'efficacy'] = 1
        df.loc[df['onlyText'].str.contains(rf'\b{self.selfEsteem}\b',regex=True, case=False), 'selfEsteem'] = 1
        df.fillna(0, inplace=True)
        df = self.unpivot(df)
        return df

    @staticmethod
    def unpivot(df):
        """Unpivot function splits the dataframe in list of dataframes having 50000 rows each, then unpivots each 
        dataframe on identity motives columns.
        Finally, it filteres the records which are having a value either in country or countryem column, removes
        the column flag and returns unpivoted filtered dataframe

        :param df: Dataframe to unpivot
        :type df: Pandas Dataframe

        :return: unpivoted dataframe
        :rtype: Pandas Dataframe
        """
        n = 50000
        df_list = [df[i:i+n] for i in range(0, df.shape[0], n)]
        df = pd.concat([i.melt(id_vars=(['Id', 'source_Type', 'data_Type', 'onlyText', 'sentiments',
                                                   'country', 'countryem']),
                                         value_vars=(['meaning', 'belonging', 'continuity', 'distinctiveness',
                                                      'efficacy', 'selfEsteem']),
                                         var_name='identityMotive',
                                         value_name='flag') for i in df_list], ignore_index=True)

        df = df[df['flag']!=0]
        Country_Filter = ((df['country'] != "") | (df['countryem'] != ""))
        df = df[Country_Filter]
        del df['flag']
        return df


    def get_clean_motives(self, motive):
        """get_clean_motives is used for tokenizing the list of motives, removing stop words from the tokenized
        list and return the string joined with the " ".

        :param motive: list of synonyms of all the identity motives
        :type motive: List

        :return: motive
        :rtype: String
        """

        motive = ' '.join(motive)
        motive = word_tokenize(motive)
        motive = self.remove_stop_words(motive)
        return motive
    

    def get_synonyms(self):
        """get_synonyms extracts synonyms of six identity motives from the database and assigns them to the
        lists initialized by IdentityMotiveTagging class
        """
        synonyms = getCollection('00_Assets', 'IdentityMotives')
        self.meaning = self.get_clean_motives(synonyms.Synonyms[0])
        self.efficacy = self.get_clean_motives(synonyms.Synonyms[1])
        self.belonging = self.get_clean_motives(synonyms.Synonyms[2])
        self.continuity = self.get_clean_motives(synonyms.Synonyms[3])
        self.distinctiveness = self.get_clean_motives(synonyms.Synonyms[4])
        self.selfEsteem = self.get_clean_motives(synonyms.Synonyms[5])


if __name__ == '__main__':

    identityMotiveTagging = IdentityMotiveTagging()
    identityMotiveTagging.get_synonyms()

    df = getCollection('07_PreProcessing', 'ni_post_preprocessed')
    df = identityMotiveTagging.tagging(df)
    insertCollection('08_PreTrain', 'train_data', df)

    df = getCollection('07_PreProcessing', 'ni_comment_preprocessed')
    df = identityMotiveTagging.tagging(df)
    insertCollection('08_PreTrain', 'train_data', df)

    df = getCollection('07_PreProcessing', 'ni_subcomment_preprocessed')
    df = identityMotiveTagging.tagging(df)
    insertCollection('08_PreTrain', 'train_data', df)
