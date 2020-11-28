import re
import emojis
from scripts.mongoConnection import getCollection, insertCollection

class NationalIdentityTagging:
    """NationalIdentityTagging class represents a class for finding national identity in the text

    :func:
        The sequence of execution of functions in this class for extracting national identity from the text
        is as follows:
        1 :func: extract_emojis
        2 :func: remove_emojis_from_onlytext
        3 :func: extract_country_emojis
        4 :func: find_national_identity
    """
    def __init__(self):
        self.flags = []

    def find_national_identity(self, text):
        """ returns list of countries present in the text given to this function

        The text passed to this function is converted to lowercase as the list of countries in this function
        defined as country_list is in lowercase. Using list comprehension, it returns the list of
        countries present in country_list that are present in the text given to this function.

        :param text: text from the 'text' column of a dataframe
        :type text: String

        :return: country- list of countries found in the text
        :rtype: []

        """
        country_list = ["albania", "belgium", "croatia", "czech republic", "england", "france", "germany", "hungary",
                        "iceland", "italy", "northen ireland", "poland", "portugal", "republic of ireland", "romania",
                        "russia", "slovakia", "spain", "sweden", "switzerland ", "turkey", "ukraine", "wales",
                        "austria"]

        comment = text.lower()
        country = [country for country in country_list if country in comment]
        return country


    def extract_country_emojis(self, emojis):
        """returns the emojis which are flags of a country
        
        List of flag emojis is collected from the 00_Assets database. The list of emojis passed to 
        this function is compared against the flag emojis from the database. All matching emojis are returned

        :param emojis: emojis from the 'emojis' column of a dataframe
        :type emojis: list

        :return: country_flags- list of countries found in the text
        :rtype: []
        """

        country_flags = [flag for flag in emojis if flag in self.flags]
        return country_flags


    def remove_emojis_from_onlytext(self, onlyText):
        """removes all the emojis from the given text

        The text given to this function is passed to the 'get' function of emojis package. Which returns the set of 
        found emojis in the text. If the set is empty, the text received is returned as it is. If not, found emojis 
        are removed from the text using regex.
        Finally text without emojis is returned.

        :param onlyText: text from the 'onlyText' column of a dataframe
        :type onlyText: String

        :return: onlyText- Text without emojis
        :rtype: String
        """
        emojiList = emojis.get(onlyText)
        if (len(emojiList) == 0):
            return onlyText
        else:
            emojiList = ' | '.join(emojiList)
            onlyText = re.sub(rf'\b{emojiList}\b', '', onlyText)
            return onlyText


    def extract_emojis(self, onlyText):
        """extract all the emojis from the given text

        The text given to this function is passed to the 'get' function of emojis package. Which returns the set of 
        found emojis in the text. If the set is empty, the text received is returned as it is. If not, found emojis 
        are removed from the text using regex.
        Finally text without emojis is returned.

        :param onlyText: text from the 'onlyText' column of a dataframe
        :type onlyText: String

        :return: emojis- list of emojis
        :rtype: List
        """
        emojiList = emojis.get(onlyText)
        if (len(emojiList) == 0):
            return []
        else:
            return list(emojiList)



    def get_flags(self):
        """Fetches flag_emojis from the database '00_Assets' and returns list of flags

        :return: list of flags
        :rtype: List
        """
        flags = getCollection('00_Assets', 'flag_emojis')
        flags = flags['flag_emoji'].str.strip()
        self.flags = list(flags)


    def postData(self, post, collection):
        """Extracts emojis and country from a dataframe and returns transformed dataframe

        To extract emojis and countries from a dataframe given to this function, it applies extract_emojis,
        remove_emojis_from_onlytext, extract_country_emojis and find_national_identity on the dataframe.
        Finally the resultant dataframe is inserted into the database.
        
        :param1 post: dataframe of post/comment/subcomment
        :param2 collection: Collection name in which the result is to be inserted
        :type1 post: pandas dataframe object
        :type2 collection: String
        
        """
        ##TODO :param instead of :param1, same with :type
        # Initialisations

        post = post.fillna('0')

        post['emojis'] = post['onlyText'].apply(self.extract_emojis)
        post['onlyText'] = post['onlyText'].apply(self.remove_emojis_from_onlytext)
        post['countryem'] = post['emojis'].apply(self.extract_country_emojis)
        post['country'] = post['text'].apply(self.find_national_identity)

        insertCollection('05_NationalIdentity', collection, post)

if __name__ == "__main__":
    
    nationalIdentityTaggingObj = NationalIdentityTagging()
    nationalIdentityTaggingObj.get_flags()

    df = getCollection('04_NationalIdentity_Sentiment', 'sentiment_post_Collection')
    nationalIdentityTaggingObj.postData(df, 'ni_post')

    df = getCollection('04_NationalIdentity_Sentiment', 'sentiment_comment_Collection')
    df.rename(columns={'Comment':'text'}, inplace=True)
    nationalIdentityTaggingObj.postData(df, 'ni_comment')
    
    df = getCollection('04_NationalIdentity_Sentiment', 'sentiment_subcomment_Collection')
    df.rename(columns={'Sub_Comment':'text'}, inplace=True)
    nationalIdentityTaggingObj.postData(df, 'ni_subcomment')

