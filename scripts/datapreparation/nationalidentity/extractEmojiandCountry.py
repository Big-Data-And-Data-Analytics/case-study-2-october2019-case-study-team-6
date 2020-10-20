import re

import emojis
import numpy as np
import pandas as pd
from datapreparation.mongoConnection import connectMongo, getCollection


def find_national_identity(text):
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

def extract_country_emojis(emojis):
    """returns the emojis which are flags of a country
    
    List of flag emojis is collected from the 00_NationalIdentity_Assets database. The list of emojis passed to this function 
    is compared against the flag emojis from the database. All matching emojis are returned

    :param emojis: emojis from the 'emojis' column of a dataframe
    :type text: list

    :return: country_flags- list of countries found in the text
    :rtype: []
    """

    asset_connection = connectMongo('00_NationalIdentity_Assets', 'flag_emojis')
    flags = getCollection(asset_connection)
    flags = flags['flag_emoji'].str.strip()
    flags_set = list(flags)
    country_flags = [flag for flag in emojis if flag in flags_set]
    return country_flags


def remove_emojis_from_onlytext(onlyText):
    """removes all the emojis from the given text

    The text given to this function is passed to the 'get' function of emojis package. Which returns the set of found emojis
    in the text. If the set is empty, the text received is returned as it is. If not, found emojis are removed from the text using
    regex.
    Finally text without emojis is returned

    :param onlyText: text from the 'onlyText' column of a dataframe
    :type text: String

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


def extract_emojis(onlyText):
    """extract all the emojis from the given text

    The text given to this function is passed to the 'get' function of emojis package. Which returns the set of found emojis
    in the text. If the set is empty, the text received is returned as it is. If not, found emojis are removed from the text using
    regex.
    Finally text without emojis is returned

    :param onlyText: text from the 'onlyText' column of a dataframe
    :type text: String

    :return: emojis- list of emojis
    :rtype: List
    """
    emojiList = emojis.get(onlyText)
    if (len(emojiList) == 0):
        return []
    else:
        return list(emojiList)


def postData(post, ni_col):
    """Extracts emojis and country from a dataframe and inserts cleaned data into the database

    To extract emojis and countries from a dataframe given to this function, it applies extract_emojis,
    remove_emojis_from_onlytext, extract_country_emojis and find_national_identity on the dataframe.
    Finally the resultant dataframe is inserted back into database '05_NationalIdentity'
    
    :param1 post: dataframe of post/comment/subcomment
    :param2 ni_col: collection object in which resultant dataframe will be inserted
    :type1 post: pandas dataframe object
    :type2 ni_col: pymongo connection object
    
    """
    # Initialisations

    global df
    post = post.fillna('0')

    # Call Extract Emojis - extract_emojis()
    post['emojis'] = post['onlyText'].apply(extract_emojis)

    post['onlyText'] = post['onlyText'].apply(remove_emojis_from_onlytext)

    post['countryem'] = post['emojis'].apply(extract_country_emojis)

    post['country'] = post['text'].apply(find_national_identity)

    ni_col.insert_many(post.to_dict('records'))


# Source Collection
if __name__ == "__main__":
    
    sentiment_post = connectMongo('04_NationalIdentity_Sentiment', 'sentiment_post_Collection')
    df = getCollection(sentiment_post)

    # Target Collection
    ni_post = connectMongo('05_NationalIdentity', 'ni_post')
    postData(df, ni_post)

    sentiment_comment = connectMongo('04_NationalIdentity_Sentiment', 'sentiment_comment_Collection')
    df = getCollection(sentiment_comment)
    # Target Collection
    ni_comment = connectMongo('05_NationalIdentity', 'ni_comment')
    df.rename(columns={'Comment':'text'}, inplace=True)
    postData(df, ni_comment)


    sentiment_subcomment = connectMongo('04_NationalIdentity_Sentiment', 'sentiment_subcomment_Collection')
    df = getCollection(sentiment_subcomment)
    # Target Collection
    ni_subcomment = connectMongo('05_NationalIdentity', 'ni_subcomment')
    df.rename(columns={'Sub_Comment':'text'}, inplace=True)
    postData(df, ni_subcomment)
