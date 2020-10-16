# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 02:23:52 2020

@author: shubh
"""
import emojis
import pandas as pd
import langid
from langdetect import detect
import time
import numpy as np
from ..mongoConnection import connectMongo, getCollection


def find_national_identity(df):
    print('Running find_national_identity')
    country_list = ["Albania", "Belgium", "Croatia", "Czech Republic", "England", "France", "Germany", "Hungary",
                    "Iceland", "Italy", "Northen Ireland", "Poland", "Portugal", "Republic of Ireland", "Romania",
                    "Russia", "Slovakia", "Spain", "Sweden", "Switzerland ", "Turkey", "Ukraine", "Wales",
                    "Austria"]

    comment = df['text'].upper()
    for country in country_list:
        cntry = country.upper()
        if (cntry in comment):
            return str(cntry)

    print('Completed find_national_identity')

def extract_country_emojis(flags_set,emojis):
    print("Inside extract_country_emojis")

    current_emoji_set = str(emojis)
    # Strip white spaces
    current_emoji_set = current_emoji_set.strip()
    # Convert the String 'set' to a list to access each item
    current_emoji_set = current_emoji_set.replace("{", "").replace("}", "").replace("'", "").replace(" ", "")
    list_emo = current_emoji_set.split(",")
    emojis_in_df = set(list_emo)

    # Get only common emojis
    flag = emojis_in_df.intersection(flags_set)

    if len(flag) >= 1:
        return str(flag)
    else:
        return str("")


def remove_emojis_from_onlytext(df):

    print("Inside remove_emojis_from_onlytext")
    str2 = str(df['onlyText'])
    emojis = df['emojis'].replace("{", "").replace("}", "").replace("'", "").replace(" ", "")
    list_emo = list(set(emojis.split(",")))

    for x in list_emo:
        str1 = str2.replace(x, '')
        str2 = str1
    return str2


def extract_emojis(onlyText):
    emojiList = emojis.get(onlyText)
    if (len(emojiList) == 0):
        return ' '
    else:
        return str(emojiList)


def postData(post, ni_col):
    print("Inside PostData")

    # Initialisations

    global df
    post = post.fillna('0')

    # Dataframes
    temp_df = pd.DataFrame()  # Vectorizing Emojis

    # Call Extract Emojis - extract_emojis()
    extract_emojis_function = np.vectorize(extract_emojis)
    temp_df['emojis'] = extract_emojis_function(post['onlyText'])
    post['emojis'] = pd.Series(temp_df['emojis'], index=df.index)
    df = post.copy()


    # Call Remove Emojis from Text - remove_emojis_from_onlytext()
    post['onlyText'] = df.apply(remove_emojis_from_onlytext,axis = 1)
    df = post.copy()

    # Call Extract Country Emojis - extract_country_emojis()
    flags = pd.read_csv("flags_smiley.csv", sep=":")
    flags = flags['emoji'].str.strip()
    flags_set = set(flags)

    extract_country_emojis_function = np.vectorize(extract_country_emojis)
    temp_df['countryem'] = extract_country_emojis_function(flags_set,post['emojis'])
    post['countryem'] = pd.Series(temp_df['countryem'], index=df.index)
    df = post.copy()

    # Call Find National Identity - find_national_identity()
    df['country'] = df.apply(find_national_identity, axis = 1)
    ni_col.insert_many(df.to_dict('records'))

    # Call Language Detection - langid and detectlang
    #df['langdetect'] = df.apply(lang_detect,axis = 1)
    #df['langid'] = df.apply(lang_id, axis=1)


# Source Collection
if __name__ == "__main__":
    start_time = time.time()
    sentiment_post = connectMongo('04_NationalIdentity_Sentiment', 'sentiment_post_Collection')
    df = getCollection(sentiment_post)
    # Target Collection
    ni_post = connectMongo('05_NationalIdentity', 'ni_post')     # 05_NationalIdentity Connection Object
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

    print("--- %s seconds ---" % (time.time() - start_time))