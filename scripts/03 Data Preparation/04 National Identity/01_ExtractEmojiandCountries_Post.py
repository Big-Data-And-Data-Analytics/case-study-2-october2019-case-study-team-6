# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 02:23:52 2020

@author: shubh
"""
import emojis
import pandas as pd
from pymongo import MongoClient
import langid
from langdetect import detect
import time
import numpy as np
start_time = time.time()

# Source Collection
client = MongoClient('localhost', 27017)
db = client['04_NationalIdentity_Sentiment'] # 04_NationalIdentity_Sentiment Connection Object
sentiment_post = db.sentiment_post_Collection
post = pd.DataFrame(list(sentiment_post.find({})))
df = post.copy()

# Target Collection
db_ni = client['05_NationalIdentity']
ni_post = db_ni.ni_post     # 05_NationalIdentity Connection Object


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
    print(f'Before: {str2}')
    print(f'Emojis : {df["emojis"]}')
    emojis = df['emojis'].replace("{", "").replace("}", "").replace("'", "").replace(" ", "")
    list_emo = list(set(emojis.split(",")))
    #emojis_in_df = list(set(list_emo))

    print(list_emo)
    for x in list_emo:
        print(x)
        str1 = str2.replace(x, '')
        str2 = str1
    print(f'After: {str2}')
    return str2




def extract_emojis(onlyText):
    #print("Running Emojis")
    emojiList = emojis.get(onlyText)
    if (len(emojiList) == 0):
        return ' '
    else:
        return str(emojiList)


def lang_detect(df):
    try:
        lang = detect(str(df['onlyText']))
    except Exception:
        print(Exception)
    else:
        return lang



def lang_id(df):
    try:
        lang = langid.classify(str(df['onlyText']))[0]
    except Exception:
        print(Exception)
    else:
        return lang




def postData(post):
    print("Inside PostData")

    # Initialisations

    global df
    post['emojis'] = ' '
    post['country'] = ' '
    post['countryem'] = ' '
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
    flags = pd.read_csv("01 Setup/02 Input_Files/flags_smiley.csv", sep=":")
    flags = flags['emoji'].str.strip()
    flags_set = set(flags)

    extract_country_emojis_function = np.vectorize(extract_country_emojis)
    temp_df['countryem'] = extract_country_emojis_function(flags_set,post['emojis'])
    post['countryem'] = pd.Series(temp_df['countryem'], index=df.index)
    df = post.copy()

    # Call Find National Identity - find_national_identity()
    df['country'] = df.apply(find_national_identity, axis = 1)
    ni_post.insert_many(df.to_dict('records'))

    # Call Language Detection - langid and detectlang
    #df['langdetect'] = df.apply(lang_detect,axis = 1)
    #df['langid'] = df.apply(lang_id, axis=1)


postData(df)

print("--- %s seconds ---" % (time.time() - start_time))