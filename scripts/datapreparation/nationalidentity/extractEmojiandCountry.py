import re
import time

import emojis
import langid
import numpy as np
import pandas as pd
from langdetect import detect
from datapreparation.mongoConnection import connectMongo, getCollection
import csv



def find_national_identity(text):
    country_list = ["albania", "belgium", "croatia", "czech republic", "england", "france", "germany", "hungary",
                    "iceland", "italy", "northen ireland", "poland", "portugal", "republic of ireland", "romania",
                    "russia", "slovakia", "spain", "sweden", "switzerland ", "turkey", "ukraine", "wales",
                    "austria"]

    comment = text.lower()
    country = [country for country in country_list if country in comment]
    return country

def extract_country_emojis(emojis):
    global flags_set
    country_flags = [flag for flag in emojis if flag in flags_set]
    return country_flags


def remove_emojis_from_onlytext(onlyText):
    emojiList = emojis.get(onlyText)
    if (len(emojiList) == 0):
        return onlyText
    else:
        emojiList = ' | '.join(emojiList)
        onlyText = re.sub(rf'\b{emojiList}\b', '', onlyText)
        return onlyText


def extract_emojis(onlyText):
    emojiList = emojis.get(onlyText)
    if (len(emojiList) == 0):
        return []
    else:
        return list(emojiList)


def postData(post, ni_col):
    print("Inside PostData")

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
    start_time = time.time()
    sentiment_post = connectMongo('04_NationalIdentity_Sentiment', 'sentiment_post_Collection')
    df = getCollection(sentiment_post)

    asset_connection = connectMongo('00_NationalIdentity_Assets', 'flag_emojis')
    flags = getCollection(asset_connection)
    flags = flags['flag_emoji'].str.strip()
    flags_set = list(flags)
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
