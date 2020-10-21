import pymongo
import pandas as pd
from datapreparation.mongoConnection import connectMongo, getCollection, insertCollection

class extractTagHashtag:
    def decorator_taglist(func):
        def wrapper_function(value):
            hashtag_removed = func(value)
            hashtag_removed['tagList'] = hashtag_removed['text'].str.findall(r'@.*?(?=\s|$)')
            hashtag_removed['tagList'] = pd.Series(hashtag_removed['tagList'], dtype="string")
            hashtag_removed['tagList'] =  hashtag_removed['tagList'].str.strip("[]")
            hashtag_removed['onlyText'] = hashtag_removed['onlyText'].str.replace(r'@.*?(?=\s|$)',"")
            return hashtag_removed
        return wrapper_function

    def decorator_hashtag(func): 
        def wrapper_function(value): 
            value['hashtag'] = value['text'].str.findall(r'#.*?(?=\s|$)')
            value['hashtag'] = pd.Series(value['hashtag'], dtype="string")
            value['hashtag'] =  value['hashtag'].str.strip("[]")
            value['onlyText'] = value['text'].str.replace(r'#.*?(?=\s|$)',"")
            return value
        return wrapper_function

    @decorator_taglist
    @decorator_hashtag
    def hastag_taglist_onlytext_function(post):
        return post



if __name__ == "__main__":

    df_post = connectMongo('03_NationalIdentity_Combined', 'common_post_Combined')
    df_comment = connectMongo('03_NationalIdentity_Combined', 'common_comment_Combined')
    df_subcomment = connectMongo('03_NationalIdentity_Combined', 'common_subcomment_Combined')
    df_comment.rename(columns={'Comment':'text'}, inplace=True)
    df_subcomment.rename(columns={'Sub_Comment':'text'}, inplace=True)

    result_post = extractTagHashtag.hastag_taglist_onlytext_function(df_post)
    result_comment = extractTagHashtag.hastag_taglist_onlytext_function(df_comment)
    result_subcomment = extractTagHashtag.hastag_taglist_onlytext_function(df_subcomment)

    result_post = result_post.fillna('0')
    result_comment = result_comment.fillna('0')
    result_subcomment = result_subcomment.fillna('0')
    insertCollection('03_NationalIdentity_Combined', 'common_post_inserted', result_post)
    insertCollection('03_NationalIdentity_Combined', 'common_comment_inserted', result_comment)
    insertCollection('03_NationalIdentity_Combined', 'common_subcomment_inserted', result_subcomment)
