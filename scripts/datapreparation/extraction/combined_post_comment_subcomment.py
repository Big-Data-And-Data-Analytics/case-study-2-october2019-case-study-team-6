import pymongo
import pandas as pd
from datapreparation.mongoConnection import connectMongo, getCollection, insertCollection

class extractTagHashtag:
    def decorator_taglist(func):
        def wrapper_function(value):
            #taglist=value['Comment'].str.findall(r'@.*?(?=\s|$)')
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
    collection_post = connectMongo('03_NationalIdentity_Combined', 'common_post_Combined')
    collection_comment = connectMongo('03_NationalIdentity_Combined', 'common_comment_Combined')
    collection_subcomment = connectMongo('03_NationalIdentity_Combined', 'common_subcomment_Combined')
    df_post = pd.DataFrame(list(collection_post.find({})))
    # df_comment = pd.DataFrame(list(collection_comment.find({})))
    # df_subcomment = pd.DataFrame(list(collection_subcomment.find({})))
    # df_comment.rename(columns={'Comment':'text'}, inplace=True)
    # df_subcomment.rename(columns={'Sub_Comment':'text'}, inplace=True)
    result_post = extractTagHashtag.hastag_taglist_onlytext_function(df_post)
    # result_comment = extractTagHashtagObj.hastag_taglist_onlytext_function(df_comment)
    # result_subcomment = extractTagHashtagObj.hastag_taglist_onlytext_function(df_subcomment)

    result_post = result_post.fillna('0')
    insertCollection('03_NationalIdentity_Combined', 'common_post_inserted', result_post)
    #result_comment = result_comment.to_dict(orient = "records")
    #collection_comment.insert_many(result_comment)
    #result_subcomment = result_subcomment.to_dict(orient = "records")
    #collection_subcomment.insert_many(result_subcomment)