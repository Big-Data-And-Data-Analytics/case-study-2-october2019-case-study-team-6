import pymongo
import pandas as pd
from pymongo import MongoClient
import time


client = MongoClient('localhost', 27017)
db = client['03_NationalIdentity_Combined'] 
collection = db.common_comment_Combined
df_post = pd.DataFrame(list(collection.find({})))

##TODO Use "03_Combined" data

def decorator_taglist(func):
    def wrapper_function(value):
        #taglist=value['Comment'].str.findall(r'@.*?(?=\s|$)')
        hashtag_removed = func(value)
        hashtag_removed['tagList'] = hashtag_removed['Comment'].str.findall(r'@.*?(?=\s|$)')
        hashtag_removed['tagList'] = pd.Series(hashtag_removed['tagList'], dtype="string")
        hashtag_removed['tagList'] =  hashtag_removed['tagList'].str.strip("[]")
        hashtag_removed['onlyText'] = hashtag_removed['onlyText'].str.replace(r'@.*?(?=\s|$)',"")
        return hashtag_removed
    return wrapper_function

def decorator_hashtag(func): 
    def wrapper_function(value): 
        value['hashtag'] = value['Comment'].str.findall(r'#.*?(?=\s|$)')
        value['hashtag'] = pd.Series(value['hashtag'], dtype="string")
        value['hashtag'] =  value['hashtag'].str.strip("[]")
        value['onlyText'] = value['Comment'].str.replace(r'#.*?(?=\s|$)',"")
        return value
    return wrapper_function

@decorator_taglist
@decorator_hashtag
def hastag_taglist_onlytext_function(post):
    return post


result1=hastag_taglist_onlytext_function(df_post)
result1.to_csv('test4.csv')

