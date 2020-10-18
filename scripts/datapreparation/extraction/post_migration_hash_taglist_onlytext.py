import pymongo
import pandas as pd
from pymongo import MongoClient


client = MongoClient('localhost', 27017)
db = client['migrate_instagram'] 
collection = db.instagram_post
data_post = pd.DataFrame(list(collection.find({})))
df_post = data_post.copy()

def decorator_taglist(func):
    def wrapper_function(value):
        #taglist=value['post_Caption'].str.findall(r'@.*?(?=\s|$)')
        hashtag_removed = func(value)
        hashtag_removed['tagList'] = hashtag_removed['post_Caption'].str.findall(r'@.*?(?=\s|$)')
        hashtag_removed['onlyText'] = hashtag_removed['onlyText'].str.replace(r'@.*?(?=\s|$)',"")
        return hashtag_removed
    return wrapper_function

def decorator_hashtag(func): 
    def wrapper_function(value): 
        value['hashtag'] = value['post_Caption'].str.findall(r'#.*?(?=\s|$)')
        value['onlyText'] = value['post_Caption'].str.replace(r'#.*?(?=\s|$)',"")
        return value
    return wrapper_function

@decorator_taglist
@decorator_hashtag
def hastag_taglist_onlytext_function(post):
    return post


result1=hastag_taglist_onlytext_function(df_post)
print(result1)
result1.to_csv('test4.csv')
