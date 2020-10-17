import pandas as pd
from pymongo import MongoClient
import time
import numpy as np
start_time = time.time()


# Source Pre-Train Post Collection
client = MongoClient('localhost', 27017)
db = client['08_PreTrain']
print('Connected to Mongo')

def filter(df):
    df['country'] = pd.Series(df['country'], dtype="str")
    df['countryem'] = pd.Series(df['countryem'], dtype="str")

    df = df.fillna('')
    print(f'Length Before Filter: {len(df)}')

    Country_Filter = ((df['country'] != "") | (df['countryem'] != ""))
    df = df[Country_Filter]
    print(f'Length after filter: {len(df)}')
    return df

# Source Pre-Train post Collection
pretrain_post = db.ni_post_pretrain
post = pd.DataFrame(list(pretrain_post.find()))
print('Post Data Loaded')

# Source Pre-Train Comment Collection
pretrain_comment = db.ni_comment_pretrain
comment = pd.DataFrame(list(pretrain_comment.find({})))
print('Comment Data Loaded')

# Source Pre-Train Sub Comment Collection
pretrain_subcomment = db.ni_subcomment_pretrain
subcomment = pd.DataFrame(list(pretrain_subcomment.find({})))
print('SubComment Data Loaded')

post = post [['Id', 'source_Type', 'data_Type', 'onlyText', 
            'sentiments', 'country', 'countryem', 'identityMotive']]

comment = comment [['Id', 'source_Type', 'data_Type', 'onlyText', 
            'sentiments', 'country', 'countryem', 'identityMotive']]

subcomment = subcomment [['Id', 'source_Type', 'data_Type', 'onlyText', 
            'sentiments', 'country', 'countryem', 'identityMotive']]

final_data = pd.concat([post,comment,subcomment])
final_data = filter(final_data)
# Target Collection
train_data_collection = db.train_data
train_data_collection.insert_many(final_data.to_dict('records'))

# COUNT OF THE CATEGORIES
print(final_data['identityMotive'].value_counts())