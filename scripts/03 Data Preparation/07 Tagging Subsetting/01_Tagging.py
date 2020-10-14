#!/usr/bin/env python
# coding: utf-8

# In[109]:


from pymongo import MongoClient
import pandas as pd
# Source Collection
client = MongoClient('localhost', 27017)
db_source = client['07_PreProcessing'] #source
db_target = client['08_PreTrain'] #target


# In[86]:


from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# In[87]:


meaning = ['fair','love','thankyou','thank you', 'gift', 'happy', 'respect','think', 'intend', 'meaning', 'have in mind', 'signification', 'significance', 'stand for', 'import', 'mean', 'entail', 'imply', 'signify', 'substance', 'pregnant', 'significant', 'think of']
belonging = ['our','we','together','all together','altogether','jointly','go', 'belong', 'belong to', 'belonging']
continuity = ['history', 'ever', 'always', 'memory', 'memories', 'remembering','persistence','persist', 
                        'remember', 'continuous', 'flow', 'connection', 'line', 'unity', 'unite', 'whole', 'link',
                        'cohesion', 'stamina', 'train', 'constancy', 'survival', 'survive', 'duration', 'prolong', 
                        'long', 'durable', 'durability', 'vitality', 'vital', 'endure', 'endurance', 'chain', 'extend',
                        'extension', 'sequence', 'linking', 'stable', 'stability', 'connect', 'flow', 'succession']
distinctiveness = ['they', 'revenge', 'vengeance', 'opponent', 'faceup','speciality', 'disparateness', 'peculiarity', 'specialness', 'distinctiveness', 'specialty']
efficacy = ['anticipate', 'foresee', 'i knew it', 'as i said', 'as i was saying', 'predicted','efficacy', 'efficaciousness','effective','effectiveness','success', 'productive', 'fruitful', 'potency','power', 'benefit', 'advantage', 'value', 'virtue', 'use', 'usefulness', 'adequacy','ability','vigor','capability',
                       'energy','strength','weight','force','capacity','influence','use','performance']
selfEsteem = ['proud', 'proudly', 'courage', 'assertion', 'dominating','pride', 'superbia', 'pridefulness',
                         'plume', 'congratulate', 'congratulation', 'self respect', 'dignity', 'morale', 'confidence',
                         'assurance', 'narcissism', 'satisfaction', 'conceit', 'vanity', 'morale', 'egotism', 
                         'regard', 'worth']


# In[88]:


def removeStopWords(listOfWords):
    filtered_list = [w for w in listOfWords if not w in stop_words]
    lemmatized_output = ' | '.join([lemmatizer.lemmatize(w) for w in filtered_list])
    return lemmatized_output


# In[89]:


meaning = ' '.join(meaning)
belonging = ' '.join(belonging)
continuity = ' '.join(continuity)
distinctiveness = ' '.join(distinctiveness)
efficacy = ' '.join(efficacy)
selfEsteem = ' '.join(selfEsteem)
meaning = word_tokenize(meaning)
belonging = word_tokenize(belonging)
continuity = word_tokenize(continuity)
distinctiveness = word_tokenize(distinctiveness)
efficacy = word_tokenize(efficacy)
selfEsteem = word_tokenize(selfEsteem)
meaning = removeStopWords(meaning)
belonging = removeStopWords(belonging)
continuity = removeStopWords(continuity)
distinctiveness = removeStopWords(distinctiveness)
efficacy = removeStopWords(efficacy)
selfEsteem = removeStopWords(selfEsteem)


# In[90]:


def tagging(df):
    df['meaning'] = df['onlyText'].str.contains(rf'\b{meaning}\b',regex=True, case=False).astype(int)
    df['belonging'] = df['onlyText'].str.contains(rf'\b{belonging}\b',regex=True, case=False).astype(int)
    df['continuity'] = df['onlyText'].str.contains(rf'\b{continuity}\b',regex=True, case=False).astype(int)
    df['distinctiveness'] = df['onlyText'].str.contains(rf'\b{distinctiveness}\b',regex=True, case=False).astype(int)
    df['efficacy'] = df['onlyText'].str.contains(rf'\b{efficacy}\b',regex=True, case=False).astype(int)
    df['selfEsteem'] = df['onlyText'].str.contains(rf'\b{selfEsteem}\b',regex=True, case=False).astype(int)
    return df


# In[99]:


post = db_source.ni_post_preprocessed
post = pd.DataFrame(list(post.find({})))
df = post.copy()
del(post)


# In[101]:


df = tagging(df)


# In[103]:


df_unpivoted = df.melt(id_vars= (['_id', 'Id', 'source_Type', 'username', 'likes', 'timestamp',
       'owner_id', 'url', 'Count', 'data_Type', 'text', 'hashtags',
       'TagList', 'onlyText', 'sentiments', 'emojis', 'country',
       'countryem', 'detectLang']), value_vars=(['meaning', 'belonging', 'continuity',
       'distinctiveness', 'efficacy', 'selfEsteem']), var_name ='identityMotive',value_name='flag')
del df_unpivoted['_id']


# In[104]:


df_unpivoted = df_unpivoted[df_unpivoted['flag']!=0]
del df_unpivoted['flag']


# In[105]:


db = db_target.ni_post_pretrain
df = df_unpivoted.to_dict(orient = "records")
db.insert_many(df)
del(df)
del(df_unpivoted)


# Comment Tagging

# In[121]:


comment = db_source.ni_comment_preprocessed
comment = pd.DataFrame(list(comment.find({})))
df = comment.copy()
del(comment)


# In[111]:


df = tagging(df)


# In[113]:


col_list = df.columns.values
col_list


# In[117]:


df_unpivoted = df[0:100000].melt(id_vars= (['_id', 'Id', 'source_Type', 'Comment_Id', 'Comment_Owner',
       'Comment', 'Comment_Likes', 'Comment_Time_Posted', 'hashtags',
       'TagList', 'onlyText', 'Count', 'data_Type', 'sentiments',
       'emojis', 'country', 'countryem', 'detectLang']), value_vars=(['meaning', 'belonging', 'continuity',
       'distinctiveness', 'efficacy', 'selfEsteem']), var_name ='identityMotive',value_name='flag')
del df_unpivoted['_id']


# In[120]:
df_unpivoted = df_unpivoted[df_unpivoted['flag']!=0]
del df_unpivoted['flag']

# %%
db = db_target.ni_comment_pretrain
df_comment = df_unpivoted.to_dict(orient = "records")
db.insert_many(df_comment)
del(df_unpivoted)

# %%
df_unpivoted = df[100000:200000].melt(id_vars= (['_id', 'Id', 'source_Type', 'Comment_Id', 'Comment_Owner',
       'Comment', 'Comment_Likes', 'Comment_Time_Posted', 'hashtags',
       'TagList', 'onlyText', 'Count', 'data_Type', 'sentiments',
       'emojis', 'country', 'countryem', 'detectLang']), value_vars=(['meaning', 'belonging', 'continuity',
       'distinctiveness', 'efficacy', 'selfEsteem']), var_name ='identityMotive',value_name='flag')
del df_unpivoted['_id']


# In[120]:
df_unpivoted = df_unpivoted[df_unpivoted['flag']!=0]
del df_unpivoted['flag']

# %%
db = db_target.ni_comment_pretrain
df_comment = df_unpivoted.to_dict(orient = "records")
db.insert_many(df_comment)
del(df_unpivoted)
del(df_comment)

# In[120]:
df_unpivoted = df[200000:300000].melt(id_vars= (['_id', 'Id', 'source_Type', 'Comment_Id', 'Comment_Owner',
       'Comment', 'Comment_Likes', 'Comment_Time_Posted', 'hashtags',
       'TagList', 'onlyText', 'Count', 'data_Type', 'sentiments',
       'emojis', 'country', 'countryem', 'detectLang']), value_vars=(['meaning', 'belonging', 'continuity',
       'distinctiveness', 'efficacy', 'selfEsteem']), var_name ='identityMotive',value_name='flag')
del df_unpivoted['_id']

df_unpivoted = df_unpivoted[df_unpivoted['flag']!=0]
del df_unpivoted['flag']

# %%
db = db_target.ni_comment_pretrain
df_comment = df_unpivoted.to_dict(orient = "records")
db.insert_many(df_comment)
del(df_unpivoted)
del(df_comment)

# %%
df_unpivoted = df[300000:].melt(id_vars= (['_id', 'Id', 'source_Type', 'Comment_Id', 'Comment_Owner',
       'Comment', 'Comment_Likes', 'Comment_Time_Posted', 'hashtags',
       'TagList', 'onlyText', 'Count', 'data_Type', 'sentiments',
       'emojis', 'country', 'countryem', 'detectLang']), value_vars=(['meaning', 'belonging', 'continuity',
       'distinctiveness', 'efficacy', 'selfEsteem']), var_name ='identityMotive',value_name='flag')
del df_unpivoted['_id']

df_unpivoted = df_unpivoted[df_unpivoted['flag']!=0]
del df_unpivoted['flag']

# %%
db = db_target.ni_comment_pretrain
df_comment = df_unpivoted.to_dict(orient = "records")
db.insert_many(df_comment)
del(df_unpivoted)
del(df_comment)

# Subcomment
# %%
subcomment = db_source.ni_subcomment_preprocessed
subcomment = pd.DataFrame(list(subcomment.find({})))
df = subcomment.copy()
del(subcomment)


# In[101]:


df = tagging(df)

# %%

col_list = df.columns.values
col_list

# In[103]:


df_unpivoted = df.melt(id_vars= (['_id', 'Id', 'source_Type', 'Sub_Comment_Id', 'Sub_Comment_Owner',
       'Sub_Comment', 'Sub_Comment_Likes', 'Sub_Comment_Time_Posted',
       'hashtags', 'TagList', 'onlyText', 'Count', 'data_Type',
       'sentiments', 'emojis', 'country', 'countryem', 'detectLang']), value_vars=(['meaning', 'belonging', 'continuity',
       'distinctiveness', 'efficacy', 'selfEsteem']), var_name ='identityMotive',value_name='flag')
del df_unpivoted['_id']


# In[104]:


df_unpivoted = df_unpivoted[df_unpivoted['flag']!=0]
del df_unpivoted['flag']


# In[105]:


db = db_target.ni_subcomment_pretrain
df = df_unpivoted.to_dict(orient = "records")
db.insert_many(df)
del(df)
del(df_unpivoted)
# %%
