#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
import pandas as pd
from langdetect import detect
from pymongo import MongoClient
from googletrans import Translator
from googletrans import Translator


# In[2]:


# Source Collection
client = MongoClient('localhost', 27017)


# Post Langdetect and Translation

# In[23]:


db = client['05_NationalIdentity'] # 04_NationalIdentity_Sentiment Connection Object
ni_post = db.ni_post
post = pd.DataFrame(list(ni_post.find({})))
df = post.copy()


# In[4]:


count =0
for index, row in df['onlyText'].iteritems():
    try:
        lang = detect(row)
        df.loc[index, 'detectLang']=lang
    except:
        count+=1
print (count)


# In[7]:


english_file = df.loc[df['detectLang']=='en']
not_english_file = df.loc[df['detectLang']!='en']


# In[9]:


translator = Translator()


# In[12]:


count =0
print(time.asctime())
for index, row in not_english_file['onlyText'].iteritems():
    try:
        translated = translator.translate(row)
        not_english_file.loc[index, 'googleTranslation']=(translated.text)
        time.sleep(0.2)
    except Exception as e:
        print(e)
        count+=1
print (count)
print(time.asctime())


# In[ ]:


del not_english_file['onlyText']
cols = list(not_english_file.columns.values)
cols


# In[ ]:


df_rearranged = not_english_file[['_id', 'Id', 'source_Type', 'username', 'likes', 'timestamp', 'owner_id', 'url', 'Count', 'data_Type', 'text', 'hashtags', 'TagList','googleTranslation', 'sentiments', 'emojis', 'country', 'countryem', 'detectLang']]
df_final = df_rearranged.rename(columns={'googleTranslation':'onlyText'})
df_final


# In[63]:


df_final_combined = pd.concat([english_file, df_final])
df_final_combined


# In[141]:


db_database = client ['06_Translated']
db = db_database.ni_post_translated # 04_NationalIdentity_Sentiment Connection Object
df_post_mongo = df_final_combined.to_dict(orient = "records")
result = db.insert_many(df_post_mongo)


# Subcomment Langdetect and Translation

# In[66]:


db = client['05_NationalIdentity'] # 04_NationalIdentity_Sentiment Connection Object
ni_subcomment = db.ni_subcomment
subcomment = pd.DataFrame(list(ni_subcomment.find({})))
df = subcomment.copy()


# In[69]:


count =0
for index, row in df['onlyText'].iteritems():
    try:
        lang = detect(row)
        df.loc[index, 'detectLang']=lang
    except:
        count+=1
print (count)


# In[71]:


english_file = df.loc[df['detectLang']=='en']
not_english_file = df.loc[df['detectLang']!='en']


# In[ ]:


count =0
print(time.asctime())
for index, row in not_english_file['onlyText'].iteritems():
    try:
        translated = translator.translate(row)
        not_english_file.loc[index, 'googleTranslation']=(translated.text)
        time.sleep(0.2)
    except Exception as e:
        print(e)
        count+=1
print (count)
print(time.asctime())


# In[ ]:


del not_english_file['onlyText']
cols = list(not_english_file.columns.values)
cols


# In[74]:


df_rearranged = not_english_file[['_id', 'Id', 'source_Type', 'username', 'likes', 'timestamp', 'owner_id', 'url', 'Count', 'data_Type', 'text', 'hashtags', 'TagList','googleTranslation', 'sentiments', 'emojis', 'country', 'countryem', 'detectLang']]
df_final = df_rearranged.rename(columns={'googleTranslation':'onlyText'})
df_final


# In[ ]:


df_final_combined = pd.concat([english_file, df_final])
df_final_combined


# In[ ]:


db_database = client ['06_Translated']
db = db_database.ni_subcomment_translated # 04_NationalIdentity_Sentiment Connection Object
df_post_mongo = df_final_combined.to_dict(orient = "records")
result = db.insert_many(df_post_mongo)


# Comment Langdetect and Translation

# In[93]:


db = client['05_NationalIdentity'] # 04_NationalIdentity_Sentiment Connection Object
ni_comment = db.ni_comment
comment = pd.DataFrame(list(ni_comment.find({})))
df = comment.copy()


# In[95]:


count =0
for index, row in df['onlyText'].iteritems():
    try:
        lang = detect(row)
        df.loc[index, 'detectLang']=lang
    except:
        count+=1
print (count)


# In[96]:


english_file = df.loc[df['detectLang']=='en']
not_english_file = df.loc[df['detectLang']!='en']


# In[ ]:


count =0
print(time.asctime())
for index, row in not_english_file['onlyText'].iteritems():
    try:
        translated = translator.translate(row)
        not_english_file.loc[index, 'googleTranslation']=(translated.text)
        time.sleep(0.2)
    except Exception as e:
        print(e)
        count+=1
print (count)
print(time.asctime())


# In[103]:


del not_english_file['onlyText']
cols = list(not_english_file.columns.values)
cols


# In[104]:


df_rearranged = not_english_file[['_id', 'Id', 'source_Type', 'username', 'likes', 'timestamp', 'owner_id', 'url', 'Count', 'data_Type', 'text', 'hashtags', 'TagList','googleTranslation', 'sentiments', 'emojis', 'country', 'countryem', 'detectLang']]
df_final = df_rearranged.rename(columns={'googleTranslation':'onlyText'})
df_final


# In[105]:


df_final_combined = pd.concat([english_file, df_final])
df_final_combined


# In[106]:


db_database = client ['06_Translated']
db = db_database.ni_comment_translated # 04_NationalIdentity_Sentiment Connection Object
df_post_mongo = df_final_combined.to_dict(orient = "records")
result = db.insert_many(df_post_mongo)

