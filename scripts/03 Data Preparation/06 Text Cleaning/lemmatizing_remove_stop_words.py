from pymongo import MongoClient
import pandas as pd
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Source Collection
client = MongoClient('localhost', 27017)
db = client['06_NationalIdentity_Translated']  

# Target Collection
db_database = client ['07_PreProcessing'] #Target DB

post = db.ni_post_translated
post = pd.DataFrame(list(post.find({})))
comment = db.ni_comment_translated
comment = pd.DataFrame(list(comment.find({})))
subcomment = db.ni_subcomment_translated
subcomment = pd.DataFrame(list(subcomment.find({})))


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def removeStopWords(listOfWords):
    filtered_list = [w for w in listOfWords if not w in stop_words]
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in filtered_list])
    return lemmatized_output

def lemmatizeRemoveStopWords(df):
    df['onlyText'] = df['onlyText'].str.lower()
    df['onlyText']=df['onlyText'].replace(r'@', '', regex=True)
    df['onlyText']=df['onlyText'].replace(r'http\S+', '', regex=True).replace(r'http \S+', '', regex=True).replace(r'www\S+', '', regex=True)
    df['onlyText']=df['onlyText'].str.replace('[^\w\s]','')
    df['onlyText']=df['onlyText'].str.replace('\s\s+',' ')
    df['onlyText'] = df['onlyText'].str.strip()
    df = df.dropna(subset=['onlyText'])
    df['onlyText']=df['onlyText'].apply(word_tokenize)
    df['onlyText']=df['onlyText'].apply(removeStopWords)
    del df['_id']
    return df

df = post.copy()
df_post = lemmatizeRemoveStopWords(df)
print("Post data lemmatized")

db_post = db_database.ni_post_preprocessed
df_post = df_post.to_dict(orient = "records")
db_post.insert_many(df_post)
print("post inserted into mongo")
del(df)
del(post)
del(df_post)

df = comment.copy()
df_comment = lemmatizeRemoveStopWords(df)
print("comment data lemmatized")

db_comment = db_database.ni_comment_preprocessed
df_comment = df_comment.to_dict(orient = "records")
db_comment.insert_many(df_comment)
print("comment inserted into mongo")
del(df)
del(comment)
del(df_comment)

df = subcomment.copy()
df_subcomment = lemmatizeRemoveStopWords(df)
print("subcomment data lemmatized")

db_subcomment = db_database.ni_subcomment_preprocessed
df_subcomment = df_subcomment.to_dict(orient = "records")
db_subcomment.insert_many(df_subcomment)
print("subcomment inserted into mongo")
del(df)
del(subcomment)
del(df_subcomment)
