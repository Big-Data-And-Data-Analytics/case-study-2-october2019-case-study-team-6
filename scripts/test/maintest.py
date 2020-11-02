from scripts.datapreparation.cleaning.cleaning import Cleaning
from scripts.mongoConnection import getCollection, insertCollection
from scripts.datapreparation.extraction.extraction import ExtractTagHashtag
from scripts.datapreparation.sentiment.sentiment import Sentiment
from scripts.datapreparation.balancing.balancing import BalancingData
import os

"""
01. Cleaning Step
"""
data = getCollection('01_NationalIdentity_Crawled', 'youTube_Video_Comments_Raw')
cleaner = Cleaning(data)
data = cleaner.remove_duplicates()
data = cleaner.remove_white_spaces("textOriginal")
cleaned_data = cleaner.change_empty_tona("textOriginal")
del data
insertCollection('01_NationalIdentity_Crawled', 'cleaned_data', cleaned_data)
del cleaned_data

"""
02. Combined Step: Data Model Creation
"""


"""
03. Extraction Step
"""

df_post = getCollection('03_NationalIdentity_Combined', 'common_post_Combined')
df_comment = getCollection('03_NationalIdentity_Combined', 'common_comment_Combined')
df_subcomment = getCollection('03_NationalIdentity_Combined', 'common_subcomment_Combined')
df_comment.rename(columns={'Comment': 'text'}, inplace=True)
df_subcomment.rename(columns={'Sub_Comment': 'text'}, inplace=True)

result_post = ExtractTagHashtag.extract(df_post)
result_comment = ExtractTagHashtag.extract(df_comment)
result_subcomment = ExtractTagHashtag.extract(df_subcomment)

result_post = result_post.fillna('0')
result_comment = result_comment.fillna('0')
result_subcomment = result_subcomment.fillna('0')

insertCollection('03_NationalIdentity_Combined', 'common_post_inserted', result_post)
insertCollection('03_NationalIdentity_Combined', 'common_comment_inserted', result_comment)
insertCollection('03_NationalIdentity_Combined', 'common_subcomment_inserted', result_subcomment)

del df_post
del df_comment
del df_subcomment

del result_post
del result_comment
del result_subcomment

"""
04. Translation
"""


"""
05. Sentiment
"""

post = getCollection('03_NationalIdentity_Combined', 'common_post_Combined')
comment = getCollection('03_NationalIdentity_Combined', 'common_comment_Combined')
sub_comment = getCollection('03_NationalIdentity_Combined', 'common_subcomment_Combined')

##TODO Check if ran twice does data load?
sentiment = Sentiment()
sentiment.apply_load_sentiment(post, 'sentiment_post_Collection')
sentiment.apply_load_sentiment(comment, 'sentiment_comment_Collection')
sentiment.apply_load_sentiment(sub_comment, 'sentiment_subcomment_Collection')

del post
del comment
del sub_comment


"""
06. Lemmitization
"""




"""
07. Tagging
    a. Identity Motives
    b. National Identity
    
"""
##TODO Added them they are ready


"""
08. Data Balancing 
"""

##TODO Add parameter for loading data to a different Collection and Database
df_source_collection = getCollection('08_PreTrain', 'train_data')

filepath = input("Enter the path of your file with '/': ")

if os.path.isdir(filepath):
    f = open(r"filepath", "w")
else:
    print("Directory does not exist.")

balancing_input = BalancingData(filepath, df_source_collection)
balancing_input.split_train_test()
balancing_input.threading_function()

"""
 
"""


