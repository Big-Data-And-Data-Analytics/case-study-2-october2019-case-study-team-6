from scripts.mongoConnection import getCollection, insertCollection
from scripts.datapreparation.cleaning.cleaning import Cleaning
from scripts.datapreparation.cleaning.textcleaning import TextCleaning
from scripts.datapreparation.extraction.extraction import ExtractTagHashtag
from scripts.datapreparation.sentiment.sentiment import Sentiment
from scripts.datapreparation.balancing.balancing import BalancingData
from scripts.datapreparation.extraction.datamodel import DataModel
from scripts.datapreparation.translation.translation import Translation
from scripts.datapreparation.nationalidentity.motive import IdentityMotiveTagging
from scripts.datapreparation.nationalidentity.tagging import NationalIdentityTagging
from scripts.datapreparation.featureselection.featureselection import FeatureSelection
import os

"""
************************************************ DATA CLEANING *********************************************************
"""

data = getCollection('01_NationalIdentity_Crawled', 'youTube_Video_Comments_Raw')
cleaner = Cleaning(data)
data = cleaner.remove_duplicates()
data = cleaner.remove_white_spaces("textOriginal")
cleaned_data = cleaner.change_empty_tona("textOriginal")
del data
insertCollection('02_NationalIdentity_Cleaned', 'youTube_Video_Comments_Cleaned', cleaned_data)
del cleaned_data


"""
************************************************ DATA MODEL ************************************************************
"""

apply_model = DataModel()

#### YOUTUBE POST ####
# Get the data from MongoDB
youTube_Video_Stats_Cleaned = getCollection("02_NationalIdentity_Cleaned", "youTube_Video_Stats_Cleaned")
youTube_Videos_Cleaned = getCollection("02_NationalIdentity_Cleaned", "youTube_Videos_Cleaned")
youTube_Videos_Data = apply_model.run_youtube_post(data1=youTube_Video_Stats_Cleaned, data2=youTube_Videos_Cleaned)
insertCollection("03_NationalIdentity_Combined", "common_post_Combined", youTube_Videos_Data)

#### YOUTUBE COMMENT ####
youTube_Video_Comments_Cleaned = getCollection("02_NationalIdentity_Cleaned", "youTube_Video_Comments_Cleaned")
youTube_Video_Comments = apply_model.run_youtube_comment(data=youTube_Video_Comments_Cleaned)
insertCollection("03_NationalIdentity_Combined", "common_comment_Combined", youTube_Video_Comments)

#### YOUTUBE SUBCOMMENT ####
youTube_Video_Comments_Cleaned = getCollection("02_NationalIdentity_Cleaned", "youTube_Video_Comments_Cleaned")
youTube_Video_SubComments = apply_model.run_youtube_subcomment(data=youTube_Video_Comments_Cleaned)
insertCollection("03_NationalIdentity_Combined", "common_subcomment_Combined", youTube_Video_SubComments)

#### REDDIT POST ####
reddit_Data_Cleaned = getCollection("02_NationalIdentity_Cleaned", "reddit_Data_Cleaned")
reddit_Data_Post = apply_model.run_reddit_post(data=reddit_Data_Cleaned)
insertCollection("03_NationalIdentity_Combined", "common_post_Combined", reddit_Data_Post)

#### REDDIT COMMENT ####
reddit_Data_Cleaned = getCollection("02_NationalIdentity_Cleaned", "reddit_Data_Cleaned")
reddit_Data_Comment = apply_model.run_reddit_comment(data=reddit_Data_Cleaned)
insertCollection("03_NationalIdentity_Combined", "common_comment_Combined", reddit_Data_Comment)

#### REDDIT SUBCOMMENT ####
reddit_Data_Cleaned = getCollection("02_NationalIdentity_Cleaned", "reddit_Data_Cleaned")
reddit_Data_Subcomment = apply_model.run_reddit_subcomment(data=reddit_Data_Cleaned)
insertCollection("03_NationalIdentity_Combined", "common_subcomment_Combined", reddit_Data_Subcomment)

#### TWITTER POST ####
twitter_Data_Cleaned = getCollection("02_NationalIdentity_Cleaned", "Twitter_Post_Cleaned")
twitter_post = apply_model.run_twitter_post(data=twitter_Data_Cleaned)
insertCollection("03_NationalIdentity_Combined", "common_post_Combined", twitter_post)

#### INSTAGRAM POST ####
instagram_Data_Cleaned = getCollection("02_NationalIdentity_Cleaned", "Instagram_Post_Cleaned")
instagram_post = apply_model.run_instagram_post(data=instagram_Data_Cleaned)
insertCollection("03_NationalIdentity_Combined", "common_post_Combined", instagram_post)

#### INSTAGRAM COMMENT ####
instagram_comment_Cleaned = getCollection("02_NationalIdentity_Cleaned", "Instagram_Comment_Cleaned")
instagram_comment = apply_model.run_instagram_comment(data=instagram_comment_Cleaned)
insertCollection("03_NationalIdentity_Combined", "common_comment_Combined", instagram_comment)

#### INSTAGRAM SUBCOMMENT ####
instagram_comment_Cleaned = getCollection("02_NationalIdentity_Cleaned", "Instagram_Comment_Cleaned")
instagram_subcomment = apply_model.run_instagram_subcomment(data=instagram_comment_Cleaned)
insertCollection("03_NationalIdentity_Combined", "common_subcomment_Combined", instagram_subcomment)

"""
************************************************ EXTRACTION ************************************************************
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

insertCollection('04_NationalIdentity_Extract', 'common_post_extract', result_post)
insertCollection('04_NationalIdentity_Extract', 'common_comment_extract', result_comment)
insertCollection('04_NationalIdentity_Extract', 'common_subcomment_extract', result_subcomment)
# TODO A Target is needed, should it be? 04_Extraction?
del df_post
del df_comment
del df_subcomment

del result_post
del result_comment
del result_subcomment

"""
************************************************ TRANSLATION ***********************************************************
"""
translate = Translation()

#TODO Compare columns from previous step and test, source and target dbnames need to be changed

# S 04_Extraction
# T 05_NationalIdentity_Translated

df = getCollection('04_NationalIdentity_Extract', 'common_post_extract')
df = translate.detect_translate(df)
insertCollection('05_NationalIdentity_Translated', 'ni_post_translated', df)

df = getCollection('05_NationalIdentity_Extract', 'common_comment_extract')
df = translate.detect_translate(df)
insertCollection('05_NationalIdentity_Translated', 'ni_comment_translated', df)

df = getCollection('04_NationalIdentity_Extract', 'common_subcomment_extract')
df = translate.detect_translate(df)
insertCollection('05_NationalIdentity_Translated', 'ni_subcomment_translated', df)


"""
************************************************ SENTIMENT *************************************************************
"""


# S 05_NationalIdentity_Translated
# T 06_NationalIdentity_Sentiment

#TODO Compare columns from previous step and test, source and target dbnames need to be changed
post = getCollection('05_NationalIdentity_Translated', 'ni_post_translated')
comment = getCollection('05_NationalIdentity_Translated', 'ni_comment_translated')
sub_comment = getCollection('05_NationalIdentity_Translated', 'ni_subcomment_translated')

##TODO Check if ran twice does data load?
sentiment = Sentiment()
post_data = sentiment.apply_load_sentiment(post)
insertCollection('06_NationalIdentity_Sentiment', 'sentiment_post_Collection', post_data)

comment_data = sentiment.apply_load_sentiment(comment)
insertCollection('06_NationalIdentity_Sentiment', 'sentiment_comment_Collection', comment_data)

sub_comment_data = sentiment.apply_load_sentiment(sub_comment)
insertCollection('06_NationalIdentity_Sentiment', 'sentiment_subcomment_Collection', sub_comment_data)

del post
del comment
del sub_comment


"""
************************************************ LEMMITIZATION *********************************************************
"""

# S 06_NationalIdentity_Sentiment
# T 07_PreProcessing

#TODO Compare columns from previous step and test, source and target dbnames need to be changed
textCleaning = TextCleaning()

df_post = getCollection('06_NationalIdentity_Sentiment', 'sentiment_post_Collection')
df_post = textCleaning.get_clean_df(df=df_post, col="onlyText")
insertCollection('07_PreProcessing', 'ni_post_preprocessed', df_post)

df_comment = getCollection('06_NationalIdentity_Sentiment', 'sentiment_comment_Collection')
df_comment = textCleaning.get_clean_df(df=df_comment, col="onlyText")
insertCollection('07_PreProcessing', 'ni_comment_preprocessed', df_comment)

df_subcomment = getCollection('06_NationalIdentity_Sentiment', 'sentiment_subcomment_Collection')
df_subcomment = textCleaning.get_clean_df(df=df_subcomment, col="onlyText")
insertCollection('07_PreProcessing', 'ni_subcomment_preprocessed', df_subcomment)


#TODO Merge Motive and National Identity under one class
""" 
************************************************ TAGGING **************************************************************
***************************************** NATIONAL IDENTITY TAGGING ****************************************************
"""
#TODO Compare columns from previous step and test, source and target dbnames need to be changed

# T 07_PreProcessing
# S 08_NationalIdentity
# 05_NationalIdentity Needs to be changed

nationalIdentityTaggingObj = NationalIdentityTagging()
nationalIdentityTaggingObj.get_flags()

df = getCollection('07_PreProcessing', 'sentiment_post_Collection')
nationalIdentityTaggingObj.postData(df, 'ni_post') ##TODO Change the method name postData sound too specific

df = getCollection('07_PreProcessing', 'sentiment_comment_Collection')
df.rename(columns={'Comment': 'text'}, inplace=True)
nationalIdentityTaggingObj.postData(df, 'ni_comment')

df = getCollection('07_PreProcessing', 'sentiment_subcomment_Collection')
df.rename(columns={'Sub_Comment': 'text'}, inplace=True)
nationalIdentityTaggingObj.postData(df, 'ni_subcomment')


"""
****************************************** IDENTITY MOTIVE TAGGING *****************************************************
"""

# T 07_PreProcessing
# S 08_PreTrain

#TODO Compare columns from previous step and test, source and target dbnames need to be changed
identityMotiveTagging = IdentityMotiveTagging()
identityMotiveTagging.get_synonyms()

df = getCollection('07_PreProcessing', 'ni_post_preprocessed')
df = identityMotiveTagging.tagging(df)
insertCollection('08_PreTrain', 'train_data', df)

df = getCollection('07_PreProcessing', 'ni_comment_preprocessed')
df = identityMotiveTagging.tagging(df)
insertCollection('08_PreTrain', 'train_data', df)

df = getCollection('07_PreProcessing', 'ni_subcomment_preprocessed')
df = identityMotiveTagging.tagging(df)
insertCollection('08_PreTrain', 'train_data', df)

# motive.py
# tagging should be here, possible database name changes

"""
************************************************ DATA BALANCING ********************************************************
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
*********************************************** FEATURE SELECTION ******************************************************
"""

filepath = "D:/OneDrive - SRH IT/Case Study 1/02 Input_Data/03 Model"
featureSelection = FeatureSelection(1000, filepath)
featureSelection.balancing()

"""
****************************************************** MODEL ***********************************************************
"""
#TODO Need steps from Max

"""
************************************************ PREDICTION ************************************************************
"""
#TODO Need steps from Max



