from scripts.datapreparation.cleaning.cleaning import Cleaning
from scripts.mongoConnection import getCollection, insertCollection
from scripts.datapreparation.extraction.extraction import ExtractTagHashtag

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

# TODO Move the insert statements to decorator and fillna()
result_post = result_post.fillna('0')
result_comment = result_comment.fillna('0')
result_subcomment = result_subcomment.fillna('0')

# TODO Change the name of the Combined DB Remove Numbering, Collection name
insertCollection('03_NationalIdentity_Combined', 'common_post_inserted', result_post)
insertCollection('03_NationalIdentity_Combined', 'common_comment_inserted', result_comment)
insertCollection('03_NationalIdentity_Combined', 'common_subcomment_inserted', result_subcomment)

"""
04. Translation
"""


"""
05. Sentiment
"""



"""
06. Lemmitization
"""




"""
07. Tagging
    a. Identity Motives
    b. National Identity
"""



"""
08. Data Balancing 
"""

"""
 
"""


