import pymongo
import pandas as pd
from datapreparation.mongoConnection import getCollection, insertCollection

class ExtractTagHashtag:
    """ExtractTagHashtag class represents extraction of hashtags, taglists and text without hashtags, taglists i.e onlyText
        from the text
        The sequentially used functions  are:
        1) :func : extract
        2) :func : decorator_hashtag
        3) :func : decorator_taglist
    """
    def decorator_taglist(func):
        """ Extracts hashtag from the text and replaces the text having hashtags with space

        When the text is given to this decorator, the hashtag is extracted and stored in a new column named "hashtag".
        After a hashtag is stored in the hashtag column, the hashtag is replaced as a space in the text and stored in
        a new column "onlyText".
        If a hashtag is not present in the text, then the hastag column remains blank and it moves to the next text

        :param text: text from the column "text" of a dataframe
        :type text: string 

        :return hashtag- takes the words starting with #
        :return onlyText- text without hashtags
        """
        def wrapper_function(extract_taglist):
            hashtag_removed = func(extract_taglist)
            hashtag_removed['tagList'] = hashtag_removed['text'].str.findall(r'@.*?(?=\s|$)')
            hashtag_removed['tagList'] = pd.Series(hashtag_removed['tagList'], dtype="string")
            hashtag_removed['tagList'] =  hashtag_removed['tagList'].str.strip("[]")
            hashtag_removed['onlyText'] = hashtag_removed['onlyText'].str.replace(r'@.*?(?=\s|$)',"")
            return hashtag_removed
        return wrapper_function

    def decorator_hashtag(func): 
        """ Extracts taglist from the text and replaces the text having taglist with space

        When the text is given to this decorator, the taglist is extracted and stored in a new column named "taglist".
        After a taglist is stored in the taglist column, the taglist is replaced as a space in the onlytext and stored in
        a column "onlyText".
        If a taglist is not present in the text, then the taglist column remains blank and it moves to the next text

        :param : text from the column "text" of a dataframe
        :type text: string 

        :param : text from the column "onlyText" of a dataframe
        :type text: string

        :return taglist- takes the words starting with @
        :return onlyText- text without taglist
        """
        def wrapper_function(extract_hashtag): 
            extract_hashtag['hashtag'] = extract_hashtag['text'].str.findall(r'#.*?(?=\s|$)')
            extract_hashtag['hashtag'] = pd.Series(extract_hashtag['hashtag'], dtype="string")
            extract_hashtag['hashtag'] =  extract_hashtag['hashtag'].str.strip("[]")
            extract_hashtag['onlyText'] = extract_hashtag['text'].str.replace(r'#.*?(?=\s|$)',"")
            return extract_hashtag
        return wrapper_function

    @decorator_taglist
    @decorator_hashtag
    def extract(self,post):
        """ Takes text as the input and as decorator_hashtag is present, the entire function 
        is passed to decorator_hashtag
        The value returned from the decorator_hashtag is passed to decorator_taglist

        :param post: text from the text column of the dataframe
        """
        return post



if __name__ == "__main__":

    df_post = getCollection('03_NationalIdentity_Combined', 'common_post_Combined')
    df_comment = getCollection('03_NationalIdentity_Combined', 'common_comment_Combined')
    df_subcomment = getCollection('03_NationalIdentity_Combined', 'common_subcomment_Combined')
    df_comment.rename(columns={'Comment':'text'}, inplace=True)
    df_subcomment.rename(columns={'Sub_Comment':'text'}, inplace=True)

    result_post = ExtractTagHashtag.extract(df_post)
    result_comment = ExtractTagHashtag.extract(df_comment)
    result_subcomment = ExtractTagHashtag.extract(df_subcomment)

    result_post = result_post.fillna('0')
    result_comment = result_comment.fillna('0')
    result_subcomment = result_subcomment.fillna('0')
    insertCollection('03_NationalIdentity_Combined', 'common_post_inserted', result_post)
    insertCollection('03_NationalIdentity_Combined', 'common_comment_inserted', result_comment)
    insertCollection('03_NationalIdentity_Combined', 'common_subcomment_inserted', result_subcomment)
