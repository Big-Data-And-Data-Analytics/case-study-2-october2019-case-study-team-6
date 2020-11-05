import pandas as pd
from scripts.mongoConnection import getCollection, insertCollection

class DataModel:
    """DataModel class provides functions to apply the datamodel on the given datasource and combines them into a unified shape
    """

    def __init__(self):
        pass

    def select_and_rename_columns_post(self, data, source_col_names):
        """Selects and renames all columns to the post specified data model

        :param data: [description]
        :type data: [type]
        :param source_col_names: [description]
        :type source_col_names: [type]
        :return: [description]
        :rtype: [type]
        """
        data_filtered = data.filter(items=[
            source_col_names[0],
            source_col_names[1],
            source_col_names[2],
            source_col_names[3],
            source_col_names[4],
            source_col_names[5],
            source_col_names[6],
            source_col_names[7]
        ]
        )

        data_renamed = data_filtered.rename(columns={
            source_col_names[0] : "Id",
            source_col_names[1] : "username",
            source_col_names[2] : "text",
            source_col_names[3] : "likes",
            source_col_names[4] : "timestamp",
            source_col_names[5] : "owner_id",
            source_col_names[6] : "source_Type",
            source_col_names[7] : "data_Type"
        })
        return data_renamed

    def select_and_rename_columns_comment(self, data, source_col_names):
        """Selects and renames all columns to the comment specified data model

        :param data: [description]
        :type data: [type]
        :param source_col_names: [description]
        :type source_col_names: [type]
        :return: [description]
        :rtype: [type]
        """
        data_filtered = data.filter(items=[
            source_col_names[0],
            source_col_names[1],
            source_col_names[2],
            source_col_names[3],
            source_col_names[4],
            source_col_names[5],
            source_col_names[6],
            source_col_names[7]
        ])

        data_renamed = data_filtered.rename(columns={
            source_col_names[0] : "Id",
            source_col_names[1] : "Comment_Id",
            source_col_names[2] : "Comment_Owner",
            source_col_names[3] : "Comment",
            source_col_names[4] : "Comment_Likes",
            source_col_names[5] : "Comment_Time_Posted",
            source_col_names[6] : "source_Type",
            source_col_names[7] : "data_Type"
        })
        return data_renamed

    def select_and_rename_columns_subcomment(self, data, source_col_names):
        """Selects and renames all columns to the subcomment specified data model

        :param data: [description]
        :type data: [type]
        :param source_col_names: [description]
        :type source_col_names: [type]
        :return: [description]
        :rtype: [type]
        """
        data_filtered = data.filter(items=[
            source_col_names[0],
            source_col_names[1],
            source_col_names[2],
            source_col_names[3],
            source_col_names[4],
            source_col_names[5],
            source_col_names[6],
            source_col_names[7],
            source_col_names[8]
        ])

        data_renamed = data_filtered.rename(columns={
            source_col_names[0] : "Id",
            source_col_names[1] : "Sub_Comment_Id",
            source_col_names[2] : "Sub_Comment_Parent_Id",
            source_col_names[3] : "Sub_Comment_Owner",
            source_col_names[4] : "Sub_Comment",
            source_col_names[5] : "Sub_Comment_Likes",
            source_col_names[6] : "Sub_Comment_Time_Posted",
            source_col_names[7] : "source_Type",
            source_col_names[8] : "data_Type"
        })
        return data_renamed

    def add_data_and_source_type(self, data, data_Type, source_Type):
        """Adds the data and source type column to the dataframe

        :param data: [description]
        :type data: [type]
        :param data_Type: [description]
        :type data_Type: [type]
        :param source_Type: [description]
        :type source_Type: [type]
        :return: [description]
        :rtype: [type]
        """        
        data["source_Type"] = source_Type
        data["data_Type"] = data_Type
        return data
    
    def run_youtube_post(self, data1, data2):
        """[summary]

        :param data1: [description]
        :type data1: [type]
        :param data2: [description]
        :type data2: [type]
        :return: [description]
        :rtype: [type]
        """        
        youTube_Video_Stats_Cleaned = data1.drop(columns="_id")
        youTube_Videos_Cleaned = data2.drop(columns="_id")

        youTube_Videos_Data = pd.merge(youTube_Videos_Cleaned, youTube_Video_Stats_Cleaned, how="left", on=["video_id"])
        youTube_Videos_Data = apply_model.add_data_and_source_type(youTube_Videos_Data, "post", "YouTube")
        youTube_Videos_Data = apply_model.select_and_rename_columns_post(data=youTube_Videos_Data, source_col_names=[
            "video_id",
            "channel",
            "description",
            "viewCount",
            "publishedAt",
            "channel",
            "source_Type",
            "data_Type"
        ])

        youTube_Videos_Data.drop_duplicates(inplace=True)
        return youTube_Videos_Data

    def run_youtube_comment(self, data):
        """[summary]

        :param data: [description]
        :type data: [type]
        :return: [description]
        :rtype: [type]
        """        
        youTube_Video_Comments_Cleaned = data.drop(columns="_id")
        youTube_Video_Comments_Cleaned["parentId_check"] = youTube_Video_Comments_Cleaned.parentId.isnull()

        youTube_Video_Comments = youTube_Video_Comments_Cleaned[youTube_Video_Comments_Cleaned.parentId_check.eq(True)]
        youTube_Video_Comments = apply_model.add_data_and_source_type(youTube_Video_Comments, "comment", "YouTube")
        youTube_Video_Comments = apply_model.select_and_rename_columns_comment(data=youTube_Video_Comments, source_col_names=[
            "videoId",
            "id",
            "authorDisplayName",
            "textOriginal",
            "likeCount",
            "publishedAt",
            "source_Type",
            "data_Type"
        ])

        youTube_Video_Comments.drop_duplicates(inplace=True)
        return youTube_Video_Comments

    def run_youtube_subcomment(self, data):
        """[summary]

        :param data: [description]
        :type data: [type]
        :return: [description]
        :rtype: [type]
        """        
        youTube_Video_Comments_Cleaned = data.drop(columns="_id")
        youTube_Video_Comments_Cleaned = youTube_Video_Comments_Cleaned.astype({'parentId': 'string'})
        youTube_Video_Comments_Cleaned["parentId_check"] = youTube_Video_Comments_Cleaned.parentId.notnull()

        youTube_Video_SubComments = youTube_Video_Comments_Cleaned[youTube_Video_Comments_Cleaned.parentId_check.eq(True)]
        youTube_Video_SubComments = apply_model.add_data_and_source_type(youTube_Video_SubComments, "subcomment", "YouTube")
        youTube_Video_SubComments = apply_model.select_and_rename_columns_subcomment(data=youTube_Video_SubComments, source_col_names=[
            "videoId",
            "id",
            "parentId",
            "authorDisplayName",
            "textOriginal",
            "likeCount",
            "publishedAt",
            "source_Type",
            "data_Type"
        ])

        youTube_Video_SubComments.drop_duplicates(inplace=True)
        return youTube_Video_SubComments

    def run_reddit_post(self, data):
        """[summary]

        :param data: [description]
        :type data: [type]
        :return: [description]
        :rtype: [type]
        """        
        reddit_Data_Post = apply_model.add_data_and_source_type(data, "post", "Reddit")
        reddit_Data_Post = apply_model.select_and_rename_columns_post(data=reddit_Data_Post, source_col_names=[
            "post_id",
            "author",
            "post_text",
            "post_score",
            "post_date",
            "author",
            "source_Type",
            "data_Type"
        ])

        reddit_Data_Post.drop_duplicates(inplace=True)
        return reddit_Data_Post

    def run_reddit_comment(self, data):
        """[summary]

        :param data: [description]
        :type data: [type]
        :return: [description]
        :rtype: [type]
        """        
        reddit_Data_Comment = apply_model.add_data_and_source_type(data, "comment", "Reddit")
        reddit_Data_Comment = reddit_Data_Comment[~reddit_Data_Comment["structure"].str.contains("_")]
        reddit_Data_Comment = apply_model.select_and_rename_columns_comment(data=reddit_Data_Comment, source_col_names=[
            "post_id",
            "structure",
            "user",
            "comment",
            "comment_score",
            "post_date",
            "source_Type",
            "data_Type" 
        ])

        reddit_Data_Comment.drop_duplicates(inplace=True)
        return reddit_Data_Comment

    def run_reddit_subcomment(self, data):
        """[summary]

        :param data: [description]
        :type data: [type]
        :return: [description]
        :rtype: [type]
        """        
        reddit_Data_Subcomment = apply_model.add_data_and_source_type(data, "subcomment", "Reddit")
        reddit_Data_Subcomment = reddit_Data_Subcomment[reddit_Data_Subcomment["structure"].str.contains("_")]
        reddit_Data_Subcomment = apply_model.select_and_rename_columns_subcomment(data=reddit_Data_Subcomment, source_col_names=[
            "post_id",
            "structure",
            "structure",
            "user",
            "comment", 
            "comment_score", 
            "post_date", 
            "source_Type",
            "data_Type"
        ])

        reddit_Data_Subcomment.drop_duplicates(inplace=True)
        return reddit_Data_Subcomment

    def run_twitter_post(self, data):
        """[summary]

        :param data: [description]
        :type data: [type]
        :return: [description]
        :rtype: [type]
        """        
        twitter_post = apply_model.add_data_and_source_type(data, "post", "Twitter")
        twitter_post = apply_model.select_and_rename_columns_post(data=twitter_post, source_col_names=[
            "tweet_id",
            "screen_name",
            "text",
            "likes",
            "timestamp",
            "user_id",
            "source_Type",
            "data_Type"
        ])

        twitter_post.drop_duplicates(inplace=True)
        return twitter_post

    def run_instagram_post(self, data):
        """[summary]

        :param data: [description]
        :type data: [type]
        :return: [description]
        :rtype: [type]
        """        
        instagram_post = apply_model.add_data_and_source_type(data, "post", "Instagram")
        instagram_post = apply_model.select_and_rename_columns_post(data=instagram_post, source_col_names=[
            "post_Id",
            "owner_username",
            "post_Caption",
            "post_Like_Count",
            "post_Time_Posted",
            "owner_id",
            "source_Type",
            "data_Type"
        ])

        instagram_post.drop_duplicates(inplace=True)
        return instagram_post

    def run_instagram_comment(self, data):
        """[summary]

        :param data: [description]
        :type data: [type]
        :return: [description]
        :rtype: [type]
        """        
        instagram_comment = apply_model.add_data_and_source_type(data, "comment", "Instagram")
        instagram_comment = apply_model.select_and_rename_columns_comment(data=instagram_comment, source_col_names=[
            "post_Id",
            "comment_id",
            "comment_owner_username",
            "comment_Text",
            "comment_liked_count",
            "comment_Time_Posted",
            "source_Type",
            "data_Type"
        ])

        instagram_comment.drop_duplicates(inplace=True)
        return instagram_comment

    def run_instagram_subcomment(self, data):
        """[summary]

        :param data: [description]
        :type data: [type]
        :return: [description]
        :rtype: [type]
        """        
        comment_comment_id_check = data.comment_comment_id != "NULL"

        instagram_comment_Cleaned = data[comment_comment_id_check==True]
        instagram_subcomment = instagram_comment_Cleaned.copy()
        instagram_subcomment = apply_model.add_data_and_source_type(instagram_comment_Cleaned, "subcomment", "Instagram")

        instagram_subcomment = apply_model.select_and_rename_columns_subcomment(data=instagram_subcomment, source_col_names=[
            "post_Id",
            "comment_comment_id",
            "comment_id",
            "comment_comment_owner_username",
            "comment_comment_text",
            "comment_comment_liked_count",
            "comment_comment_Time_Posted",
            "source_Type",
            "data_Type"
        ])

        instagram_subcomment.drop_duplicates(inplace=True)
        return instagram_subcomment


if __name__ == "__main__":

    apply_model = DataModel()

    #### YOUTUBE POST ####
    # Get the data from MongoDB
    youTube_Video_Stats_Cleaned = getCollection("02_NationalIdentity_Cleaned", "youTube_Video_Stats_Cleaned")
    youTube_Videos_Cleaned = getCollection("02_NationalIdentity_Cleaned", "youTube_Videos_Cleaned")

    youTube_Videos_Data = apply_model.run_youtube_post(data1=youTube_Video_Stats_Cleaned, data2=youTube_Videos_Cleaned)

    insertCollection("03_NationalIdentity_Combined_Test", "common_post_Combined", youTube_Videos_Data)

    #### YOUTUBE COMMENT ####
    youTube_Video_Comments_Cleaned = getCollection("02_NationalIdentity_Cleaned", "youTube_Video_Comments_Cleaned")

    youTube_Video_Comments = apply_model.run_youtube_comment(data=youTube_Video_Comments_Cleaned)

    insertCollection("03_NationalIdentity_Combined_Test", "common_comment_Combined", youTube_Video_Comments)

    #### YOUTUBE SUBCOMMENT ####
    youTube_Video_Comments_Cleaned = getCollection("02_NationalIdentity_Cleaned", "youTube_Video_Comments_Cleaned")

    youTube_Video_SubComments = apply_model.run_youtube_subcomment(data=youTube_Video_Comments_Cleaned)

    insertCollection("03_NationalIdentity_Combined_Test", "common_subcomment_Combined", youTube_Video_SubComments)

    #### REDDIT POST ####
    reddit_Data_Cleaned = getCollection("02_NationalIdentity_Cleaned", "reddit_Data_Cleaned")

    reddit_Data_Post = apply_model.run_reddit_post(data=reddit_Data_Cleaned)

    insertCollection("03_NationalIdentity_Combined_Test", "common_post_Combined", reddit_Data_Post)

    #### REDDIT COMMENT ####
    reddit_Data_Cleaned = getCollection("02_NationalIdentity_Cleaned", "reddit_Data_Cleaned")

    reddit_Data_Comment = apply_model.run_reddit_comment(data=reddit_Data_Cleaned)

    insertCollection("03_NationalIdentity_Combined_Test", "common_comment_Combined", reddit_Data_Comment)

    #### REDDIT SUBCOMMENT ####
    reddit_Data_Cleaned = getCollection("02_NationalIdentity_Cleaned", "reddit_Data_Cleaned")

    reddit_Data_Subcomment = apply_model.run_reddit_subcomment(data=reddit_Data_Cleaned)

    insertCollection("03_NationalIdentity_Combined_Test", "common_subcomment_Combined", reddit_Data_Subcomment)

    #### TWITTER POST ####
    twitter_Data_Cleaned = getCollection("02_NationalIdentity_Cleaned", "Twitter_Post_Cleaned")

    twitter_post = apply_model.run_twitter_post(data=twitter_Data_Cleaned)

    insertCollection("03_NationalIdentity_Combined_Test", "common_post_Combined", twitter_post)

    #### INSTAGRAM POST ####
    instagram_Data_Cleaned = getCollection("02_NationalIdentity_Cleaned", "Instagram_Post_Cleaned")

    instagram_post = apply_model.run_instagram_post(data=instagram_Data_Cleaned)

    insertCollection("03_NationalIdentity_Combined_Test", "common_post_Combined", instagram_post)

    #### INSTAGRAM COMMENT ####
    instagram_comment_Cleaned = getCollection("02_NationalIdentity_Cleaned", "Instagram_Comment_Cleaned")

    instagram_comment = apply_model.run_instagram_comment(data=instagram_comment_Cleaned)

    insertCollection("03_NationalIdentity_Combined_Test", "common_comment_Combined", instagram_comment)

    #### INSTAGRAM SUBCOMMENT ####
    instagram_comment_Cleaned = getCollection("02_NationalIdentity_Cleaned", "Instagram_Comment_Cleaned")

    instagram_subcomment = apply_model.run_instagram_subcomment(data=instagram_comment_Cleaned)

    insertCollection("03_NationalIdentity_Combined_Test", "common_subcomment_Combined", instagram_subcomment)
