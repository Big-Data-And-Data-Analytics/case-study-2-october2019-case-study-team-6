from pymongo import MongoClient
import pandas as pd
from scripts.mongoConnection import getCollection, insertCollection

class DataModel:
    """DataModel class provides functions to apply the datamodel on the given datasource and combines them into a unified shape
    """

    def __init__(self):
        pass

    def rename_columns_post(self, data, source_col_names):
        """Renames all columns to the post specified data model

        :param data: [description]
        :type data: [type]
        :param source_col_names: [description]
        :type source_col_names: [type]
        :return: [description]
        :rtype: [type]
        """
        data_renamed = data.rename(columns={
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

    def rename_columns_comment(self, data, source_col_names):
        """Renames all columns to the comment specified data model

        :param data: [description]
        :type data: [type]
        :param source_col_names: [description]
        :type source_col_names: [type]
        :return: [description]
        :rtype: [type]
        """
        data_renamed = data.rename(columns={
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

    def rename_columns_subcomment(self, data, source_col_names):
        """Renames all columns to the subcomment specified data model

        :param data: [description]
        :type data: [type]
        :param source_col_names: [description]
        :type source_col_names: [type]
        :return: [description]
        :rtype: [type]
        """
        data_renamed = data.rename(columns={
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

#if __name__ == "__main__":

##################

apply_model = DataModel()

#### YOUTUBE POST ####
# Get the data from MongoDB
youTube_Video_Stats_Cleaned = getCollection("02_NationalIdentity_Cleaned", "youTube_Video_Stats_Cleaned")
youTube_Videos_Cleaned = getCollection("02_NationalIdentity_Cleaned", "youTube_Videos_Cleaned")

youTube_Video_Stats_Cleaned = youTube_Video_Stats_Cleaned.drop(columns="_id")
youTube_Videos_Cleaned = youTube_Videos_Cleaned.drop(columns="_id")

youTube_Videos_Data = youTube_Videos_Cleaned.set_index("video_id").join(youTube_Video_Stats_Cleaned.set_index("video_id"), on="video_id", how="left", lsuffix="_caller")

youTube_Videos_Data = apply_model.add_data_and_source_type(youTube_Videos_Data, "post", "YouTube")

youTube_Videos_Data = apply_model.rename_columns_post(youTube_Videos_Data, source_col_names=[
    "video_id",
    "channel",
    "description",
    "viewCount",
    "publishedAt",
    "channel",
    "source_Type",
    "data_Type"])

insertCollection("02_NationalIdentity_Combined_Test", "common_post_Combined", youTube_Videos_Data)

#### YOUTUBE COMMENT ####
youTube_Video_Comments_Cleaned = getCollection("02_NationalIdentity_Cleaned", "youTube_Video_Comments_Cleaned")
youTube_Video_Comments_Cleaned = youTube_Video_Comments_Cleaned.drop(columns="_id")

youTube_Video_Comments = youTube_Video_Comments_Cleaned.parentId.isnull()
youTube_Video_Comments = pd.DataFrame(youTube_Video_Comments)

youTube_Video_Comments = youTube_Video_Comments[youTube_Video_Comments.parentId.eq(True)]

youTube_Video_Comments = youTube_Video_Comments_Cleaned.join(youTube_Video_Comments, how="right", lsuffix="_caller")

youTube_Video_Comments = apply_model.add_data_and_source_type(youTube_Video_Comments, "comment", "YouTube")

youTube_Video_Comments = youTube_Video_Comments.filter(items=[
    "videoId",
    "id",
    "authorDisplayName",
    "textOriginal",
    "likeCount",
    "publishedAt",
    "source_Type",
    "data_Type"])

youTube_Video_Comments = apply_model.rename_columns_comment(youTube_Video_Comments, source_col_names=[
    "videoId",
    "id",
    "authorDisplayName",
    "textOriginal",
    "likeCount",
    "publishedAt",
    "source_Type",
    "data_Type"])

insertCollection("02_NationalIdentity_Combined_Test", "common_comment_Combined", youTube_Video_Comments)

#### YOUTUBE SUBCOMMENT ####
youTube_Video_Comments_Cleaned = getCollection("02_NationalIdentity_Cleaned", "youTube_Video_Comments_Cleaned")
youTube_Video_Comments_Cleaned = youTube_Video_Comments_Cleaned.drop(columns="_id")
youTube_Video_Comments_Cleaned = youTube_Video_Comments_Cleaned.astype({'parentId': 'string'})

youTube_Video_SubComments = youTube_Video_Comments_Cleaned["parentId"].dropna()
youTube_Video_SubComments = pd.DataFrame(youTube_Video_SubComments)
youTube_Video_SubComments['parentId'] = youTube_Video_SubComments['parentId'].astype('string')
youTube_Video_SubComments = youTube_Video_Comments_Cleaned.merge(youTube_Video_SubComments, on="parentId", how="right",
                                                                suffixes=["_cleaned", ""])

youTube_Video_SubComments = youTube_Video_SubComments.filter(items=[
    "videoId",
    "id",
    "parentId" # Might be called parent_Id_cleaned or parent_Id_joined
    "authorDisplayName",
    "textOriginal",
    "likeCount",
    "publishedAt"])

youTube_Video_SubComments = apply_model.add_data_and_source_type(youTube_Video_SubComments, "subcomment", "YouTube")

youTube_Video_SubComments = apply_model.rename_columns_subcomment(youTube_Video_SubComments, source_col_names=[
    "videoId",
    "id",
    "parentId"
    "authorDisplayName",
    "textOriginal",
    "likeCount",
    "publishedAt",
    "source_Type",
    "data_Type"])

insertCollection("02_NationalIdentity_Combined_Test", "common_subcomment_Combined", youTube_Video_SubComments)

#### REDDIT POST ####
reddit_Data_Cleaned = getCollection("02_NationalIdentity_Cleaned", "reddit_Data_Cleaned")
reddit_Data_Cleaned = reddit_Data_Cleaned.drop(columns="_id")

#reddit_Data_Post =