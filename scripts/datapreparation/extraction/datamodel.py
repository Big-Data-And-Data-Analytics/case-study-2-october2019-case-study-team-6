from pymongo import MongoClient
import pandas as pd
#import scripts.mongoConnection as mongocon

class DataModel:
    """DataModel class provides functions to apply the datamodel on the given datasource and combines them into a unified shape
    """

    def __init__(self):
        pass

    def rename_columns_post(self, data, source_col_names):
        """[summary]

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
            source_col_names[6] : "source_type",
            source_col_names[7] : "data_type"
        })
        return data_renamed

if __name__ == "__main__":

################## REMOVE AGAIN BECAUSE PROBLEM WITH IMPORT
    def getCollection(db, col):
        """Retruns the data for a given database and collection

        :param db: Database that should be returned
        :type db: String
        :param col: Collection that should be returned
        :type col: String
        :return: Data from the given collection
        :rtype: pandas dataframe
        """    
        conn = MongoClient("localhost" , 27017)
        collobj = conn[db][col]
        collection = pd.DataFrame(list(collobj.find({})))
        return collection.copy()
##################


#### YOUTUBE POST ####
    # Get the data from MongoDB
    youTube_Video_Stats_Cleaned = getCollection("02_NationalIdentity_Cleaned", "youTube_Video_Stats_Cleaned")
    youTube_Videos_Cleaned = getCollection("02_NationalIdentity_Cleaned", "youTube_Videos_Cleaned")

    youTube_Video_Stats_Cleaned = youTube_Video_Stats_Cleaned.drop(columns="_id")
    youTube_Videos_Cleaned = youTube_Videos_Cleaned.drop(columns="_id")

    youTube_Videos_Data = youTube_Videos_Cleaned.set_index("video_id").join(youTube_Video_Stats_Cleaned.set_index("video_id"), on="video_id", how="left", lsuffix="_caller")

    youTube_Videos_Data["source_type"] = "YouTube"
    youTube_Videos_Data["data_type"] = "post"

    # Instanceiate class
    apply_model = DataModel()

    youTube_Videos_Data = apply_model.rename_columns_post(youTube_Videos_Data, source_col_names=[
        "video_id",
        "channel",
        "description",
        "viewCount",
        "publishedAt",
        "channel",
        "source_type",
        "data_type"])

    youTube_Videos_Data = youTube_Videos_Data.reset_index()

#### YOUTUBE COMMENT ####
    youTube_Video_Comments_Cleaned = getCollection("02_NationalIdentity_Cleaned", "youTube_Video_Comments_Cleaned")

    youTube_Video_Comments_Cleaned = youTube_Video_Comments_Cleaned.drop(columns="_id")

    youTube_Video_Comment = youTube_Video_Comments_Cleaned["parentId"].dropna()
    youTube_Video_Comment = youTube_Video_Comments_Cleaned.join(youTube_Video_Comment, on="parentId", how="right", lsuffix="_caller")
    
    youTube_Videos_Data = apply_model.rename_columns_post(youTube_Videos_Data, source_col_names=[
        "Id",
        "Comment_Id",
        "Comment_Owner",
        "Comment",
        "Comment_Likes",
        "Comment_Time_Posted",
        "hashtags",
        "TagList",
        "onlyText",
        "source_Type",
        "data_Type"])
