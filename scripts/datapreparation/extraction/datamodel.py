class DataModel:
    """DataModel class provides
    """

    def __init__(self):
        pass

    def apply_Post(self):
        pass

    def apply_Comments(self):
        pass

    def apply_SubComments(self):
        pass

    def rename_Columns(self, **kwargs, data_Type, source_Type):
        """Takes a given set of source column names to rename them into the specified data model
        """        
        pass

######################################
from pymongo import MongoClient
import pandas as pd

# Source
client = MongoClient('localhost', 27017)
db = client['02_NationalIdentity_Cleaned']
youTube_Video_Stats_Cleaned = db.youTube_Video_Stats_Cleaned	
youTube_Video_Stats_Cleaned = pd.DataFrame(list(youTube_Video_Stats_Cleaned.find({})))
youTube_Video_Stats_Cleaned = youTube_Video_Stats_Cleaned.drop(columns="_id")

youTube_Videos_Cleaned = db.youTube_Videos_Cleaned	
youTube_Videos_Cleaned = pd.DataFrame(list(youTube_Videos_Cleaned.find({})))
youTube_Videos_Cleaned = youTube_Videos_Cleaned.drop(columns="_id")

youTube_Videos_Data = youTube_Videos_Cleaned.set_index("video_id").join(youTube_Video_Stats_Cleaned.set_index("video_id"), on="video_id", how="left", lsuffix="_caller")

youTube_Videos_Data["source_type"] = "YouTube"
youTube_Videos_Data["data_type"] = "post"

youTube_Videos_Data = youTube_Videos_Data.reset_index()

youTube_Videos_Data = youTube_Videos_Data[[
    "video_id",
    "channel",
    "description",
    "viewCount",
    "publishedAt",
    "channel",
    "source_type",
    "data_type"]]

youTube_Common_Data_Post = youTube_Videos_Data.rename(columns={
    "video_id" : "Id",
    "channel" : "username",
    "description" : "text",
    "viewCount" : "likes",
    "publishedAt" : "timestamp",
    "channel" : "owner_id",
    "source_type" : "source_type",
    "data_type" : "data_type"
    })


insert_data = youTube_Common_Data_Post.to_dict("records")

# Target
db = client['02_NationalIdentity_Combined_Test']
collection = db.youTubeJoinTest
db.drop_collection(collection)
result = collection.insert_many(insert_data)
