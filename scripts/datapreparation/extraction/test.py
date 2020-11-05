import pandas as pd
from scripts.mongoConnection import getCollection, insertCollection

instagram_comment_Cleaned = getCollection("02_NationalIdentity_Cleaned", "Instagram_Comment_Cleaned")

source_Type = "Instagram"
data_Type = "Post"
class d:
    def test(self, data, source_Type, data_Type):
        data["source_Type"] = source_Type
        data["data_Type"] = data_Type
        return instagram_comment_Cleaned


objs = d()
r = objs.test(data=instagram_comment_Cleaned, source_Type=source_Type, data_Type=data_Type)
