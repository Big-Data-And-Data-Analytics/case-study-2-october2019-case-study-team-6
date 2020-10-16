class Cleaning:
    def __init__(self):
        pass

    def removeWhiteSpaces(self, data, column): # Remove whitepsaces in the given data-column
        data[column] = data[column].str.strip()
        data[column] = data[column].str.replace("  +", " ")
        return data

    #def removeLineBreaks(self, data, column): # Remove linebreaksa in the given data-column; MongoDB won't allow linebreaks in the JSON, so this step is not necessary at this point and should be catched right after crawling, before importing to MongoDB
    #    data[column] = data[column].replace(to_replace="[\n\r]", value=" ", regex=True)
    #    print(f'Linebreaks removed from column {column}')
    #    return data

    def changeEmptyToNA(self, data, column): # Change empty strings to NA in the given data-column
        data[column].replace(r'^\s*$', np.nan, regex=True)

    def removeDuplicates(self, data): # Remove duplicates
        data = data.drop(columns="_id")
        data = data.drop_duplicates()
        print(f'Duplicates removed from column {data}')
        return data

########
from pymongo import MongoClient
import pandas as pd
import numpy as np

client = MongoClient('localhost', 27017)
db = client['01_NationalIdentity_Crawled']
data = db.youTube_Videos_Raw
data = data.find({})
data = list(data)


data = pd.DataFrame(list())

cleaner = Cleaning()

# Encodinc issue
# reddit
# youtube comments
# youtube videos