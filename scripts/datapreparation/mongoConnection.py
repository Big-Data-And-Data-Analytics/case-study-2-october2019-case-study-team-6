from pymongo import MongoClient
import pandas as pd

def connectMongo(db, col):
    conn = MongoClient("localhost" , 27017)
    return conn[db][col]

def getCollection(col):
    collection = pd.DataFrame(list(col.find({})))
    return collection.copy()

def insertCollection(db, col, result):
    result = result.to_dict("records")
    conn = MongoClient("localhost" , 27017)
    connObj =  conn[db][col]
    connObj.insert_many(result)