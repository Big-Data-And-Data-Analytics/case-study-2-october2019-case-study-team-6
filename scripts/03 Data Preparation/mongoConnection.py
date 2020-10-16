from pymongo import MongoClient
import pandas as pd

def connectMongo(db, col):
    conn = MongoClient("localhost" , 27017)
    return conn[db][col]

def getCollection(col):
    collection = pd.DataFrame(list(col.find({})))
    return collection.copy()