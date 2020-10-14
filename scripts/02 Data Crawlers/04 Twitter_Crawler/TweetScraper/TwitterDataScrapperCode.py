# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:48:49 2019

@author: shubh
"""

from twitterscraper import query_tweets
import datetime as dt
import pandas as pd
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.admin
twitter_source = db.twitter_source



## TODO REPLACE WITH MONGODB
begin_date = dt.date(2016,6,5)    
end_date = dt.date(2016,8,10)

limit = 10000


tweets = query_tweets("#uefa2016", begindate=begin_date,enddate=end_date)
df = pd.DataFrame(t.__dict__ for t in tweets)

file_name = "Uefa2016Hastag_new" + ".csv"
df.to_csv(file_name,sep=';',index=False)