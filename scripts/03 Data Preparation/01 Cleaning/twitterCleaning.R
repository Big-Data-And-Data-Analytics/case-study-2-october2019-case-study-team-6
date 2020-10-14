
# MongoDB connection to existing collection
twitter_Data_Con <- mongo(collection = "Twitter_Post_Raw", db = "01_NationalIdentity_Crawled", url = "mongodb://localhost")

# MongoDB connection to future collection
twitter_Data_Con_Cleaned <- mongo(collection = "Twitter_Post_Cleaned", db = "02_NationalIdentity_Cleaned", url = "mongodb://localhost")


twitterData <- twitter_Data_Con$find(query = "{}")

twitterData$text <- gsub("[\r\n]", " ", twitterData$text)
twitterData$text_html <- gsub("[\r\n]", " ", twitterData$text_html)

twitter_Data_Con_Cleaned$insert(twitterData)

rm(twitterData)
rm(twitter_Data_Con, twitter_Data_Con_Cleaned)

print("Twitter cleaning done")
