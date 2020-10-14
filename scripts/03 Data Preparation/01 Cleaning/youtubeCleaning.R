
# MongoDB Connection
youTube_Video_Comments_Con <- mongo(collection = "youTube_Video_Comments_Raw", db = "01_NationalIdentity_Crawled", url = "mongodb://localhost")
youTube_Video_Stats_Con <- mongo(collection = "youTube_Video_Stats_Raw", db = "01_NationalIdentity_Crawled", url = "mongodb://localhost")
youTube_Video_Videos_Con <- mongo(collection = "youTube_Videos_Raw", db = "01_NationalIdentity_Crawled", url = "mongodb://localhost")

# MongoDB Query Data
youTube_Video_Comments <-  youTube_Video_Comments_Con$find(query = "{}")
youTube_Video_Stats <-  youTube_Video_Stats_Con$find(query = "{}")
youTube_Video_Videos <-  youTube_Video_Videos_Con$find(query = "{}")

# Delete Duplicates
youTube_Video_Comments <- unique(youTube_Video_Comments)
youTube_Video_Stats <- unique(youTube_Video_Stats)
youTube_Video_Videos <- unique(youTube_Video_Videos)

# Delete linebreaks in data
youTube_Video_Comments$textDisplay <- gsub("[\r\n]", " ", youTube_Video_Comments$textDisplay)
youTube_Video_Comments$textOriginal <- gsub("[\r\n]", " ", youTube_Video_Comments$textOriginal)

youTube_Video_Videos$channel <- gsub("[\r\n]", " ", youTube_Video_Videos$channelTitle)
youTube_Video_Videos$description <- gsub("[\r\n]", " ", youTube_Video_Videos$description)

# Change datatype of dates
youTube_Video_Comments <- youTube_Video_Comments %>% 
  mutate(publishedAt = ymd_hms(publishedAt),
         publishedAt_TZ = attr(publishedAt, "tzone"),
         updatedAt = ymd_hms(updatedAt),
         updatedAt_TZ = attr(updatedAt, "tzone"),
         scan_Date = ymd_hms(scan_Date),
         scan_Date_TZ = attr(scan_Date, "tzone"))

# Replace "" in text with NA
for (x in 1:nrow(youTube_Video_Videos)) {
  if(nzchar(youTube_Video_Videos[x,5]) == FALSE){
    youTube_Video_Videos[x,5] <- NA
  }
}

for (x in 1:nrow(youTube_Video_Comments)) {
  if(nzchar(youTube_Video_Comments[x,2]) == FALSE){
    youTube_Video_Comments[x,2] <- NA
  }
}

for (x in 1:nrow(youTube_Video_Comments)) {
  if(nzchar(youTube_Video_Comments[x,3]) == FALSE){
    youTube_Video_Comments[x,3] <- NA
  }
}

# Create connection to new MongoDB collections
youTube_Video_Comments_Cleaned_Con <- mongo(collection = "youTube_Video_Comments_Cleaned", db = "02_NationalIdentity_Cleaned", url = "mongodb://localhost")
youTube_Video_Stats_Cleaned_Con <- mongo(collection = "youTube_Video_Stats_Cleaned", db = "02_NationalIdentity_Cleaned", url = "mongodb://localhost")
youTube_Video_Videos_Cleaned_Con <- mongo(collection = "youTube_Videos_Cleaned", db = "02_NationalIdentity_Cleaned", url = "mongodb://localhost")

# Drop existing collections
youTube_Video_Comments_Cleaned_Con$drop()
youTube_Video_Stats_Cleaned_Con$drop()
youTube_Video_Videos_Cleaned_Con$drop()

# Insert cleaned data into newly created collections
youTube_Video_Comments_Cleaned_Con$insert(youTube_Video_Comments)
youTube_Video_Stats_Cleaned_Con$insert(youTube_Video_Stats)
youTube_Video_Videos_Cleaned_Con$insert(youTube_Video_Videos)

print("YouTube cleaning done")
