# This is a optional script
# It was used to transfer the YouTube URLs in CSV form from one machine to another and then import it into MongoDB

library(mongolite)
library(readr)

con_Videos <- mongo(collection = "youTube_Videos", db = "01_NationalIdentity_Crawled", url = "mongodb://localhost")

videos_self_scanned <- read_csv("01 Setup/02 Input_Files/02 YouTube/YouTubeVideoURLs.csv")
videos_self_scanned$description <- gsub("[\r\n]", " ", videos_self_scanned$description)
videos_self_scanned$video_id <- gsub("/watch?v=", "", videos_self_scanned$video_id, fixed = TRUE)

con_Videos$insert(videos_self_scanned)
