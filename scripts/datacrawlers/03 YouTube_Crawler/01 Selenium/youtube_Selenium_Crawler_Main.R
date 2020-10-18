source("02 Data Crawlers/03 YouTube_Crawler/01 Selenium/youTube_Selenium_Crawler_Functions.R")

# Connect to MongoDB
con_Videos <- mongo(collection = "youTube_Videos", db = "02_NationalIdentity_Crawled", url = "mongodb://localhost")

# Variables
youTubeDataFrame = 0
totalCommentsCounter <<- 0
videoScrolls = 20
windowSizeX = 1920
windowSizeY = 1080
hashtag <- "euro+2016"

duplicateVideoDetection <- data.frame(
  "Video_Url" = character(0),
  "Video_Title" = character(0),
  "Channel_name" = character(0),
  "Clicks" = character(0),
  "Time_Posted" = character(0),
  "Description" = character(0)
)

############# START CONTAINER WITH CHROME/CONNECT TO CONTAINER #############
#Create container
#rstudioapi::terminalExecute("docker run -d -p 4445:4444 selenium/standalone-firefox", show = FALSE)
#rstudioapi::terminalExecute("docker run -d -p 4445:4444 -v /dev/shm:/dev/shm selenium/standalone-firefox:3.141.59-yttrium", show = FALSE)
rstudioapi::terminalExecute("docker run -d -p 4445:4444 -v /dev/shm:/dev/shm selenium/standalone-chrome:3.141.59-yttrium", show = FALSE) 

remDr <- dockerConnection(browserName = "chrome", windowSizeX = windowSizeX, windowSizeY = windowSizeY)

############ EXTRACT VIDEO_IDS AND META DATA FROM YOUTUBE VIDEO OVERVIEW PAGE ############
videoURLs <- getVideos(hashtag = hashtag)
print(paste(videoCounter, "videos found, saving to MongoDB..."))

############ SAVE VIDEO_IDS AND META DATA INTO MONGO ############
con_Videos$insert(videoURLs)
print(paste(videoCounter, "videos stored in MongoDB."))
