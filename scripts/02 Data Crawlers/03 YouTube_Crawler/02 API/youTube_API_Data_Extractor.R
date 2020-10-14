library(tuber)
library(mongolite)
library(tidyverse)

con_Videos <- mongo(collection = "youTube_Videos_Raw", db = "01_NationalIdentity_Crawled", url = "mongodb://localhost")
con_VideoStats <- mongo(collection = "youTube_Video_Stats_Raw", db = "01_NationalIdentity_Crawled", url = "mongodb://localhost")
con_VideoComments <- mongo(collection = "youTube_Video_Comments_Raw", db = "01_NationalIdentity_Crawled", url = "mongodb://localhost")

app_id = "128997915733-ode3mjtrsvqetblm62k2fhckmnlifd5p.apps.googleusercontent.com"
app_secret = "TaVrhfmpGPtKNGDIqKqD992p"

# Mavis
#app_id = 1080527739876-kj90i9l6k744u6ubhe55kv265smblnq7.apps.googleusercontent.com
#app_secret = 5yIZSy4aek0BOHVfIbWggjyA

# Varad
#app_id = 63241825871-e05mn8t7qu4dt9nfs86rimooeolnp5lk.apps.googleusercontent.com
#app_secret = tE__R4AJ-a-qevkxAnDnWpLE

yt_oauth(app_id = app_id, app_secret = app_secret)

#videos <- yt_search(term = "euro 2016")
#con_Videos$drop()
#con_Videos$insert(videos)

videos <- con_Videos$find(query = "{}", fields = "{\"_id\":0, \"video_id\":1, \"title\":1}")


for (i in 122:nrow(videos)) {
  
  tryCatch(
    {
      print(paste("Data extracting from:", videos[i,2]))
      
      video_stats <- get_stats(videos[i,1])
      video_stats <- as.data.frame(video_stats)
      video_stats$scan_Date <- Sys.time()
      colnames(video_stats) <- c("video_id", "viewCount", "likeCount", "dislikeCount", "favoriteCount", "commentCount", "scan_Date")
      
      con_VideoStats$insert(video_stats)
      
      comments <- get_all_comments(videos[i,1])
      comments$scan_Date <- Sys.time()
      
      con_VideoComments$insert(comments)
    },
    error = function(cond) {
      
      print("An error ocurred")
      print(cond)
      
    },
    warning = function(cond) {
      
      print("An warning ocurred")
      print(cond)
      
    },
    finally = {
      
      print(paste("Done,", nrow(videos) - i, "videos left"))
      Sys.sleep(2)
      
    } # finally
  ) # tryCatch
} # for loop
