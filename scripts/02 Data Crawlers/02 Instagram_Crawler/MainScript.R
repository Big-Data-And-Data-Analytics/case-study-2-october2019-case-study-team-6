#Script
#1.	Purpose
#    MainScript.R is used a as an orchestrator for the Instagram Crawler.
#    Which is required to run the functions responsible for Acquiring Instagram Post Links and then crawling these posts one by one,
#    and then extracting comments from those posts
#
#2. Output
#    MongoDB Collections - post, collection
#
#3.	Requirements
#    Hastags.txt
#    1.	Input
#    #Hastag

## TODO Replace the Static Db and Collection names, include them in the main script
## TODO Testing Pending
#Libraries and Functions
setwd("E:/Instagram-Crawler")
source("packages.R")
source("CommentExtraction.R")
source("URLExtractor.R")

#Set (initial) parameters
hashtagCounter <- 1
sysSleepTimer <- 5
index <- 1
conComment <- ""
conPost <- "" 
conTest <- ""
alternativeCode <<- 0
i <- 1
#########################################################################################
############  MENTION THE DATABASE AND THE COLLECTIONS CREATED FOR STORAGE  #############
#########################################################################################

db_name <- "admin"
cursor_url_collection <- "test"
post_collection <-  "post"
comment_collection <- "comment"



#########################################################################################


#Load and set first hashtag
hashtagsData <- read_csv("hashtags.txt")
hashtag <<- hashtagsData[hashtagCounter,1]
hashtag <<- "euro2016"

#Get first posts
url_start <- paste("http://instagram.com/explore/tags/", hashtag, "/?__a=1", sep = "")
json <- fromJSON(url_start)
edge_hashtag_to_media <- json$graphql$hashtag$edge_hashtag_to_media
posts <- edge_hashtag_to_media$edges$node
totalPostCount <- json[["graphql"]][["hashtag"]][["edge_hashtag_to_media"]][["count"]]
end_cursor <-  ""
#Start crawling
print(paste("Hashtag:", hashtag))
print(paste("TotalPosts:", totalPostCount))

###################################################################
####### DatabaseConnections() as the name suggests, 
####### all the database connections are created in the function
####### below
###################################################################

# databaseConnections <- Has all the details for the mongodb connections

databaseConnections <- function(){
  conComment <<- mongo(
    collection = comment_collection,
    db = db_name,
    url = "mongodb://localhost",
    verbose = FALSE,
    options = ssl_options()
  )
  
  conPost <<- mongo(
    collection = post_collection,
    db = db_name,
    url = "mongodb://localhost",
    verbose = FALSE,
    options = ssl_options()
  )
  
  conTest <<- mongo(
    collection = cursor_url_collection,
    db = db_name,
    url = "mongodb://localhost",
    verbose = FALSE,
    options = ssl_options()
  )
  
}


#############################################################
## latest_end_cursor() function is used  for checking whether any post is crawled before if not then 
# extract the latest cursor from the cursor data collection, start crawling.
#############################################################

reset <- function(err){
  print(err)
  Sys.sleep(300)
  latest_end_cursor()
}

latest_end_cursor <- function(){
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  print("!!!!!!!!!!!!                 CODE WAS RESET                !!!!!!!!!!!!!!!")
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  databaseConnections()
  #
  end_cursor_lastest <- conTest$find(query = '{"Type":"EC"}',sort = '{"_id":-1}',limit = 1)
  if(nrow(end_cursor_lastest) > 0){
    end_cursor <<- end_cursor_lastest$End_Cursor
  }
  else{
    end_cursor <<- edge_hashtag_to_media$page_info$end_cursor
  }
  print(end_cursor)
  # If in try block Crawler is blocked, catch will call latest_end_cursor() function so that the crawler again continues to crawl the data.
  tryCatch(extractInfo(index,conTest,conPost,conComment), error = function(err){ reset(err = err )} )
}

latest_end_cursor()

#For testing
#end_cursor = "QVFBdDRBSWJMRG1sT2NQYjVRd3Z1TWo5YVpWWG5oSEd0aWx6V2R4eUpBS2owUUMwY0NFempPdFVoeEE5dkJZdnZabzFhX1Jra3FsdDhVeXE0dUtabnV2RA=="
#QVFEMnZLX19HNzltOG56aHRKWGl4MnNRUDRvSXRCc1FLek9vU200NWN2Nk14U042V3lkVmxtZXR1ZEdlSXU4bWZpdHptVmJLRlU0NzY0VTF5SmlhV1NfQw==

