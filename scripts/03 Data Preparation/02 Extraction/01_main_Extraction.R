library(mongolite)
library(dplyr)

source("03 Data Preparation/02 Extraction/main_Extraction_Functions.R")

before <- Sys.time()
source("03 Data Preparation/02 Extraction/TwitterData.R")
time_twit <-  Sys.time() - before

before <- Sys.time()
source("03 Data Preparation/02 Extraction/RedditData.R")
time_red <- Sys.time() - before

before <- Sys.time()
source("03 Data Preparation/02 Extraction/InstagramData.R")
time_insta <- Sys.time() - before

before <- Sys.time()
source("03 Data Preparation/02 Extraction/YoutubeData.R")
time_yt <- Sys.time() - before

paste("Time took: Twitter:", time_twit, "seconds")
paste("Time took: Reddit:", time_red, "seconds")
paste("Time took: Instagram:", time_insta, "seconds")
paste("Time took: YouTube:", time_yt, "seconds")
paste("Total time took:", time_yt + time_red + time_twit + time_insta)

# Benchmark results (Notebook, power supply not plugged in, 1 run each):
## Excluding Instagram
### R 4.0.2: 48 seconds
### R 3.5.2 (Microsoft R-Open): 46 seconds

# Benchmark results (Desktop, 1 run each):
## Including Instagram
### R 4.0.2: 2613 secs, about 45 mins
### R 3.5.2 (Microsoft R-Open): 1875 secs, about 31 mins

# Establish MongoDB connections
common_post <- mongo(collection = "common_post_Combined", db = "03_NationalIdentity_Combined", url = "mongodb://localhost")
common_comment <- mongo(collection = "common_comment_Combined", db = "03_NationalIdentity_Combined", url = "mongodb://localhost")
common_subcomment <- mongo(collection = "common_subcomment_Combined", db = "03_NationalIdentity_Combined", url = "mongodb://localhost")

# Connect to MongoDB
common_post$drop()
common_comment$drop()
common_subcomment$drop()

# Post
postData_file <- renamecolumns_n_combinerows_post()

postData_file <- unique(postData_file)

common_post$insert(postData_file)

# Comment
comment_file <- renamecolumns_n_combinerows_comment()

comment_file <- unique(comment_file)

common_comment$insert(comment_file)

# Subcomment
sub_comment_file <- renamecolumns_n_combinerows_subcomment()

sub_comment_file <- unique(sub_comment_file)

common_subcomment$insert(sub_comment_file)

# Clear the whole environment
#rm(list = ls())
