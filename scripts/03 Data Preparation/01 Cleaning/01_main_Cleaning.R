library(tidyverse)
library(mongolite)
library(lubridate)

# Run cleaning scripts
source("03 Data Preparation/01 Cleaning/instagramCleaning.R")
source("03 Data Preparation/01 Cleaning/redditCleaning.R")
source("03 Data Preparation/01 Cleaning/twitterCleaning.R")
source("03 Data Preparation/01 Cleaning/youtubeCleaning.R")

print("Cleaning done.")
