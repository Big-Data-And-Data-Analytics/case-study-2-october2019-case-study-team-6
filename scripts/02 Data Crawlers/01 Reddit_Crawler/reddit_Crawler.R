library(RedditExtractoR)
library(dplyr)
library(mongolite)

#Github Repository Link of the package
# https://github.com/ivan-rivera/RedditExtractoR

# Variables to control the result of the API call
search_terms <- c("euro 2016", "em 2016")
numberOfPages <- 10

# Establish MongoDB connection
reddit_Con <- mongo(collection = "reddit_Data_Raw", db = "01_NationalIdentity_Crawled")

for (x in 1:length(search_terms)) {
  
  # Call API
  data_reddit <- get_reddit(search_terms = search_terms[x], page_threshold = numberOfPages, wait_time = 10, subreddit = "soccer")
  
  # Add the search term into a additional column
  data_reddit$search_phrase <- search_terms[x]
  
  # Add unique ids to the data
  data_reddit_id <- data_reddit %>% 
    distinct(title)
  
  data_reddit_id$post_id <- row.names(data_reddit_id)
  data_reddit <- left_join(data_reddit, data_reddit_id)
  data_reddit$comment_id <- row.names(data_reddit)
  
  # Write data to MongoDB
  reddit_Con$insert(data_reddit)
  
}
