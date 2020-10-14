
# Establish MongoDB connection to existing collection
reddit_Con <- mongo(collection = "Reddit_Data_Raw", db = "01_NationalIdentity_Crawled")
# Establish MongoDB connection to new collection
reddit_Cleaned_Con <- mongo(collection = "reddit_Data_Cleaned", db = "02_NationalIdentity_Cleaned")

# Get the data from the database
data_reddit <- reddit_Con$find(query = "{}")

data_reddit$id <- gsub(" ", "", data_reddit$id)
data_reddit$num_comments <- gsub(" ", "", data_reddit$num_comments)
data_reddit$comment_score <- gsub(" ", "", data_reddit$comment_score)
data_reddit$post_score <- gsub(" ", "", data_reddit$post_score)
data_reddit$comment <- gsub("[\r\n]", " ", data_reddit$comment)
data_reddit$title <- gsub("[\r\n]", " ", data_reddit$title)
data_reddit$post_text <- gsub("[\r\n]", " ", data_reddit$post_text)

for (x in 1:nrow(data_reddit)) {
  if(nzchar(data_reddit[x,13]) == FALSE){
    data_reddit[x,13] <- NA
  }
}

for (x in 1:nrow(data_reddit)) {
  if(nzchar(data_reddit[x,14]) == FALSE){
    data_reddit[x,14] <- NA
  }
}

for (x in 1:nrow(data_reddit)) {
  if(nzchar(data_reddit[x,15]) == FALSE){
    data_reddit[x,15] <- NA
  }
}

reddit_Cleaned_Con$drop()

reddit_Cleaned_Con$insert(data_reddit)

print("Reddit cleaning done")
