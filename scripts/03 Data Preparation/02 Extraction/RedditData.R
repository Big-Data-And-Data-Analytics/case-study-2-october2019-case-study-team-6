################# POST ################# 
library(dplyr)
library(mongolite)
library(stringr)

# Connect to MongoDB and query data
reddit_Con <- mongo(collection = "reddit_Data_Cleaned", db = "02_NationalIdentity_Cleaned")
data_reddit <- reddit_Con$find(query = "{}")

#### PREPARE REDDIT COMMON POST DATA ####
post_reddit <- data_reddit %>% 
  mutate(source_Type = "Reddit", source_Type = "Reddit", data_Type = "post") %>% 
  distinct(Id = post_id,
           source_Type = source_Type,
           username = author,
           text = post_text,
           likes = post_score,
           timestamp = post_date,
           owner_id = author,
           url = URL,
           data_Type = data_Type)

#### PREPARE REDDIT COMMON COMMENT DATA ####
comment_reddit <- data_reddit %>% 
  mutate(source_Type = "Reddit", data_Type = "comment", comment = ifelse(comment == "", NA, comment)) %>% 
  filter(!grepl("_", structure)) %>%
  distinct(Id = id,
           source_Type = source_Type,
           Comment_Id = structure,
           Comment_Owner = user,
           Comment = comment,
           Comment_Likes = comment_score,
           Comment_Time_Posted = post_date,
           data_Type = data_Type)

#### PREPARE REDDIT COMMON SUBCOMMENT DATA ####
sub_comment_reddit <- data_reddit %>% 
  mutate(source_Type = "Reddit", data_Type = "sub_comment", comment = ifelse(comment == "", NA, comment)) %>% 
  filter(grepl("_", structure)) %>% 
  distinct(Id = id,
           source_Type = source_Type,
           Sub_Comment_Id = structure,
           Sub_Comment_Owner = user,
           Sub_Comment = comment, 
           Sub_Comment_Likes = comment_score, 
           Sub_Comment_Time_Posted = post_date, 
           data_Type = data_Type)

#### PREPARE REDDIT UNCOMMON DATA ####

reddit_uncommon_data <- data_reddit %>% 
  mutate(source_Type = "Reddit", data_Type = "comment") %>% 
  filter(!grepl("_", structure)) %>%
  distinct(num_comments,
           subreddit,
           upvote_prop,
           title,
           link,
           domain)


# Function to extract hashtags, "@-tags" and the "onlyText" from data_Type = post
reddit_Post_Hashtag_Extractor <- function(){
  
  post_reddit$hashtags <- ""
  post_reddit$TagList <- ""
  post_reddit$onlyText <- ""
  
  total <- nrow(post_reddit)
  
  print("Extracting Hashtags, Taglist and Text from reddit Post Documents is in progress...")
  pb <- txtProgressBar(min = 0, max = total, style = 3)
  
  for (j in 1:nrow(post_reddit)) {

    replace_Hash_with_space <- str_replace(post_reddit$text[j], "#", " #")
    str <- str_replace(replace_Hash_with_space, "@", " @")
    
    if(is.na(str)) {
      post_reddit$hashtags[j] <- NA
      post_reddit$TagList[j] <- NA
      post_reddit$onlyText[j] <- NA
      next
    }
    
    hashtag_List <- ""
    Tag_List <- ""
    onlyText <- ""
    
    #SPLIT THE TEXT INTO TOKENS
    result <- unlist(strsplit(str,"[[:space:]]"))
    
    for (i in 1:length(result)) {
      
      setTxtProgressBar(pb, j)
      
      x <-  c(result[i])
      
      # CHECK IF WORD IS A HASHTAG AND ADD TO HASHTAGS LIST
      if (substr(x, 0, 1) == '#'){
        if (is.na(hashtag_List)) {
          hashtag_List <- paste(hashtag_List, "", sep = "")
          
        } else {
          hashtag_List <- paste(hashtag_List, x, sep=" | ")  
          
        }
        
      } else {
        
        if (substr(x, 0, 1) == '@') {
          Tag_List <- paste(Tag_List, x, sep=" | ") 
          
        } else {
          Tag_List <- paste(Tag_List, "", sep="")
          
        }
      }
      
      if (substr(x, 0, 1)!='#') {
        if(is.na(onlyText)){
          onlyText <- paste(onlyText, "", sep="")
          
        } else {
          onlyText <- paste(onlyText, x, sep=" ")
          
        }
      }
      
      if (i == length(result)){
        post_reddit$hashtags[j] <<- hashtag_List
        post_reddit$TagList[j] <<- Tag_List
        post_reddit$onlyText[j] <<- onlyText
        onlyText <- ""
        hashtag_List <- ""
        Tag_List <- ""
        
      }
      
    }
    
  }
  
  close(pb)
  
}

# Function to extract hashtags, "@-tags" and the "onlyText" from data_Type = comment
reddit_Comment_Hashtag_Extractor <- function(){
  
  comment_reddit$hashtags <- ""
  comment_reddit$TagList <- ""
  comment_reddit$onlyText <- ""
  
  total <- nrow(comment_reddit)
  
  print("Extracting Hashtags, Taglist and Text from reddit Comment Documents is in progress...")
  pb <- txtProgressBar(min = 0, max = total, style = 3)
  
  for (j in 1:nrow(comment_reddit)) {
    
    replace_Hash_with_space <- str_replace(comment_reddit$Comment[j], "#", " #")
    str <- str_replace(replace_Hash_with_space, "@", " @")
    
    if(is.na(str)) {
      comment_reddit$hashtags[j] <- NA
      comment_reddit$TagList[j] <- NA
      comment_reddit$onlyText[j] <- NA
      next
    }
    
    hashtag_List <- ""
    Tag_List <- ""
    onlyText <- ""
    
    #SPLIT THE TEXT INTO TOKENS
    result <- unlist(strsplit(str,"[[:space:]]"))
    
    for (i in 1:length(result)) {
      
      setTxtProgressBar(pb, j)
      
      x <-  c(result[i])
      
      # CHECK IF WORD IS A HASHTAG AND ADD TO HASHTAGS LIST
      if (substr(x, 0, 1) == '#'){
        if (is.na(hashtag_List)) {
          hashtag_List <- paste(hashtag_List, "", sep = "")
          
        } else {
          hashtag_List <- paste(hashtag_List, x, sep=" | ")  
          
        }
        
      } else {
        
        if (substr(x, 0, 1) == '@') {
          Tag_List <- paste(Tag_List, x, sep=" | ") 
          
        } else {
          Tag_List <- paste(Tag_List, "", sep="")
          
        }
      }
      
      if (substr(x, 0, 1)!='#') {
        if(is.na(onlyText)){
          onlyText <- paste(onlyText, "", sep="")
          
        } else {
          onlyText <- paste(onlyText, x, sep=" ")
          
        }
      }
      
      if (i == length(result)){
        comment_reddit$hashtags[j] <<- hashtag_List
        comment_reddit$TagList[j] <<- Tag_List
        comment_reddit$onlyText[j] <<- onlyText
        onlyText <- ""
        hashtag_List <- ""
        Tag_List <- ""
        
      }
      
    }
    
  }
  
  close(pb)
  
}

# Function to extract hashtags, "@-tags" and the "onlyText" from data_Type = subcomment
reddit_SubComment_Hashtag_Extractor <- function(){
  
  sub_comment_reddit$hashtags <- ""
  sub_comment_reddit$TagList <- ""
  sub_comment_reddit$onlyText <- ""
  
  total <- nrow(sub_comment_reddit)
  
  print("Extracting Hashtags, Taglist and Text from Reddit Subcomments Documents is in progress...")
  pb <- txtProgressBar(min = 0, max = total, style = 3)
  
  for (j in 1:nrow(sub_comment_reddit)) {
    
    replace_Hash_with_space <- str_replace(sub_comment_reddit$Sub_Comment[j], "#", " #")
    str <- str_replace(replace_Hash_with_space, "@", " @")
    
    if (is.na(str)) {
      sub_comment_reddit$hashtags[j] <- NA
      sub_comment_reddit$TagList[j] <- NA
      sub_comment_reddit$onlyText[j] <- NA
      next
    }
    
    hashtag_List <- ""
    Tag_List <- ""
    onlyText <- ""
    
    #SPLIT THE TEXT INTO TOKENS
    result <- unlist(strsplit(str,"[[:space:]]"))
    
    for (i in 1:length(result)) {
      
      setTxtProgressBar(pb, j)
      
      x <-  c(result[i])
      
      # CHECK IF WORD IS A HASHTAG AND ADD TO HASHTAGS LIST
      if (substr(x, 0, 1) == '#'){
        if (is.na(hashtag_List)) {
          hashtag_List <- paste(hashtag_List, "", sep = "")
          
        } else {
          hashtag_List <- paste(hashtag_List, x, sep=" | ")  
          
        }
        
      } else {
        
        if (substr(x, 0, 1) == '@') {
          Tag_List <- paste(Tag_List, x, sep=" | ") 
          
        } else {
          Tag_List <- paste(Tag_List, "", sep="")
          
        }
      }
      
      if (substr(x, 0, 1)!='#') {
        if(is.na(onlyText)){
          onlyText <- paste(onlyText, "", sep="")
          
        } else {
          onlyText <- paste(onlyText, x, sep=" ")
          
        }
      }
      
      if (i == length(result)){
        sub_comment_reddit$hashtags[j] <<- hashtag_List
        sub_comment_reddit$TagList[j] <<- Tag_List
        sub_comment_reddit$onlyText[j] <<- onlyText
        onlyText <- ""
        hashtag_List <- ""
        Tag_List <- ""
        
      }
      
    }
    
  }
  
  close(pb)
  
}


# Run functions
reddit_Post_Hashtag_Extractor()
reddit_Comment_Hashtag_Extractor()
reddit_SubComment_Hashtag_Extractor()

# Keep the environment clean, code green!
rm(reddit_Con, data_reddit)
