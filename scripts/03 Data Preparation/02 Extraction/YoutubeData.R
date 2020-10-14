library(stringr)
library(jsonlite)
library(dplyr)
library(mongolite)

# MongoDB connection
youTube_Video_Stats_Con <<- mongo(collection = "youTube_Video_Stats_Cleaned", db = "02_NationalIdentity_Cleaned", url = "mongodb://localhost")
youTube_Video_Comments_Con <<- mongo(collection = "youTube_Video_Comments_Cleaned", db = "02_NationalIdentity_Cleaned", url = "mongodb://localhost")
youTube_Videos_Con <<- mongo(collection = "youTube_Videos_Cleaned", db = "02_NationalIdentity_Cleaned", url = "mongodb://localhost")

# Load data
youTube_Video_Stats_Data <<- youTube_Video_Stats_Con$find(query = "{}")
youTube_Video_Comments_Data <<- youTube_Video_Comments_Con$find(query = "{}")
youTube_Videos_Data <<- youTube_Videos_Con$find(query = "{}")

# Splits youTube data into different source types "post", "comment" and "subcomment"
dataCollector <- function() {
  
  youTube_Videos_Data <- left_join(youTube_Videos_Data, youTube_Video_Stats_Data)
  
  youTube_Common_Data_Post <<- youTube_Videos_Data %>% 
    mutate(Id = video_id, username = channel, text = description, likes = viewCount, timestamp = publishedAt, owner_id = channel, source_Type = "YouTube",
           data_Type = "post", url = paste("https://www.youtube.com/watch?v=", video_id, sep = "")) %>% 
    select(Id, source_Type, username, text, likes, timestamp, owner_id, url, data_Type)
  
  youTube_Common_Data_Comment <<- youTube_Video_Comments_Data %>% 
    filter(is.na(parentId)) %>% 
    mutate(source_Type = "YouTube", data_Type = "comment", Id = videoId, Comment_Id = id, Comment_Owner = authorDisplayName, Comment = textOriginal,
           Comment_Likes = likeCount, Comment_Time_Posted = publishedAt) %>% 
    select(Id, source_Type, Comment_Id, Comment_Owner, Comment, Comment_Likes, Comment_Time_Posted, data_Type)
  
  youTube_Common_Data_SubComment <<- youTube_Video_Comments_Data %>% 
    filter(!is.na(parentId)) %>% 
    mutate(source_Type = "YouTube", data_Type = "subcomment", Id = parentId, Sub_Comment_Id = id, Sub_Comment_Owner = authorDisplayName, Sub_Comment = textOriginal,
           Sub_Comment_Likes = likeCount, Sub_Comment_Time_Posted = publishedAt) %>% 
    select(Id, source_Type, Sub_Comment_Id, Sub_Comment_Owner, Sub_Comment, Sub_Comment_Likes, Sub_Comment_Time_Posted, data_Type)
  
}

# Function to extract hashtags, "@-tags" and the "onlyText" from data_Type = post
youTube_Post_Hashtag_Extractor <- function(){
  
  youTube_Common_Data_Post$hashtags <- ""
  youTube_Common_Data_Post$TagList <- ""
  youTube_Common_Data_Post$onlyText <- ""
  
  total <- nrow(youTube_Common_Data_Post)
  
  print("Extracting Hashtags, Taglist and Text from youtube Post Documents is in progress...")
  pb <- txtProgressBar(min = 0, max = total, style = 3)
  
  for (j in 1:nrow(youTube_Common_Data_Post)) {
    
    replace_Hash_with_space <- str_replace(youTube_Common_Data_Post$text[j], "#", " #")
    str <- str_replace(replace_Hash_with_space, "@", " @")
    
    if(is.na(str)) {
      youTube_Common_Data_Post$hashtags[j] <- NA
      youTube_Common_Data_Post$TagList[j] <- NA
      youTube_Common_Data_Post$onlyText[j] <- NA
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
        youTube_Common_Data_Post$hashtags[j] <- hashtag_List
        youTube_Common_Data_Post$TagList[j] <- Tag_List
        youTube_Common_Data_Post$onlyText[j] <- onlyText
        onlyText <- ""
        hashtag_List <- ""
        Tag_List <- ""
        
      }
      
    }

  }
  youTube_Common_Data_Post <<- youTube_Common_Data_Post
  close(pb)
  
}

# Function to extract hashtags, "@-tags" and the "onlyText" from data_Type = comment
youTube_Comment_Hashtag_Extractor <- function(){
  
  youTube_Common_Data_Comment$hashtags <- ""
  youTube_Common_Data_Comment$TagList <- ""
  youTube_Common_Data_Comment$onlyText <- ""
  
  total <- nrow(youTube_Common_Data_Comment)
  
  print("Extracting Hashtags, Taglist and Text from youtube Comment Documents is in progress...")
  pb <- txtProgressBar(min = 0, max = total, style = 3)
  
  for (j in 1:nrow(youTube_Common_Data_Comment)) {
    
    replace_Hash_with_space <- str_replace(youTube_Common_Data_Comment$Comment[j], "#", " #")
    str <- str_replace(replace_Hash_with_space, "@", " @")
    
    if(is.na(str)) {
      youTube_Common_Data_Comment$hashtags[j] <- NA
      youTube_Common_Data_Comment$TagList[j] <- NA
      youTube_Common_Data_Comment$onlyText[j] <- NA
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
        youTube_Common_Data_Comment$hashtags[j] <- hashtag_List
        youTube_Common_Data_Comment$TagList[j] <- Tag_List
        youTube_Common_Data_Comment$onlyText[j] <- onlyText
        onlyText <- ""
        hashtag_List <- ""
        Tag_List <- ""
        
      }
      
    }
    
  }
  
  youTube_Common_Data_Comment <<- youTube_Common_Data_Comment
  close(pb)
  
}

# Function to extract hashtags, "@-tags" and the "onlyText" from data_Type = subcomment
youTube_SubComment_Hashtag_Extractor <- function(){
  
  youTube_Common_Data_SubComment$hashtags <- ""
  youTube_Common_Data_SubComment$TagList <- ""
  youTube_Common_Data_SubComment$onlyText <- ""
  
  total <- nrow(youTube_Common_Data_SubComment)
  
  print("Extracting Hashtags, Taglist and Text from Youtube Subcomments Documents is in progress...")
  pb <- txtProgressBar(min = 0, max = total, style = 3)
  
  for (j in 1:nrow(youTube_Common_Data_SubComment)) {
    
    replace_Hash_with_space <- str_replace(youTube_Common_Data_SubComment$Sub_Comment[j], "#", " #")
    str <- str_replace(replace_Hash_with_space, "@", " @")
    
    if(is.na(str)) {
      youTube_Common_Data_SubComment$hashtags[j] <- NA
      youTube_Common_Data_SubComment$TagList[j] <- NA
      youTube_Common_Data_SubComment$onlyText[j] <- NA
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
        youTube_Common_Data_SubComment$hashtags[j] <- hashtag_List
        youTube_Common_Data_SubComment$TagList[j] <- Tag_List
        youTube_Common_Data_SubComment$onlyText[j] <- onlyText
        onlyText <- ""
        hashtag_List <- ""
        Tag_List <- ""
        
      }
      
    }
    
  }
  
  youTube_Common_Data_SubComment <<- youTube_Common_Data_SubComment
  close(pb)
  
}


# Run the functions
dataCollector()

youTube_Post_Hashtag_Extractor()
youTube_Comment_Hashtag_Extractor()
youTube_SubComment_Hashtag_Extractor()

# Keep the environment clean, code green!
rm(youTube_Video_Comments_Data, youTube_Video_Stats_Data, youTube_Videos_Data)
rm(youTube_Video_Comments_Con, youTube_Video_Stats_Con, youTube_Videos_Con)
