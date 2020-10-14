library(stringr)
library(jsonlite)
library(tidyverse)

# Database connnections
post_Collection <- mongo(collection = "Instagram_Post_Cleaned", db = "02_NationalIdentity_Cleaned", url = "mongodb://localhost")
comment_Collection <- mongo(collection = "Instagram_Comment_Cleaned", db = "02_NationalIdentity_Cleaned", url = "mongodb://localhost")

# Functions
Instagram_Extract_Hashtags_Instagram_Post_Data <- function(){
  
  post_Documents <- unique(post_Collection$find(query = "{}"))
  
  # Rename the column to postURL from post_url
  names(post_Documents)[names(post_Documents)=="post_url"] <- "postURL"
  
  # Reorder the postURL Column to match the mappingt to the existing code
  
  post_Documents <- post_Documents[,c(11,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16)]

  cursorData <- post_Documents
  
  cursorData['source_Type'] <- "Instagram"
  cursorData['hashtags'] <- "#"
  cursorData['TagList'] <- "@"
  cursorData['onlyText'] <- "OT"
  
  temp <- cursorData %>% select(1,6,7,8,9,10,16,11,13,14)
  instagram_postData_uncommon_data <<- unique(temp)
  
  Instagram_Post_Data1 <- cursorData %>% select(2,17,15,3,5,4,18,12,1,19,20)

  Instagram_Post_Data <<- Instagram_Post_Data1[!duplicated(Instagram_Post_Data1$post_Id),]
  
  total <- nrow(Instagram_Post_Data)
  
  print("Extracting Hashtags, Taglist and Text from Instagram Post Documents is in progress...")
  pb <- txtProgressBar(min = 0, max = total, style = 3)
  
  for (j in 1:nrow(Instagram_Post_Data)) {
    
    setTxtProgressBar(pb, j)
    
    replace_Hash_with_space <- gsub("#", " #", Instagram_Post_Data$post_Caption[j])    
    str <- str_replace(replace_Hash_with_space,"@", " @")
    str <- trimws(str, which = c("both", "left", "right"))
    
    if(str == ""){
      next
    }
    
    if(is.na(str)){
      Instagram_Post_Data$hashtags[j] <<- NA
      Instagram_Post_Data$TagList[j] <<- NA
      Instagram_Post_Data$onlyText[j] <<- NA
      next
    }
    
    hashtag_List <- ""
    Tag_List <- ""
    onlyText <- ""

    #SPLIT THE TEXT INTO TOKENS
    result <- unlist(strsplit(str," "))
    
    for (i in 1:length(result)) {
      
      x <<-  c(result[i])
      #print(x)
      
      #CHECKS IF IT IS A HASHTAGS AND ADDS TO HASHTAGS LIST
      
      # Extract hashtags
      if(substr(x,0,1)=='#'){
        if(is.na(hashtag_List)){
          hashtag_List <- paste(hashtag_List,"",sep="")
        }else{
          hashtag_List <- paste(hashtag_List,x,sep=" | ")  
        }
      }else{
      # Extract tag list
        if(substr(x,0,1)=='@'){
          Tag_List <- paste(Tag_List,x,sep=" | ")  
        }else{
          Tag_List <- paste(Tag_List,"",sep="")  
        }
      }
      # Extract only text
      if(substr(x,0,1)!='#'){
        if(is.na(onlyText)){
          onlyText <- paste(onlyText,"",sep="")
        }else{
          onlyText <- paste(onlyText,x,sep=" ")
          
        }
      }
      
      if(i == length(result)){
        
        Instagram_Post_Data$hashtags[j] <- hashtag_List
        Instagram_Post_Data$TagList[j] <- Tag_List
        Instagram_Post_Data$onlyText[j] <- onlyText
        onlyText <- ""
        hashtag_List <- ""
        Tag_List <- ""
        
      }
    }
  }
  
  close(pb)
  Instagram_Post_Data$post_Caption <- gsub("[\r\n]", "", Instagram_Post_Data$post_Caption)
  
  Instagram_Post_Common_Data <<- Instagram_Post_Data
  Instagram_Post_Uncommon_Data <<- instagram_postData_uncommon_data
}

Instagram_Extract_Hashtags_comment_Data <- function(){
  
  
  comment_Documents <- unique(comment_Collection$find(query = "{}"))
  
  comment_Documents <- comment_Documents %>% select(-19)

  cursorData <- comment_Documents
  cursorData['hashtags'] <- "#"
  cursorData['TagList'] <- "@"
  cursorData['onlyText'] <- "OT"
  cursorData['source_Type'] <- "Instagram"
  
  temp <- cursorData %>% select(1,2,6,7,5,10)
  instagram_comment_uncommon_data <- temp[!duplicated(temp$comment_id),]
  
  comment_Data <<- cursorData %>% select(1,22,2,8,3,9,4,19,20,21)
  comment_Data <<-  comment_Data[!duplicated(comment_Data$comment_id),]
  
  total <- nrow(comment_Data)
  
  print("Extracting Hashtags, Taglist and Text from Instagram Comment Documents is in progress...")
  pb <- txtProgressBar(min = 0, max = total, style = 3)
  
  for (j in 1:nrow(comment_Data)) {
    
    setTxtProgressBar(pb, j)
    
    replace_Hash_with_space <- str_replace(comment_Data$comment_Text[j],"#", " #")
    str <- str_replace(replace_Hash_with_space,"@", " @")
    str <- trimws(str, which = c("both", "left", "right"))
    
    if(str == ""){
      next
    }
    
    if(is.na(str))
    {
      comment_Data$hashtags[j] <<- NA
      comment_Data$TagList[j] <<- NA
      comment_Data$onlyText[j] <<- NA
      next
    }
    
    hashtag_List <- ""
    Tag_List <- ""
    onlyText <- ""

    #SPLIT THE TEXT INTO TOKENS
    result <- unlist(strsplit(str,"[[:space:]]"))

    for (i in 1:length(result)) {
      
      
      x <<-  c(result[i])
      
      #CHECKS IF IT IS A HASHTAGS AND ADDS TO HASHTAGS LIST
      
      if(substr(x,0,1)=='#'){
        if(is.na(hashtag_List)){
          hashtag_List <- paste(hashtag_List,"",sep="")
        }else{
          hashtag_List <- paste(hashtag_List,x,sep=" | ")  
        }
      }else{
        if(substr(x,0,1)=='@'){
          Tag_List <- paste(Tag_List,x,sep=" | ")  
        }else{
          Tag_List <- paste(Tag_List,"",sep="")  
        }
      }
      if(substr(x,0,1)!='#'){
        
        if(is.na(onlyText)){
          onlyText <- paste(onlyText,"",sep="")
        }else{
          onlyText <- paste(onlyText,x,sep=" ")
          
        }
      }
      
      if(i==length(result)){
        comment_Data$hashtags[j] <- hashtag_List
        comment_Data$TagList[j] <- Tag_List
        comment_Data$onlyText[j] <- onlyText
        onlyText <- ""
        hashtag_List <- ""
        Tag_List <- ""
      }
      
    }

    #CHECKS THE SPLITS IF ARE USERNAME TAGS AND ADDS THEM TO LIST OF PERSON/PLAYER USERNAME TAGGED ON THE POST
    
    #CREATES TEXT OF POST WITHOUT THE HASHTAGS
    
  }
  
  close(pb)
  comment_Data1 <- comment_Data[!(is.na(comment_Data$comment_id) | comment_Data$comment_id=="NULL" |  comment_Data$comment_id==""), ]
  
  Instagram_Comment_Common_Data <<- data.frame(comment_Data1)
  Instagram_Comment_Uncommon_Data <<- instagram_comment_uncommon_data
}

Instagram_Extract_Hashtags_sub_comment_Data <- function(){
  
  comment_Documents <- unique(comment_Collection$find(query = "{}"))
  comment_Documents <- comment_Documents %>% select(-19)
  colnames(comment_Documents)
  
  cursorData <- comment_Documents 
  
  cursorData['source_Type'] <- "Instagram"
  cursorData['hashtags'] <- "#"
  cursorData['TagList'] <- "@"
  cursorData['onlyText'] <- "OT"
  sub_comment_Data <<- cursorData %>% select(2,19,11,17,12,18,13,20,21,22)
  temp <- cursorData %>% select(2,14,15,16)
  
  instagram_subcomment_uncommon_data <- unique(temp)
  
  total <- nrow(cursorData)
  print("Extracting Hashtags, Taglist and Text from Instagram Sub Comment Documents is in progress...")
  pb <- txtProgressBar(min = 0, max = total, style = 3)
  
  for (j in 1:nrow(sub_comment_Data)) {
    
    setTxtProgressBar(pb, j)
    replace_Hash_with_space <- str_replace(sub_comment_Data$comment_comment_text[j],"#", " #")
    str <- str_replace(replace_Hash_with_space,"@", " @")
    
    if(is.na(str))
    {
      sub_comment_Data$hashtags[j] <<- NA
      sub_comment_Data$TagList[j] <<- NA
      sub_comment_Data$onlyText[j] <<- NA
      next
    }
    
    hashtag_List <- ""
    Tag_List <- ""
    onlyText <- ""
    
    #SPLIT THE TEXT INTO TOKENS
    result <- unlist(strsplit(str,"[[:space:]]"))

    for (i in 1:length(result)) {
      
      
      x <-  c(result[i])
      
      #CHECKS IF IT IS A HASHTAGS AND ADDS TO HASHTAGS LIST
      
      if(substr(x,0,1)=='#'){
        if(is.na(hashtag_List)){
          hashtag_List <- paste(hashtag_List,"",sep="")
        }else{
          hashtag_List <- paste(hashtag_List,x,sep=" | ")  
        }
      }else{
        if(substr(x,0,1)=='@'){
          Tag_List <- paste(Tag_List,x,sep=" | ")  
        }else{
          Tag_List <- paste(Tag_List,"",sep="")  
        }
      }
      if(substr(x,0,1)!='#'){
        
        if(is.na(onlyText)){
          onlyText <- paste(onlyText,"",sep="")
        }else{
          onlyText <- paste(onlyText,x,sep=" ")
          
        }
      }
      
      if(i==length(result)){
        sub_comment_Data$hashtags[j] <<- hashtag_List
        sub_comment_Data$TagList[j] <<- Tag_List
        sub_comment_Data$onlyText[j] <<- onlyText
        onlyText <- ""
        hashtag_List <- ""
        Tag_List <- ""
      }
      
    }

    #CHECKS THE SPLITS IF ARE USERNAME TAGS AND ADDS THEM TO LIST OF PERSON/PLAYER USERNAME TAGGED ON THE POST

    #CREATES TEXT OF POST WITHOUT THE HASHTAGS
  }
  
  close(pb)
  
  sub_comment_Data1 <- sub_comment_Data[!(is.na(sub_comment_Data$comment_comment_owner_username) | sub_comment_Data$comment_comment_owner_username=="NULL" |  sub_comment_Data$comment_comment_owner_username==""), ]
  sub_comment_Data <<- data.frame(sub_comment_Data1)
  
  Instagram_Subcomment_Common_Data <<- sub_comment_Data[!duplicated(sub_comment_Data$comment_comment_id),]
  Instagram_Subcomment_Uncommon_Data <<- instagram_subcomment_uncommon_data
}


# Run functions
Instagram_Extract_Hashtags_Instagram_Post_Data()
Instagram_Extract_Hashtags_comment_Data()
Instagram_Extract_Hashtags_sub_comment_Data()

# Code green, keep the environment clean!
#rm(post_Collection, comment_Collection)
#rm(Instagram_Post_Common_Data, Instagram_Post_Uncommon_Data, Instagram_Comment_Common_Data, Instagram_Comment_Uncommon_Data, Instagram_Subcomment_Common_Data, Instagram_Subcomment_Uncommon_Data, x)
#rm(Instagram_Extract_Hashtags_Instagram_Post_Data, Instagram_Extract_Hashtags_comment_Data, Instagram_Extract_Hashtags_sub_comment_Data)
