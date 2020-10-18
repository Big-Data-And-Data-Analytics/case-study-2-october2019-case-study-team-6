library(stringr)
library(jsonlite)
library(tidyverse)
library(xml2)

# MongoDB connection
twitter_post_Data_Collection <- mongo(collection = "Twitter_Post_Cleaned", db = "02_NationalIdentity_Cleaned", url = "mongodb://localhost")

twitter_data <- twitter_post_Data_Collection$find("{}")

options("scipen"=100, "digits"=4)

Twitter_Extract_Hashtags_Post_Data <- function(){
  
  cursorData <- twitter_data
  cursorData['source_Type'] <- "Twitter"
  cursorData['TagList'] <- "@"
  cursorData['onlyText'] <- "OT"
  cursorData['hashtags1'] <- "#"
  
  twitter_post_Data1 <- cursorData %>% select(16,17,22,12,13,6,15,2,19,18,21,23,24)
  
  temp <- cursorData %>% select(17,1,3,4,5,9,10,11,12,14,21,8,7)
  Twitter_Post_Data_Uncommon <<- unique(temp)
  
  #second
  #twitter_post_Data1 <- cursorData %>% select(2,17,4,5,12,3,1,15)
  twitter_post_Data <<- unique(twitter_post_Data1) 
  
  #first
  #twitter_post_Data <<- cursorData %>% select(1,16,3,4,11,17,2,14)
  #twitter_post_Data['hashtags1'] <- "$"
  
  total <- nrow(twitter_post_Data)
  print("Extracting Hashtags, Taglist and Text from Twitter Post Documents is in progress...")
  pb <- txtProgressBar(min = 0, max = total, style = 3)
  
  for (j in 1:nrow(twitter_post_Data)) {
    
    setTxtProgressBar(pb,j)
    replace_Hash_with_space <- str_replace(twitter_post_Data$text[j],"#", " #")
    str <- str_replace(replace_Hash_with_space,"@", " @")
    
    if(is.na(str)){
      twitter_post_Data$hashtags1[j] <<- NA
      twitter_post_Data$TagList[j] <<- NA
      twitter_post_Data$onlyText[j] <<- NA
      next
    }
    
    hashtag_List <- ""
    Tag_List <- ""
    onlyText <- ""
    
    #SPLIT THE TEXT INTO TOKENS
    
    result <- unlist(strsplit(str,"[[:space:]]"))
    
    #result <- unlist(strsplit(str," "))
    
    
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
        twitter_post_Data$hashtags1[j] <- hashtag_List
        twitter_post_Data$TagList[j] <- Tag_List
        twitter_post_Data$onlyText[j] <- onlyText
        onlyText <- ""
        hashtag_List <- ""
        Tag_List <- ""
      }
      
    }

    #CHECKS THE SPLITS IF ARE USERNAME TAGS AND ADDS THEM TO LIST OF PERSON/PLAYER USERNAME TAGGED ON THE POST
    
    
    #CREATES TEXT OF POST WITHOUT THE HASHTAGS
    
    
  }
  
  colnames(twitter_post_Data)
  Twitter_Post_Data_Common <<- twitter_post_Data
  
}

Twitter_Extract_Hashtags_Post_Data()
