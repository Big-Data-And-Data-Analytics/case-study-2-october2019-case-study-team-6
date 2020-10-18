#Script for the sentiment analysis for each comment_column for each data source
library(tidyverse)
library(tidytext)
library(mongolite)

# Establish connection to MongoDB and query data
common_post_Collection <- mongo(collection = "common_post_Combined", db = "03_NationalIdentity_Combined", url = "mongodb://localhost")
common_comment_Collection <- mongo(collection = "common_comment_Combined", db = "03_NationalIdentity_Combined", url = "mongodb://localhost")
common_subcomment_Collection <- mongo(collection = "common_subcomment_Combined", db = "03_NationalIdentity_Combined", url = "mongodb://localhost")

post <- common_post_Collection$find(query ="{}")
comment <- common_comment_Collection$find(query ="{}")
subcomment <- common_subcomment_Collection$find(query ="{}")

# Establish connection to future collections in MongoDB
sentiment_post_Collection <- mongo(collection = "sentiment_post_Collection", db = "04_NationalIdentity_Sentiment", url = "mongodb://localhost")
sentiment_comment_Collection <- mongo(collection = "sentiment_comment_Collection", db = "04_NationalIdentity_Sentiment", url = "mongodb://localhost")
sentiment_subcomment_Collection <- mongo(collection = "sentiment_subcomment_Collection", db = "04_NationalIdentity_Sentiment", url = "mongodb://localhost")

sentiment_post_Collection$drop()
sentiment_comment_Collection$drop()
sentiment_subcomment_Collection$drop()

# Select dictionaries for sentiments
bing_Dictionary <- get_sentiments("bing") # Categorical
nrc_Dictionary <- get_sentiments("nrc") # Categorical
loughran_Dictionary <- get_sentiments("loughran") # Categorical
#afinn_Dictionary <- get_sentiments("afinn") # Numerical

sentiment_Dictionary <- rbind(bing_Dictionary, nrc_Dictionary, loughran_Dictionary)
sentiment_Dictionary <- sentiment_Dictionary %>% distinct_all()

# Sentiment analysis functions
sentiment_analysis_postData <- function() {

  data_sentiment <- post

  sentimentFrame <- data_sentiment %>% 
    distinct(onlyText) %>% 
    group_by(onlyText) %>% 
    mutate(linenumber = row_number(), post_Caption_Text = onlyText) %>% 
    ungroup() %>% 
    unnest_tokens(word, post_Caption_Text)
  
  sentimentFrame <- sentimentFrame %>% 
    inner_join(sentiment_Dictionary) %>% 
    group_by(onlyText, sentiment) %>% 
    summarise(Count = n()) %>% 
    mutate(sentiments = sentiment) %>% 
    filter(Count == max(Count)) %>% 
    select(onlyText, sentiments)
  
  data_sentiment_post <- left_join(data_sentiment, sentimentFrame, by = "onlyText")
  
  data_sentiment_test_noDupl <- data_sentiment_post %>% 
    group_by(Id         ,            source_Type    ,        username              ,
             text       ,            likes          ,        timestamp             ,
             hashtags   ,            owner_id       ,        url                   ,
             TagList    ,            onlyText         ) %>% 
    summarise(Count = n()) %>%
    filter(Count == 1)

  test <- left_join(data_sentiment_test_noDupl, data_sentiment_post)
  
  data_sentiment_test_onlyDupl <- data_sentiment_post %>% 
    group_by(Id         ,            source_Type    ,        username              ,
             text       ,            likes          ,        timestamp             ,
             hashtags   ,            owner_id       ,        url                   ,
             TagList    ,            onlyText        ) %>%
    summarise(Count = n()) %>%
    filter(Count > 1) %>% 
    mutate(sentiment = "positve")
  
  test2 <- rbind(test, data_sentiment_test_onlyDupl)
  
  test2test <- test2 %>% 
    mutate(sentiment = case_when(
      is.na(onlyText) ~ sentiment
    )) %>% 
    select(-(length(test2)))
  
  #sort
  sorted_data <- test2test[order(test2test$username),]
  
  #remove duplicates on sorted data
  sentiments_Post <<- sorted_data[!duplicated(sorted_data$Id,fromLast = TRUE),]
  sentiment_post_Collection$insert(sentiments_Post)
  
  print("Sentiments for posts done.")
}
sentiment_analysis_commentData <- function(){

  data_sentiment <- comment
  
  sentimentFrame <- data_sentiment %>% 
    distinct(onlyText) %>% 
    group_by(onlyText) %>% 
    mutate(linenumber = row_number(), post_Caption_Text = onlyText) %>% 
    ungroup() %>% 
    unnest_tokens(word, post_Caption_Text)
  
  sentimentFrame <- sentimentFrame %>% 
    inner_join(sentiment_Dictionary) %>% 
    group_by(onlyText, sentiment) %>% 
    summarise(Count = n()) %>% 
    mutate(sentiments = sentiment) %>% 
    filter(Count == max(Count)) %>% 
    select(onlyText, sentiments)
  
  data_sentiment_post <- left_join(data_sentiment, sentimentFrame, by = "onlyText")
  
  data_sentiment_test_noDupl <- data_sentiment_post %>% 
    group_by( Id                ,  source_Type         ,Comment_Id          ,Comment_Owner       ,Comment            ,
              Comment_Likes     ,  Comment_Time_Posted ,hashtags            ,TagList             ,onlyText ) %>% 
    summarise(Count = n()) %>%
    filter(Count == 1)

  test <- left_join(data_sentiment_test_noDupl, data_sentiment_post)
  
  data_sentiment_test_onlyDupl <- data_sentiment_post %>% 
    group_by( Id                ,  source_Type         ,Comment_Id          ,Comment_Owner       ,Comment            ,
              Comment_Likes     ,  Comment_Time_Posted ,hashtags            ,TagList             ,onlyText   ) %>%
    summarise(Count = n()) %>%
    filter(Count > 1) %>% 
    mutate(sentiment = "positve")
  
  test2 <- rbind(test, data_sentiment_test_onlyDupl)
  
  test2test <- test2 %>% 
    mutate(sentiment = case_when(
      is.na(onlyText) ~ sentiment
    )) %>% 
    select(-(length(test2)))
  
  #sort
  sorted_data <- test2test[order(test2test$Comment_Owner),]
  
  #remove duplicates on sorted data
  sentiments_Comment <<- sorted_data[!duplicated(sorted_data$Comment,fromLast = TRUE),]
  sentiment_comment_Collection$insert(sentiments_Comment)
  
  print("Sentiments for comments done.")
}
sentiment_analysis_subcommentData <- function(){

  data_sentiment <- subcomment

  sentimentFrame <- data_sentiment %>% 
    distinct(onlyText) %>% 
    group_by(onlyText) %>% 
    mutate(linenumber = row_number(), post_Caption_Text = onlyText) %>% 
    ungroup() %>% 
    unnest_tokens(word, post_Caption_Text)
  
  sentimentFrame <- sentimentFrame %>% 
    inner_join(sentiment_Dictionary) %>% 
    group_by(onlyText, sentiment) %>% 
    summarise(Count = n()) %>% 
    mutate(sentiments = sentiment) %>% 
    filter(Count == max(Count)) %>% 
    select(onlyText, sentiments)
  
  data_sentiment_post <- left_join(data_sentiment, sentimentFrame, by = "onlyText")
  
  data_sentiment_test_noDupl <- data_sentiment_post %>% 
    group_by( Id                  ,    source_Type           ,  Sub_Comment_Id          ,Sub_Comment_Owner      ,
              Sub_Comment         ,    Sub_Comment_Likes     ,  Sub_Comment_Time_Posted ,hashtags               ,
              TagList             ,   onlyText                ) %>% 
    summarise(Count = n()) %>%
    filter(Count == 1)
  
  test <- left_join(data_sentiment_test_noDupl, data_sentiment_post)
  
  data_sentiment_test_onlyDupl <- data_sentiment_post %>% 
    group_by( Id                  ,    source_Type           ,  Sub_Comment_Id          ,Sub_Comment_Owner      ,
              Sub_Comment         ,    Sub_Comment_Likes     ,  Sub_Comment_Time_Posted ,hashtags               ,
              TagList             ,   onlyText               ) %>%
    summarise(Count = n()) %>%
    filter(Count > 1) %>% 
    mutate(sentiment = "positve")
  
  test2 <- rbind(test, data_sentiment_test_onlyDupl)
  
  test2test <- test2 %>% 
    mutate(sentiment = case_when(
      is.na(onlyText) ~ sentiment
    )) %>% 
    select(-(length(test2)))
  
  #sort
  sorted_data <- test2test[order(test2test$Sub_Comment_Owner),]
  
  #remove duplicates on sorted data
  sentiments_SubComment <<- sorted_data[!duplicated(sorted_data$Sub_Comment,fromLast = TRUE),]
  sentiment_subcomment_Collection$insert(sentiments_SubComment)
  
  print("Sentiments for subcomments done.")
}

# Run sentiment analysis
sentiment_analysis_postData()
sentiment_analysis_commentData()
sentiment_analysis_subcommentData()
