
# Function to get the post data files of each source into the same shape
renamecolumns_n_combinerows_post <- function(){
  
  #######################################################################################################################################
  #                         CHANGE COLUMN NAMES IN POST_DATA
  #######################################################################################################################################
  #colnames(post_Data) <- c("Id","source_Type","username","text","likes","timestamp","hashtags","owner_id","url","TagList","onlyText")
  #colnames(twitter_post_Data) <- c("Id","source_Type","username","text","likes","timestamp","hashtags","owner_id","url","TagList","onlyText")
  
  # Prepare YouTube Posts
  youTube_Common_Data_Post <- youTube_Common_Data_Post %>%
    select(Id, source_Type, username, text, likes, timestamp, hashtags, owner_id, url, TagList, onlyText, data_Type)
  
  # Prepare Reddit Posts
  post_reddit <- post_reddit %>% 
    select(Id, source_Type, username, text, likes, timestamp, hashtags, owner_id, url, TagList, onlyText, data_Type)
  
  # Prepare Instagram Posts
  Instagram_Post_Common_Data <- Instagram_Post_Common_Data %>% 
    select(Id = post_Id, source_Type, username = owner_username, text = post_Caption, likes = post_Like_Count, timestamp = post_Time_Posted, hashtags, owner_id, url = postURL, TagList, onlyText) %>% 
    mutate(data_Type = "post")
  
  # Prepare Twitter Data
  Twitter_Post_Common_Data <- Twitter_Post_Data_Common %>% 
    select(Id = tweet_id, source_Type, username, text, likes, timestamp = timestamp_epochs, hashtags = hashtags1, owner_id = user_id, url = tweet_url, TagList, onlyText) %>% 
    mutate(data_Type = "post")
  
  # Bind every file together into one
  ## TODO Add Twiter data to the combination
    postData_file <- rbind(
    Instagram_Post_Common_Data,
    Twitter_Post_Common_Data,
    youTube_Common_Data_Post,
    post_reddit)

  rm(youTube_Common_Data_Post, post_reddit, Instagram_Post_Common_Data, Twitter_Post_Common_Data)
  
  return(postData_file)
  
}

# Function to get the comment data files of each source into the same shape
renamecolumns_n_combinerows_comment <- function(){
  
  
  #######################################################################################################################################
  #                         CHANGE COLUMN NAMES IN COMMENT_DATA
  #######################################################################################################################################

  # Prepare YouTube Comments
  youTube_Common_Data_Comment <- youTube_Common_Data_Comment %>%
    select(Id, source_Type, Comment_Id, Comment_Owner, Comment, Comment_Likes, Comment_Time_Posted, hashtags, TagList, onlyText, data_Type)
  
  # Prepare Reddit Comments
  comment_reddit <- comment_reddit %>% 
    select(Id, source_Type, Comment_Id, Comment_Owner, Comment, Comment_Likes, Comment_Time_Posted, hashtags, TagList, onlyText, data_Type)
  
  # Prepare Instagram Comments
  Instagram_Comment_Common_Data <- Instagram_Comment_Common_Data %>% 
    select(Id = post_Id, source_Type, Comment_Id = comment_id, Comment_Owner = comment_owner_username, Comment = comment_Text, Comment_Likes = comment_liked_count, Comment_Time_Posted = comment_Time_Posted, hashtags, TagList, onlyText) %>% 
    mutate(data_Type = "comment")
  
  # Bind every file together into one
  comment_file <- rbind(
    Instagram_Comment_Common_Data,
    youTube_Common_Data_Comment,
    comment_reddit)
  
  rm(youTube_Common_Data_Comment, comment_reddit, Instagram_Comment_Common_Data)
  
  return(comment_file)
}

# Function to get the subcomment data files of each source into the same shape
renamecolumns_n_combinerows_subcomment <- function(){
  #######################################################################################################################################
  #                         CHANGE COLUMN NAMES IN SUBCOMMENT_DATA
  #######################################################################################################################################
  
  # Prepare YouTube SubComments
  youTube_Common_Data_SubComment <- youTube_Common_Data_SubComment %>%
    select(Id, source_Type, Sub_Comment_Id, Sub_Comment_Owner, Sub_Comment, Sub_Comment_Likes, Sub_Comment_Time_Posted, hashtags, TagList, onlyText, data_Type)
  
  # Prepare Reddit SubComments
  sub_comment_reddit <- sub_comment_reddit %>% 
    select(Id, source_Type, Sub_Comment_Id, Sub_Comment_Owner, Sub_Comment, Sub_Comment_Likes, Sub_Comment_Time_Posted, hashtags, TagList, onlyText, data_Type)
  
  # Prepare Instagram SubComments
  Instagram_Subcomment_Common_Data <- Instagram_Subcomment_Common_Data %>% 
    select(Id = comment_id, source_Type, Sub_Comment_Id = comment_comment_id, Sub_Comment_Owner = comment_comment_owner_username, Sub_Comment = comment_comment_text, Sub_Comment_Likes = comment_comment_liked_count, Sub_Comment_Time_Posted = comment_comment_Time_Posted, hashtags, TagList, onlyText) %>% 
    mutate(data_Type = "subcomment")
  
  # Bind every file together into one
  sub_comment_file <- rbind(
    Instagram_Subcomment_Common_Data,
    youTube_Common_Data_SubComment,
    sub_comment_reddit)
  
  rm(youTube_Common_Data_SubComment, sub_comment_reddit, Instagram_Subcomment_Common_Data)
  
  return(sub_comment_file)
}
