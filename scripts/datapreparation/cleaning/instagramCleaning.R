
# Import collections
post_raw_con <- mongo(collection = "Instagram_Post_Raw", db = "01_NationalIdentity_Crawled", url = "mongodb://localhost")
comment_raw_con <- mongo(collection = "Instagram_Comment_Raw", db = "01_NationalIdentity_Crawled", url = "mongodb://localhost")
#test_raw_con <- mongo(collection = "Instagram_Test_Raw", db = "01_NationalIdentity_Crawled", url = "mongodb://localhost")

# Export collections
post_clean_con <- mongo(collection = "Instagram_Post_Cleaned", db = "02_NationalIdentity_Cleaned", url = "mongodb://localhost")
comment_clean_con <- mongo(collection = "Instagram_Comment_Cleaned", db = "02_NationalIdentity_Cleaned", url = "mongodb://localhost")
#test_clean_con <- mongo(collection = "Instagram_Test_Cleaned", db = "02_NationalIdentity_Cleaned", url = "mongodb://localhost")

# Data import
post_data <- post_raw_con$find(query = "{}")
comment_data <- comment_raw_con$find(query = "{}")
#test_data <- test_raw_con$find(query = "{}")

post_clean_con$insert(post_data)
comment_clean_con$insert(comment_data)
#test_clean_con$insert(test_data)


# Remove connections
rm(post_raw_con, comment_raw_con)
rm(post_clean_con, comment_clean_con)

# Remove data
rm(post_data, comment_data)

print("Instagram cleaning done")
