# Functions for the YouTube-Crawler
## Additional Info
### Unicode Keycodes
# https://selenium.dev/selenium/docs/api/py/_modules/selenium/webdriver/common/keys.html

## Libraries
library(tidyverse)
library(RSelenium)
library(rvest)
library(mongolite)

############ ESTABLISH DOCKER CONNECTION ############
dockerConnection <- function(browserName, windowSizeX, windowSizeY){
  
  remDr <- RSelenium::remoteDriver(remoteServerAddr = "localhost",
                                   port = 4445L,
                                   browserName = browserName)
  # Open connection to the container
  remDr$open()
  # Sets the browsers window size, to geht the same amount of elements each scan
  remDr$setWindowSize(windowSizeX, windowSizeY)
  
  return(remDr)
  
}

############ EXTRACT VIDEO_IDS AND META DATA FROM YOUTUBE VIDEO OVERVIEW PAGE ############
getVideos <- function(hashtag){
  remDr$navigate(paste("https://www.youtube.com/results?search_query=", hashtag, sep = ""))
  Sys.sleep(1)
  # Scroll down
  if (videoScrolls > 1) {
    for (videoOverviewScrolls in 1:videoScrolls) {
      videoOverviewScrollsElement <- remDr$findElement(using = "xpath", "//html//body")
      videoOverviewScrollsElement$sendKeysToElement(sendKeys = list("\ue010"))
      Sys.sleep(3)
      remDr$screenshot(display = TRUE)
    } 
  }
  
  videoCounter <<- length(remDr$findElements(using = "xpath", paste("//body/ytd-app/div[@id='content']/ytd-page-manager[@id='page-manager']/ytd-search[@class='style-scope ytd-page-manager']/div[@id='container']/ytd-two-column-search-results-renderer[@class='style-scope ytd-search']/div[@id='primary']/ytd-section-list-renderer[@class='style-scope ytd-two-column-search-results-renderer']/div[@id='contents']/ytd-item-section-renderer[@class='style-scope ytd-section-list-renderer']/div[@id='contents']/ytd-video-renderer", sep = "")))
  mainSiteHTML <- remDr$getPageSource()[[1]] %>% 
    read_html()
  
  # Get video URLs
  for (cc in 1:videoCounter) {
    
    video_id <- as.data.frame(html_attr(html_node(mainSiteHTML, xpath = paste("//body/ytd-app/div[@id='content']/ytd-page-manager[@id='page-manager']/ytd-search[@class='style-scope ytd-page-manager']/div[@id='container']/ytd-two-column-search-results-renderer[@class='style-scope ytd-search']/div[@id='primary']/ytd-section-list-renderer[@class='style-scope ytd-two-column-search-results-renderer']/div[@id='contents']/ytd-item-section-renderer[@class='style-scope ytd-section-list-renderer']/div[@id='contents']/ytd-video-renderer[", cc, "]/div[1]/ytd-thumbnail[1]/a[1]", sep = "")), name = "href"))
    video_Title <- as.data.frame(html_text(html_node(mainSiteHTML, xpath = paste("/html[1]/body[1]/ytd-app[1]/div[1]/ytd-page-manager[1]/ytd-search[1]/div[1]/ytd-two-column-search-results-renderer[1]/div[1]/ytd-section-list-renderer[1]/div[2]/ytd-item-section-renderer[1]/div[3]/ytd-video-renderer[", cc, "]/div[1]/div[1]/div[1]/div[1]/h3[1]/a[1]/yt-formatted-string[1]", sep = ""))))
    channel <- as.data.frame(html_text(html_node(mainSiteHTML, xpath = paste("/html[1]/body[1]/ytd-app[1]/div[1]/ytd-page-manager[1]/ytd-search[1]/div[1]/ytd-two-column-search-results-renderer[1]/div[1]/ytd-section-list-renderer[1]/div[2]/ytd-item-section-renderer[1]/div[3]/ytd-video-renderer[", cc, "]/div[1]/div[1]/div[1]/ytd-video-meta-block[1]/div[1]/div[1]/ytd-channel-name[1]/div[1]/div[1]/yt-formatted-string[1]/a[1]", sep = ""))))
    clicks <- as.data.frame(html_text(html_node(mainSiteHTML, xpath = paste("/html[1]/body[1]/ytd-app[1]/div[1]/ytd-page-manager[1]/ytd-search[1]/div[1]/ytd-two-column-search-results-renderer[1]/div[1]/ytd-section-list-renderer[1]/div[2]/ytd-item-section-renderer[1]/div[3]/ytd-video-renderer[", cc, "]/div[1]/div[1]/div[1]/ytd-video-meta-block[1]/div[1]/div[2]/span[1]", sep = ""))))
    time_posted <- as.data.frame(html_text(html_node(mainSiteHTML, xpath = paste("/html[1]/body[1]/ytd-app[1]/div[1]/ytd-page-manager[1]/ytd-search[1]/div[1]/ytd-two-column-search-results-renderer[1]/div[1]/ytd-section-list-renderer[1]/div[2]/ytd-item-section-renderer[1]/div[3]/ytd-video-renderer[", cc, "]/div[1]/div[1]/div[1]/ytd-video-meta-block[1]/div[1]/div[2]/span[2]", sep = ""))))
    description <- as.data.frame(html_text(html_node(mainSiteHTML, xpath = paste("/html[1]/body[1]/ytd-app[1]/div[1]/ytd-page-manager[1]/ytd-search[1]/div[1]/ytd-two-column-search-results-renderer[1]/div[1]/ytd-section-list-renderer[1]/div[2]/ytd-item-section-renderer[1]/div[3]/ytd-video-renderer[", cc, "]/div[1]/div[1]/yt-formatted-string[1]", sep = ""))))
    scan_Date <- Sys.time()
    
    video_id <- cbind(video_id, video_Title, channel, clicks, time_posted, description)
    duplicateVideoDetection <- rbind(duplicateVideoDetection, video_id)
  }
  colnames(duplicateVideoDetection) <- c("video_id", "video_title", "channel", "clicks", "time_posted", "description")
  
  return(duplicateVideoDetection)
  
}
