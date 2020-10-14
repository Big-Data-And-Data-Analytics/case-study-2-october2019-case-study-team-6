library(reticulate)
library(mltools)
library(data.table)

#py_install("pandas")
#py_install("pymongo")
#py_install("yaml")
#py_install("scikit-learn")
#py_install("scipy")


np <- import("numpy")

filepath = "D:/OneDrive - SRH IT/Case Study 1/02 Input_Data/03 Model"

#use_python("C:/Users/maxim/AppData/Local/Programs/Python/Python38/python.exe")

ln <- function(){

  #x_test <- readline(prompt="Enter tweet:")
  x_test <- "together we can do it! go germany!"
  x_test <- write.csv(x_test, paste(filepath, "/oneHotPreprocessing/X_test.csv", sep = ""), row.names = FALSE)
  
  #setwd(dir = "04 Model/01 Models/OneHot/")
  #py_run_file("oneHot_Preprocessing.py")
  
  #Loading Testing x
  npz5 <- np$load(paste(filepath, "/OneHotPreprocessing/OneHotPredictionTest.npz", sep = ""))
  x_mt1 <- as.data.frame(as.data.frame(npz5$f[["indptr"]])[-1,])
  colnames(x_mt1)[1] <- "x"
  
  #Loading Testing y
  y_test <- read.csv(paste(filepath, "/oneHotPreprocessing/y_test.csv", sep = ""), stringsAsFactors = T, header = TRUE)
  
  test <- one_hot(as.data.table(y_test))[1,]
  
  # Belonging
  y <- predict(fit.ldb,data.frame(x1 = x_mt1$x))
  g <- table(test$identityMotive_belonging,y$class)
  y1 <- predict(fit.ldb1,data.frame(x1 = x_mt1$x))
  g1 <- table(test$identityMotive_belonging,y1$class)
  
  #Continuity
  yc <- predict(fit.ldc,data.frame(x1 = x_mt1$x))
  gc <- table(test$identityMotive_continuity,yc$class)
  yc1 <- predict(fit.ldc1,data.frame(x1 = x_mt1$x))
  gc1 <- table(test$identityMotive_continuity,yc1$class)
  
  #Distinctiveness
  yd <- predict(fit.ldd,data.frame(x1 = x_mt1$x))
  gd <- table(test$identityMotive_distinctiveness,yd$class)
  yd1 <- predict(fit.ldd1,data.frame(x1 = x_mt1$x))
  gd1 <- table(test$identityMotive_distinctiveness,yd1$class)
  
  #Efficacy
  ye <- predict(fit.lde,data.frame(x1 = x_mt1$x))
  ge <- table(test$identityMotive_efficacy,ye$class)
  ye1 <- predict(fit.lde1,data.frame(x1 = x_mt1$x))
  ge1 <- table(test$identityMotive_efficacy,ye1$class)
  
  #Meaning
  ym <- predict(fit.ldm,data.frame(x1 = x_mt1$x))
  gm <- table(test$identityMotive_meaning,ym$class)
  ym1 <- predict(fit.ldm1,data.frame(x1 = x_mt1$x))
  gm1 <- table(test$identityMotive_meaning,ym1$class)
  
  #Self esteem
  ys <- predict(fit.lds,data.frame(x1 = x_mt1$x))
  gs <- table(test$identityMotive_selfEsteem,ys$class)
  ys1 <- predict(fit.lds1,data.frame(x1 = x_mt1$x))
  gs1 <- table(test$identityMotive_selfEsteem,ys1$class)
  
  HS <-     data.frame(
    Accuracy = c(sum(diag(gm))/sum(gm), sum(diag(gm1))/sum(gm1), 
                 sum(diag(gs))/sum(gs),sum(diag(gs1))/sum(gs1),
                 sum(diag(ge))/sum(ge),sum(diag(ge1))/sum(ge1),
                 sum(diag(g))/sum(g),sum(diag(g1))/sum(g1),
                 sum(diag(gc))/sum(gc), sum(diag(gc1))/sum(gc1),
                 sum(diag(gd))/sum(gd), sum(diag(gd1))/sum(gd1)) ,
    Tagging = c("Meaning","Meaning","Self Esteem","Self Esteem",
                "Efficacy","Efficacy","Belonging","Belonging", "Continuity","Continuity", "Distinctiveness", "Distinctiveness"),
    Technique = c("SMOTE", "NearMiss"))
  
  View(HS)
  print(paste("Entered text belongs to: " + max(HS) ))
  
}


ln()
