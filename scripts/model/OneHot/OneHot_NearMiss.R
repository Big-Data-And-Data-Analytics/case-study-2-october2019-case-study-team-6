filepath = "D:/OneDrive - SRH IT/Case Study 1/02 Input_Data/03 Model"

library(data.table)
library(tidyverse)
library(mltools)
library(caret)
library(MASS)
library(reticulate)
np <- import("numpy")

#Loading Testing x
npz5 <- np$load(paste(filepath, "/NPZs/X_test.npz", sep = ""))
x_mt1 <- as.data.frame(as.data.frame(npz5$f[["indptr"]])[-1,])
colnames(x_mt1)[1] <- "x"

#Loading Testing y
y_test <- read.csv(paste(filepath, "/CSVs/y_test.csv", sep = ""), stringsAsFactors = T)

test <- one_hot(as.data.table(y_test))

#SMOTE
SMOTE_y <- read.csv(paste(filepath, "/CSVs/NearMiss_y.csv", sep = ""), stringsAsFactors = T)
data_1h <- one_hot(as.data.table(SMOTE_y))

#Loading x
npz1 <- np$load(paste(filepath, "/NPZs/NearMiss_x_matrix.npz", sep = ""))
x_m1 <- as.data.frame(as.data.frame(npz1$f[["indptr"]])[-1,])
colnames(x_m1)[1] <- "x"

#LDA
Imb <- data_1h$identityMotive_belonging 
Imc <- data_1h$identityMotive_continuity
Imd <- data_1h$identityMotive_distinctiveness
Ime <- data_1h$identityMotive_efficacy
Imm <- data_1h$identityMotive_meaning
Ims <- data_1h$identityMotive_selfEsteem

x1 <- x_m1$x

#Running the model Belonging
#LDA
fit.ldb1 <- lda(Imb ~ x1)
y1 <- predict(fit.ldb1,data.frame(x1 = x_mt1$x))
g1 <- table(test$identityMotive_belonging,y1$class)

# Accuracy of model
sum(diag(g1))/sum(g1)

#Running the model Continuity
#LDA
fit.ldc1 <- lda(Imc ~ x1)
yc1 <- predict(fit.ldc1,data.frame(x1 = x_mt1$x))
gc1 <- table(test$identityMotive_continuity,yc1$class)

# Accuracy of model
sum(diag(gc1))/sum(gc1)

#Running the model Distinctiveness
#LDA
fit.ldd1 <- lda(Imd ~ x1)
yd1 <- predict(fit.ldd1,data.frame(x1 = x_mt1$x))
gd1 <- table(test$identityMotive_distinctiveness,yd1$class)

# Accuracy of model
sum(diag(gd1))/sum(gd1)

#Running the model Efficacy
#LDA
fit.lde1 <- lda(Ime ~ x1)
ye1 <- predict(fit.lde1,data.frame(x1 = x_mt1$x))
ge1 <- table(test$identityMotive_efficacy,ye1$class)

# Accuracy of model
sum(diag(ge1))/sum(ge1)

#Running the model meaning
#LDA
fit.ldm1 <- lda(Imm ~ x1)
ym1 <- predict(fit.ldm1,data.frame(x1 = x_mt1$x))
gm1 <- table(test$identityMotive_meaning,ym1$class)

# Accuracy of model
sum(diag(gm1))/sum(gm1)

#Running the model self esteem
#LDA
fit.lds1 <- lda(Ims ~ x1)
ys1 <- predict(fit.lds1,data.frame(x1 = x_mt1$x))
gs1 <- table(test$identityMotive_selfEsteem,ys1$class)

# Accuracy of model
sum(diag(gs1))/sum(gs1)
####################################################################################
#NearMiss - class

#Loading x
npz1 <- np$load("NearMiss_x_matrix_fs_f_classif.npz")
x_m1 <- as.data.frame(as.data.frame(npz1$f[["indptr"]])[-1,])
colnames(x_m1)[1] <- "x"

#LDA
Imb <- data_1h$identityMotive_belonging 
Imc <- data_1h$identityMotive_continuity
Imd <- data_1h$identityMotive_distinctiveness
Ime <- data_1h$identityMotive_efficacy
Imm <- data_1h$identityMotive_meaning
Ims <- data_1h$identityMotive_selfEsteem

x1 <- x_m1$x

#Running the model Belonging
#LDA
fit.ldb <- lda(Imb ~ x1)
y <- predict(fit.ldb,data.frame(x1 = x_mt1$x))
g <- table(test$identityMotive_belonging,y$class)

# Accuracy of model
sum(diag(g))/sum(g)

#Running the model Continuity
#LDA
fit.ldc <- lda(Imc ~ x1)
yc <- predict(fit.ldc,data.frame(x1 = x_mt1$x))
gc <- table(test$identityMotive_continuity,yc$class)

# Accuracy of model
sum(diag(gc))/sum(gc)

#Running the model Distinctiveness
#LDA
fit.ldd <- lda(Imd ~ x1)
yd <- predict(fit.ldd,data.frame(x1 = x_mt1$x))
gd <- table(test$identityMotive_distinctiveness,yd$class)

# Accuracy of model
sum(diag(gd))/sum(gd)

#Running the model Efficacy
#LDA
fit.lde <- lda(Ime ~ x1)
ye <- predict(fit.lde,data.frame(x1 = x_mt1$x))
ge <- table(test$identityMotive_efficacy,ye$class)

# Accuracy of model
sum(diag(ge))/sum(ge)

#Running the model meaning
#LDA
fit.ldm <- lda(Imm ~ x1)
ym <- predict(fit.ldm,data.frame(x1 = x_mt1$x))
gm <- table(test$identityMotive_meaning,ym$class)

# Accuracy of model
sum(diag(gm))/sum(gm)

#Running the model self esteem
#LDA
fit.lds <- lda(Ims ~ x1)
ys <- predict(fit.lds,data.frame(x1 = x_mt1$x))
gs <- table(test$identityMotive_selfEsteem,ys$class)

# Accuracy of model
sum(diag(gs))/sum(gs)

#NearMiss - chi2

#Loading x
npz1 <- np$load("NearMiss_x_matrix_fs_chi2.npz")
x_m1 <- as.data.frame(as.data.frame(npz1$f[["indptr"]])[-1,])
colnames(x_m1)[1] <- "x"

#LDA
Imb <- data_1h$identityMotive_belonging 
Imc <- data_1h$identityMotive_continuity
Imd <- data_1h$identityMotive_distinctiveness
Ime <- data_1h$identityMotive_efficacy
Imm <- data_1h$identityMotive_meaning
Ims <- data_1h$identityMotive_selfEsteem

x1 <- x_m1$x

#Running the model Belonging
#LDA
fit.ldb <- lda(Imb ~ x1)
y <- predict(fit.ldb,data.frame(x1 = x_mt1$x))
g <- table(test$identityMotive_belonging,y$class)

# Accuracy of model
sum(diag(g))/sum(g)

#Running the model Continuity
#LDA
fit.ldc <- lda(Imc ~ x1)
yc <- predict(fit.ldc,data.frame(x1 = x_mt1$x))
gc <- table(test$identityMotive_continuity,yc$class)

# Accuracy of model
sum(diag(gc))/sum(gc)

#Running the model Distinctiveness
#LDA
fit.ldd <- lda(Imd ~ x1)
yd <- predict(fit.ldd,data.frame(x1 = x_mt1$x))
gd <- table(test$identityMotive_distinctiveness,yd$class)

# Accuracy of model
sum(diag(gd))/sum(gd)

#Running the model Efficacy
#LDA
fit.lde <- lda(Ime ~ x1)
ye <- predict(fit.lde,data.frame(x1 = x_mt1$x))
ge <- table(test$identityMotive_efficacy,ye$class)

# Accuracy of model
sum(diag(ge))/sum(ge)

#Running the model meaning
#LDA
fit.ldm <- lda(Imm ~ x1)
ym <- predict(fit.ldm,data.frame(x1 = x_mt1$x))
gm <- table(test$identityMotive_meaning,ym$class)

# Accuracy of model
sum(diag(gm))/sum(gm)

#Running the model self esteem
#LDA
fit.lds <- lda(Ims ~ x1)
ys <- predict(fit.lds,data.frame(x1 = x_mt1$x))
gs <- table(test$identityMotive_selfEsteem,ys$class)

# Accuracy of model
sum(diag(gs))/sum(gs)