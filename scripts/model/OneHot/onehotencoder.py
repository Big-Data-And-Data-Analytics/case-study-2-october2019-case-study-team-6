import os
import sys
import pickle as pi
import pandas as pd
import scipy as sc
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from pymongo import MongoClient
from scripts.mongoConnection import getCollection, insertCollection
from sklearn.preprocessing import OneHotEncoder

class OneHot:
    """One Hot class provides function to apply OneHot encoding using logistic regression on the given training and
    testing data to generate prediction with accuracy functions
    OneHotEncoder features are encoded using a one-hot encoding scheme. This creates a binary column for each category and returns a sparse matrix or dense array (depending on the sparse parameter)
    By default, the encoder derives the categories based on the unique values in each feature.
        """

    def __init__(self, filepath, score_function, use_fs_data, acc_thresh):

        self.filepath = filepath
        self.X_test = sc.sparse.load_npz(self.filepath + '/NPZs/X_test.npz')
        self.onehotencoder = OneHotEncoder(sparse=False)
        self.balancing_technique = ["SMOTEENN",
                                    "NearMiss",
                                    "SMOTETomek",
                                    "SMOTE",
                                    "TomekLinks"]
        self.score_function = score_function
        self.use_fs_data = use_fs_data
        self.acc_thresh = acc_thresh

    def training_model(self, y_test,X_test):
        counter = 0

        #Loading y_test and X_test
        #y_test = getCollection('09_TrainingData', 'y_test')

        y_test = y_test.drop(["_id", "id"], axis=1)

        y_t = self.onehotencoder.fit_transform(y_test[['identityMotive']])
        y_t1 = pd.DataFrame(y_t[:, 0].tolist()).astype(str)

        x_pred = X_test

        for tech in self.balancing_technique:
            # Load "y"
            coll_name = self.balancing_technique[counter] + "_y"
            y = getCollection("09_TrainingData",coll_name)
            #y = pd.DataFrame(list(y.find()))
            y = y.drop(["_id", "id"], axis=1)
            print(f'"y" loaded, {self.balancing_technique[counter]}')
            print(type(y))

            yoh = self.onehotencoder.fit_transform(y[['identityMotive']])
            yoh1 = pd.DataFrame(yoh[:, 0].tolist()).astype(str)

            # Load "X"
            if not self.use_fs_data:

                X = sc.sparse.load_npz(self.filepath + '/NPZs/' + self.balancing_technique[counter] +
                                       '_x_matrix.npz')
                print(f'"x" loaded, {self.balancing_technique[counter]}')
            else:
                if self.score_function == "chi2":
                    X = sc.sparse.load_npz(self.filepath + '/NPZs/' +
                                           self.balancing_technique[counter] + '_x_matrix_fs_chi2.npz')
                    print(f'"x" feature selected with chi2 loaded, {self.balancing_technique[counter]}')
                    fs = pi.load(open(self.filepath + '/Models/Feature_'
                                      + self.balancing_technique[counter] + 'fs_chi2.tchq', 'rb'))
                    print(self.balancing_technique[counter])
                    X_test_chi2 = fs.transform(X_test)
                    x_pred = X_test_chi2
                else:
                    X = sc.sparse.load_npz(self.filepath + '/NPZs/' + self.balancing_technique[counter]
                                           + '_x_matrix_fs_f_classif.npz')
                    print(f'"x" feature selected with f_classif loaded, {self.balancing_technique[counter]}')
                    print(self.balancing_technique[counter])

                    fs = pi.load(
                        open(self.filepath + '/Models/Feature_' + self.balancing_technique[
                            counter] + 'fs_f_classif.tchq', 'rb'))
                    X_test_f_classif = fs.transform(X_test)
                    x_pred = X_test_f_classif
                    print(X)
                    #### LOGISTIC REGRESSION ####

            logreg = LogisticRegression(n_jobs=6, max_iter=1000)
            print("Started fitting model with " + self.balancing_technique[counter])
            logreg.fit(X, yoh1.values.ravel())
            print("Finished fitting model")

            print("Started predicting")
            y_pred = logreg.predict(x_pred)
            print("Finished predicting")

            ###### EVALUATION ######
            print("########### LOGISTIC REGRESSION WITH " + self.balancing_technique[counter] + " ###########")

            y_true = y_pred
            # y_pred = y_t1
            print('classification_report\n %s' % metrics.classification_report(y_true, y_t1))
            print('accuracy\n %s' % metrics.accuracy_score(y_true, y_t1))
            print('f1-score\n %s' % metrics.f1_score(y_true, y_t1, average="macro"))
            print('precision_score\n %s' % metrics.precision_score(y_true, y_t1, average="macro"))
            print('recall_score\n %s' % metrics.recall_score(y_true, y_t1, average="macro"))
            # print('confusion matrix\n %s' % metrics.confusion_matrix(y_true, y_pred, labels=my_tags))


            print("Logistic Regression Done with One hot Done")

            counter += 1


if __name__ == '__main__':
    fp = "/Users/Shared/Relocated Items/Security/Abhinav's Documents/SRH IT/Kinner, Maximilian (SRH Hochschule Heidelberg Student) - 06 Case Study I/02 Input_Data/03 Model"
    #fp = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model"
    score_function = 'chi2'
    use_fs_data = False
    acc_thresh = 0.0

    oh = OneHot(fp, score_function, use_fs_data, acc_thresh)

    Xtest = sc.sparse.load_npz(fp + '/NPZs/X_test.npz')

    ytest = getCollection('09_TrainingData', 'y_test')
    oh.training_model(ytest, Xtest)
