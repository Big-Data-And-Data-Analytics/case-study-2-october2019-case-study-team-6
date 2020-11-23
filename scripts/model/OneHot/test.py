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

#TODO Delete unnecessary files cleanup the folder of one hot and rename the file to onehotencoding no spaces please

class one_hot: #TODO Capital name OneHot
    """One Hot class provides function to apply OneHot encoding using logistic regression on the given training and
    testing data to generate prediction with accuracy functions
        """
    def __init__(self, filepath, score_function, use_fs_data, acc_thresh):

        self.filepath = filepath
        self.X_test = sc.sparse.load_npz(filepath + '/NPZs/X_test.npz')#TODO #self to be added
        self.onehotencoder = OneHotEncoder(sparse=False)
        self.modelName = "LogisticRegression"
        self.balancing_technique = ["SMOTE",
                                    "SMOTEENN",
                                    "NearMiss",
                                    "SMOTETomek",
                                    "SMOTE",
                                    "TomekLinks"]
        self.score_function = score_function
        self.client = MongoClient('localhost', 27017) #TODO #Why?
        self.db = self.client['09_TrainingData'] #TODO #Why? Use mongodbconnection to get/load data needed anywhere
        self.use_fs_data = use_fs_data
        self.acc_thresh = acc_thresh
        self.SMOTE_y = getCollection('09_TrainingData', 'SMOTE_y') #TODO Remove

    def training_model(self,y_test, X_test): #TODO Rename variables to origin or pass data with different variable name
        counter = 0
        """
        #Loading y_test and X_test
        y_test = getCollection('09_TrainingData', 'y_test')
        """
        y_test = y_test.drop(["_id", "id"], axis=1)

        y_t = self.onehotencoder.fit_transform(y_test[['identityMotive']])
        y_t1 = pd.DataFrame(y_t[:, 0].tolist()).astype(str)

        x_pred = X_test

        for tech in self.balancing_technique:
            # Logging
            if self.use_fs_data:
                sys.stdout = open(
                    self.filepath + "/Model Evaluation/" + self.modelName + "_" + self.balancing_technique[
                        counter] + "_" + str(
                        self.use_fs_data) + "_" + self.score_function + '.txt', 'w')
            else:
                sys.stdout = open(
                    self.filepath + "/Model Evaluation/" + self.modelName + "_" + self.balancing_technique[
                        counter] + '.txt',
                    'w')

                # Load "y"
                coll_name = self.balancing_technique[counter] + "_y"
                y = self.db[coll_name]
                y = pd.DataFrame(list(y.find()))
                y = y.drop(["_id", "id"], axis=1)
                print(f'"y" loaded, {self.balancing_technique[counter]}')
                print(type(y))

                y1 = self.onehotencoder.fit_transform(y[['identityMotive']])
                y_1 = pd.DataFrame(y1[:, 0].tolist()).astype(str) #TODO Meaningful variable name? what data does y_1 holds?
            """
                y_2 = y1[:, 1].tolist()
                y_3 = y1[:, 2].tolist()
                y_4 = y1[:, 3].tolist()
                y_5 = y1[:, 4].tolist()
                y_6 = y1[:, 5].tolist()
            """  #TODO Removee if not using

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
            counter += 1

            logreg = LogisticRegression(n_jobs=2, max_iter=1000)
            print("Started fitting model with " + self.balancing_technique[counter])
            logreg.fit(X, y_1.values.ravel())
            print("Finished fitting model")

            print("Started predicting")
            y_pred = logreg.predict(x_pred)
            print("Finished predicting")

            ###### EVALUATION ######
            print("########### LOGISTIC REGRESSION WITH " + self.balancing_technique[counter] + " ###########")

            y_true = y_pred
            y_pred = y_t1
            print('classification_report\n %s' % metrics.classification_report(y_true, y_pred))
            print('accuracy\n %s' % metrics.accuracy_score(y_true, y_pred))
            print('f1-score\n %s' % metrics.f1_score(y_true, y_pred, average="micro"))
            print('precision_score\n %s' % metrics.precision_score(y_true, y_pred, average="micro"))
            print('recall_score\n %s' % metrics.recall_score(y_true, y_pred, average="macro"))
            # print('confusion matrix\n %s' % metrics.confusion_matrix(y_true, y_pred, labels=my_tags))

            acc = round(metrics.accuracy_score(y_true, y_pred) * 100, 2) # #TODO Remove if not using

            print("Logistic Regression Done with One hot Done")

        if __name__ == '__main__':

            fp = "/Users/Shared/Relocated Items/Security/Abhinav's Documents/SRH IT/Kinner, Maximilian (SRH Hochschule Heidelberg Student) - 06 Case Study I/02 Input_Data/03 Model"
            score_function = 'chi2'
            use_fs_data = False
            acc_thresh = 0.0

            oh = one_hot(fp, score_function,use_fs_data,acc_thresh)

            X_test = sc.sparse.load_npz(fp + '/NPZs/X_test.npz')

            y_test = getCollection('09_TrainingData', 'y_test')
            oh.training_model(y_test, X_test)

