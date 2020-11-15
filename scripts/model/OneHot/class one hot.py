import os
import sys
import pickle as pi
import pandas as pd
import scipy as sc
from sklearn.feature_extraction.text import CountVectorizer
import yaml
from pymongo import MongoClient
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from pymongo import MongoClient
#from scripts.mongoConnection import
#import matplotlib.pyplot as plt
from scripts.model.Decision_Tree import filepath
from scripts.model.OneHot.test import onehotencoder, X_test, y_test


class one_hot:
    """One Hot class provides function to apply OneHot encoding using logistic regression on the given training and testing data to
    generate prediction with accuracy functions
        """
    def __init__(self, filepath, score_function):
        self.filepath = filepath
        self.score_function = score_function
        self.my_tags = ['belonging', 'meaning', 'efficacy', 'distinctivness', 'self esteem', 'continuity']
        self.balancing_technique = [  # "ADASYN",
            "SMOTEENN",
            "NearMiss",
            "SMOTETomek",
            "SMOTE",
            "TomekLinks"]
        self.use_fs_data = False
        self.modelName = "LogisticRegression"
        self.acc_thresh = 0.0
        # MongoDB connection
        self.client = MongoClient('localhost', 27017)
        self.db = self.client['09_TrainingData']

    def training_model(self):

        for tech in self.balancing_technique:

            # Logging
            if self.use_fs_data:
                sys.stdout = open(
                    self.filepath + "/Model Evaluation/" + self.modelName + "_" + self.balancing_technique[counter] + "_" + str(
                        self.use_fs_data) + "_" + self.score_function + '.txt', 'w')
            else:
                sys.stdout = open(
                    self.filepath + "/Model Evaluation/" + self.modelName + "_" + self.balancing_technique[counter] + '.txt',
                    'w')

            # Load "y"
            coll_name = self.balancing_technique[counter] + "_y"
            y = self.db[coll_name]
            y = pd.DataFrame(list(y.find()))
            y = y.drop(["_id", "id"], axis=1)
            print(f'"y" loaded, {self.balancing_technique[counter]}')
            print(y)

            y1 = onehotencoder.fit_transform(y[['identityMotive']])
            y_1 = y1[:, 0].tolist()
            y_2 = y1[:, 1].tolist()
            y_3 = y1[:, 2].tolist()
            y_4 = y1[:, 3].tolist()
            y_5 = y1[:, 4].tolist()
            y_6 = y1[:, 5].tolist()

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
                    fs = pi.load(open(filepath + '/Models/Feature_'
                                      + self.balancing_technique[counter] + 'fs_chi2.tchq', 'rb'))
                    print(self.balancing_technique[counter])
                    X_test_chi2 = fs.transform(X_test)
                    x_pred = X_test_chi2
                else:
                    X = sc.sparse.load_npz(filepath + '/NPZs/' + self.balancing_technique[counter]
                                           + '_x_matrix_fs_f_classif.npz')
                    print(f'"x" feature selected with f_classif loaded, {self.balancing_technique[counter]}')
                    print(self.balancing_technique[counter])

                    fs = pi.load(
                        open(filepath + '/Models/Feature_' + self.balancing_technique[counter] + 'fs_f_classif.tchq', 'rb'))
                    X_test_f_classif = fs.transform(X_test)
                    x_pred = X_test_f_classif
            print(X)
            #### LOGISTIC REGRESSION ####

            logreg = LogisticRegression(n_jobs=6, max_iter=100)

            print("Started fitting model with " + self.balancing_technique[counter])
            logreg.fit(X, y)
            print("Finished fitting model")

            print("Started predicting")
            y_pred = logreg.predict(x_pred)
            print("Finished predicting")

            ###### EVALUATION ######
            print("########### LOGISTIC REGRESSION WITH " + self.balancing_technique[counter] + " ###########")

            y_true = y_pred
            y_pred = y_test
            print('classification_report\n %s' % metrics.classification_report(y_true, y_pred, target_names=self.my_tags))
            print('accuracy\n %s' % metrics.accuracy_score(y_true, y_pred))
            print('f1-score\n %s' % metrics.f1_score(y_true, y_pred, average="micro"))
            print('precision_score\n %s' % metrics.precision_score(y_true, y_pred, average="micro"))
            print('recall_score\n %s' % metrics.recall_score(y_true, y_pred, average="macro"))
            print('confusion matrix\n %s' % metrics.confusion_matrix(y_true, y_pred, labels=self.my_tags))

            acc = round(metrics.accuracy_score(y_true, y_pred) * 100, 2)
            ## Fancy Confusion Matrix
            """disp = metrics.plot_confusion_matrix(logreg, x_pred, y_test, display_labels=self.my_tags, cmap=plt.cm.Blues,
                                                 normalize="true")
            disp.ax_.set_title("Log Reg Normalized confusion matrix " + "Acc: " + str(acc))
            plt.xticks(rotation=90)
            if self.use_fs_data == True:
                plt.savefig(filepath + '/Model Evaluation/' + self.modelName + '_Confusion_matrix_with_normalization' + "_" +
                            self.balancing_technique[counter] + "_" + self.score_function + '.png', bbox_inches='tight', dpi=199)
            else:
                plt.savefig(filepath + '/Model Evaluation/' + self.modelName + '_Confusion_matrix_with_normalization' + "_" +
                            self.balancing_technique[counter] + '.png', bbox_inches='tight', dpi=199)
            """

            # titles_options = [
            #    ("Log Reg Confusion matrix, without normalization " + "Acc: " + str(acc), None),
            #    ("Log Reg Normalized confusion matrix " + "Acc: " + str(acc), 'true')]

            # conf_counter = 0

            # for title, normalize in titles_options:
            #    disp = metrics.plot_confusion_matrix(logreg, x_pred, y_test, display_labels=my_tags, cmap=plt.cm.Blues, normalize=normalize)
            #    disp.ax_.set_title(title)
            #    plt.xticks(rotation=90)
            #    if conf_counter == 0:
            #        plt.savefig('LogisticRegression_Confusion_matrix_without_normalization' + "_" + balancing_technique[counter] + '.png', bbox_inches='tight')
            #    else:
            #        plt.savefig('LogisticRegression_Confusion_matrix_with_normalization' + "_" + balancing_technique[counter] + '.png', bbox_inches='tight')

            #    conf_counter = conf_counter + 1

            # plt.show()

            # Create filename and dump model
            if not self.use_fs_data:
                if accuracy_score(y_pred, y_test) > self.acc_thresh:
                    filename = filepath + "/Models/Logistic_Regression_" + self.balancing_technique[counter] \
                               + "_" + str(acc) + ".model"
                    pi.dump(logreg, open(filename, 'wb'))
                else:
                    print("Model accuracy below " + self.acc_thresh + ", model not saved")
            else:
                if self.score_function == "chi2":
                    filename = self.filepath + "/Models/Logistic_Regression_" + self.balancing_technique[counter] \
                               + "_" + str(acc) + "_fs_chi2.model"
                    pi.dump(logreg, open(filename, 'wb'))
                else:
                    filename = self.filepath + "/Models/Logistic_Regression_" + self.balancing_technique[counter] \
                               + "_" + str(acc) + "_fs_f_classif.model"
                    pi.dump(logreg, open(filename, 'wb'))

            counter = counter + 1
            print("##################################")
            print("##################################")

        print("Logistic Regression Done with One hot Done")


if __name__ == '__main__':

    one_hot()
    getCollection('')
    training(y_test, X_test)


    pass