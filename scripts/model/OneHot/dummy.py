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

filepath = "/Users/Shared/Relocated Items/Security/Abhinav's Documents/SRH IT/Kinner, Maximilian (SRH Hochschule Heidelberg Student) - 06 Case Study I/02 Input_Data/03 Model"
X_test = sc.sparse.load_npz(filepath + '/NPZs/X_test.npz')
onehotencoder = OneHotEncoder(sparse=False)
SMOTE_y = getCollection('09_TrainingData','SMOTE_y')
y_test = getCollection('09_TrainingData','y_test')
y_test = y_test.drop(["_id", "id"], axis=1)
y_t = onehotencoder.fit_transform(y_test[['identityMotive']])
y_t1 = pd.DataFrame(y_t[:, 0].tolist()).astype(str)

modelName = "LogisticRegression"
balancing_technique = ["SMOTE"]
score_function = 'chi2'
client = MongoClient('localhost', 27017)
db = client['09_TrainingData']
use_fs_data = False
modelName = "LogisticRegression"
acc_thresh = 0.0

my_tags = ['belonging', 'meaning', 'efficacy', 'distinctivness', 'self esteem', 'continuity']





counter = 0
x_pred = X_test
            # One Hot Encoder
onehotencoder = OneHotEncoder(sparse=False)
            # Logging
if use_fs_data:
    sys.stdout = open(
    filepath + "/Model Evaluation/" + modelName + "_" + balancing_technique[counter] + "_" + str(
        use_fs_data) + "_" + score_function + '.txt', 'w')
else:
    sys.stdout = open(
                    filepath + "/Model Evaluation/" + modelName + "_" + balancing_technique[counter] + '.txt',
                    'w')

            # Load "y"
    coll_name = balancing_technique[counter] + "_y"
    y = db[coll_name]
    y = pd.DataFrame(list(y.find()))
    y = y.drop(["_id", "id"], axis=1)
    print(f'"y" loaded, {balancing_technique[counter]}')
    print(type(y))

    y1 = onehotencoder.fit_transform(y[['identityMotive']])
    y_1 = pd.DataFrame(y1[:, 0].tolist()).astype(str)
    y_2 = y1[:, 1].tolist()
    y_3 = y1[:, 2].tolist()
    y_4 = y1[:, 3].tolist()
    y_5 = y1[:, 4].tolist()
    y_6 = y1[:, 5].tolist()

            # Load "X"
if not use_fs_data:

    X = sc.sparse.load_npz(filepath + '/NPZs/' + balancing_technique[counter] +
                                       '_x_matrix.npz')
    print(f'"x" loaded, {balancing_technique[counter]}')
else:
    if score_function == "chi2":
        X = sc.sparse.load_npz(filepath + '/NPZs/' +
                                           balancing_technique[counter] + '_x_matrix_fs_chi2.npz')
        print(f'"x" feature selected with chi2 loaded, {balancing_technique[counter]}')
        fs = pi.load(open(filepath + '/Models/Feature_'
        + balancing_technique[counter] + 'fs_chi2.tchq', 'rb'))
        print(balancing_technique[counter])
        X_test_chi2 = fs.transform(X_test)
        x_pred = X_test_chi2
    else:
        X = sc.sparse.load_npz(filepath + '/NPZs/' + balancing_technique[counter]
                                           + '_x_matrix_fs_f_classif.npz')
        print(f'"x" feature selected with f_classif loaded, {balancing_technique[counter]}')
        print(balancing_technique[counter])

        fs = pi.load(
                        open(filepath + '/Models/Feature_' + balancing_technique[counter] + 'fs_f_classif.tchq', 'rb'))
        X_test_f_classif = fs.transform(X_test)
        x_pred = X_test_f_classif
        print(X)
        #### LOGISTIC REGRESSION ####

logreg = LogisticRegression(n_jobs=2, max_iter=1000)
print("Started fitting model with " + balancing_technique[counter])
logreg.fit(X, y_1.values.ravel())
print("Finished fitting model")

print("Started predicting")
y_pred = logreg.predict(x_pred)
print("Finished predicting")

            ###### EVALUATION ######
print("########### LOGISTIC REGRESSION WITH " + balancing_technique[counter] + " ###########")

y_true = y_pred
y_pred = y_t1
print('classification_report\n %s' % metrics.classification_report(y_true, y_pred))
print('accuracy\n %s' % metrics.accuracy_score(y_true, y_pred))
print('f1-score\n %s' % metrics.f1_score(y_true, y_pred, average="micro"))
print('precision_score\n %s' % metrics.precision_score(y_true, y_pred, average="micro"))
print('recall_score\n %s' % metrics.recall_score(y_true, y_pred, average="macro"))
#print('confusion matrix\n %s' % metrics.confusion_matrix(y_true, y_pred, labels=my_tags))

acc = round(metrics.accuracy_score(y_true, y_pred) * 100, 2)

print("Logistic Regression Done with One hot Done")


