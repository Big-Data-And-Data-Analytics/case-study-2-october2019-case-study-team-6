import pickle as pi
import pandas as pd
import scipy as sc
from sklearn.feature_extraction.text import CountVectorizer
import yaml
from pymongo import MongoClient
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
#import matplotlib.pyplot as plt
import sys

# Define params
acc_thresh = 0.0
use_fs_data = False
score_function = "f_classif"
modelName = "One Hot Encoding"
filepath = "/Users/Shared/Relocated Items/Security/Abhinav's Documents/SRH IT/Kinner, Maximilian (SRH Hochschule Heidelberg Student) - 06 Case Study I/02 Input_Data/03 Model"
# filepath = '/home/shubham/PycharmProjects/Case Study 1  Extras'
my_tags = ['belonging', 'meaning', 'efficacy', 'distinctivness', 'self esteem', 'continuity']

# MongoDB connection
client = MongoClient('localhost', 27017)
db = client['09_TrainingData']

# MongoDB import y-test
y_test = db.y_test
y_test = pd.DataFrame(list(y_test.find()))
y_test = y_test.drop(["_id", "id"], axis=1)
print(y_test)

onehotencoder = OneHotEncoder(sparse=False)
y_test1 = onehotencoder.fit_transform(y_test[['identityMotive']])
print(y_test1)

# Balancing technique
balancing_technique = [  # "ADASYN",
    "SMOTEENN",
    "NearMiss",
    "SMOTETomek",
    "SMOTE",
    "TomekLinks"]
counter = 0

# Import Vocabulary
cv_vocab = db.CountVectorVocabulary
cv_vocab = pd.DataFrame(list(cv_vocab.find()))
cv_vocab = cv_vocab['dict'].to_list()
d = yaml.load(cv_vocab[0])  # Get first element

# Import X_test
X_test = sc.sparse.load_npz(filepath + '/NPZs/X_test.npz')
x_pred = X_test

#### RUN CODE ####
for tech in balancing_technique:

    # Logging
    if use_fs_data == True:
        sys.stdout = open(filepath + "/Model Evaluation/" + modelName + "_" + balancing_technique[counter] + "_" + str(
            use_fs_data) + "_" + score_function + '.txt', 'w')
    else:
        sys.stdout = open(filepath + "/Model Evaluation/" + modelName + "_" + balancing_technique[counter] + '.txt',
                          'w')

    # Load "y"
    coll_name = balancing_technique[counter] + "_y"
    y = db[coll_name]
    y = pd.DataFrame(list(y.find()))
    y = y.drop(["_id", "id"], axis=1)
    print(f'"y" loaded, {balancing_technique[counter]}')
    print(y)
    y1 = onehotencoder.fit_transform(y[['identityMotive']])
    print(y1)

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

            fs = pi.load(open(filepath + '/Models/Feature_' + balancing_technique[counter] + 'fs_f_classif.tchq', 'rb'))
            X_test_f_classif = fs.transform(X_test)
            x_pred = X_test_f_classif

    #### LOGISTIC REGRESSION ####
    logreg = LogisticRegression(n_jobs=6, max_iter=100)

    print("Started fitting model with " + balancing_technique[counter])
    logreg.fit(X, y1)
    print("Finished fitting model")

    print("Started predicting")
    y_pred = logreg.predict(x_pred)
    print("Finished predicting")

    ###### EVALUATION ######
    print("########### LOGISTIC REGRESSION WITH " + balancing_technique[counter] + " ###########")

    y_true = y_pred
    y_pred = y_test
    print('classification_report\n %s' % metrics.classification_report(y_true, y_pred, target_names=my_tags))
    print('accuracy\n %s' % metrics.accuracy_score(y_true, y_pred))
    print('f1-score\n %s' % metrics.f1_score(y_true, y_pred, average="micro"))
    print('precision_score\n %s' % metrics.precision_score(y_true, y_pred, average="micro"))
    print('recall_score\n %s' % metrics.recall_score(y_true, y_pred, average="macro"))
    print('confusion matrix\n %s' % metrics.confusion_matrix(y_true, y_pred, labels=my_tags))

    acc = round(metrics.accuracy_score(y_true, y_pred) * 100, 2)
    ## Fancy Confusion Matrix
    disp = metrics.plot_confusion_matrix(logreg, x_pred, y_test, display_labels=my_tags, cmap=plt.cm.Blues,
                                         normalize="true")
    disp.ax_.set_title("Log Reg Normalized confusion matrix " + "Acc: " + str(acc))
    plt.xticks(rotation=90)
    if use_fs_data == True:
        plt.savefig(filepath + '/Model Evaluation/' + modelName + '_Confusion_matrix_with_normalization' + "_" +
                    balancing_technique[counter] + "_" + score_function + '.png', bbox_inches='tight', dpi=199)
    else:
        plt.savefig(filepath + '/Model Evaluation/' + modelName + '_Confusion_matrix_with_normalization' + "_" +
                    balancing_technique[counter] + '.png', bbox_inches='tight', dpi=199)

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
    if not use_fs_data:
        if accuracy_score(y_pred, y_test) > acc_thresh:
            filename = filepath + "/Models/Logistic_Regression_" + balancing_technique[counter] \
                       + "_" + str(acc) + ".model"
            pi.dump(logreg, open(filename, 'wb'))
        else:
            print("Model accuracy below " + acc_thresh + ", model not saved")
    else:
        if score_function == "chi2":
            filename = filepath + "/Models/Logistic_Regression_" + balancing_technique[counter] \
                       + "_" + str(acc) + "_fs_chi2.model"
            pi.dump(logreg, open(filename, 'wb'))
        else:
            filename = filepath + "/Models/Logistic_Regression_" + balancing_technique[counter] \
                       + "_" + str(acc) + "_fs_f_classif.model"
            pi.dump(logreg, open(filename, 'wb'))

    counter = counter + 1
    print("##################################")
    print("##################################")

print("Logistic Regression Done with One hot Done")
