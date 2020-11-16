import pickle as pi
from pymongo import collection
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import scripts.mongoConnection as mc

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, plot_confusion_matrix

class Model:

    my_tags = ['belonging', 'meaning', 'efficacy', 'distinctivness', 'self esteem', 'continuity']

    def __init__(self):
        self.X = None
        self.y = None

    def __import_X_train(self, npz_filepath):
        X = sc.sparse.load_npz(npz_filepath)
        return X
    
    def __import_X_test(self, npz_filepath):
        X_test = sc.sparse.load_npz(npz_filepath)
        return X_test
    
    def __import_Y_train(self, database, collection):
        y = mc.getCollection(database, collection)
        y = y.drop(["_id", "id"], axis=1)
        return y

    def __import_Y_test(self, database, collection):
        y_test = mc.getCollection(database, collection)
        y_test = y_test.drop(["_id", "id"], axis=1)
        return y_test
    
    def import_train_test_data(self, filepath, database, balancing_technique, feature_selection, fs_function):
        if feature_selection == True:
            X = self.__import_X_train(npz_filepath=filepath + balancing_technique + "_x_matrix_fs_" + fs_function + ".npz")
        else:
            X = self.__import_X_train(npz_filepath=filepath + balancing_technique + "_x_matrix.npz")

        x_test_fp = filepath + "X_test.npz"
        X_test = self.__import_X_test(x_test_fp)
        y = self.__import_Y_train(database=database, collection=balancing_technique + "_y")
        y_test = self.__import_Y_test(database=database, collection="y_test")
        return X, X_test, y, y_test

    def train_model(self, model, X, y):
        trained_model = model.fit(X, y)
        return trained_model
    
    def save_model(self, model, filepath):
        pi.dump(model, open(filepath, 'wb'))
        return print("Model saved")

    def load_model(self, filepath):
        model = pi.load(open(filepath))
        return model

    def load_features(self, filepath, balancing, balancing_tech, fs_tech, X_test):
        if balancing==True:
            filepath = filepath + balancing_tech + fs_tech + ".tchq"
            fs = pi.load(open(filepath, 'rb'))
            x_pred = fs.transform(X_test)
            return x_pred
        else:
            filepath = filepath + fs_tech + ".tchq"
            fs = pi.load(open(filepath, 'rb'))
            x_pred = fs.transform(X_test)
            return x_pred

    def evaluate(self, model, x_pred, y_test, average, normalize_cm, save_cm, filepath):

        y_pred = model.predict(x_pred)
        y_true = y_test

        print('classification_report\n %s' % classification_report(y_true, y_pred, target_names = self.my_tags))
        print('accuracy\n %s' % accuracy_score(y_true, y_pred))
        print('f1-score\n %s' % f1_score(y_true, y_pred, average=average))
        print('precision_score\n %s' % precision_score(y_true, y_pred, average=average))
        print('recall_score\n %s' % recall_score(y_true, y_pred, average=average))
        #print('confusion matrix\n %s' % confusion_matrix(y_true, y_pred, labels=self.my_tags))

        disp = plot_confusion_matrix(model, x_pred, y_true, display_labels=self.my_tags, cmap=plt.cm.Blues, normalize=normalize_cm)
        disp.ax_.set_title("Normalized confusion matrix")
        plt.xticks(rotation=90)

        if save_cm == True:
            plt.savefig(filepath, bbox_inches='tight', dpi=199)

    def predict_model(self):
        pass

#if __name__ == "__main__":

# Set paths
filepath_NPZ = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/NPZs/"
filepath_Model = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/Models_Test/"
filepath_Eval = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/Model_Eval_Test/"

# Model input parameters
seed = 69
n_cores = 6
iterations = 100
estimators = 100

# Istanciate classes
modeller = Model()
dt = DecisionTreeClassifier(random_state=seed)
logreg = LogisticRegression(n_jobs=n_cores, max_iter=seed)
nb = MultinomialNB()
rf = RandomForestClassifier(n_jobs=n_cores, n_estimators=estimators, random_state=seed)
svm = SGDClassifier(n_jobs=n_cores, loss='hinge', penalty='l2', alpha=iterations, random_state=seed, max_iter=iterations, tol=None)

# Model selection parameters
models = [dt, logreg, nb, rf, svm]
balancingTechniques = ["SMOTEENN", "NearMiss", "SMOTETomek","SMOTE", "TomekLinks"]
featureSelections = ["None", "chi2", "f_classif"]

for model in models:
    for balancingTechnique in balancingTechniques:
        for featureSelection in featureSelections:
            print("Model: " + str(model) + ", Balancing: " + balancingTechnique + ", Features: " + featureSelection + " started!")
            # import train and test data
            X, X_test, y, y_test = modeller.import_train_test_data(
                filepath=filepath_NPZ,
                database="09_TrainingData",
                balancing_technique=balancingTechnique,
                feature_selection=False,
                fs_function=featureSelections)

            trained_model = modeller.train_model(model=model, X=X, y=y)

            modeller.evaluate(
                model=trained_model,
                x_pred=X_test,
                y_test=y_test,
                average="macro",
                normalize_cm="true",
                save_cm=True,
                filepath=filepath_Eval + str(model) + "_" + balancingTechnique + "_" + featureSelection + ".png")

            modeller.save_model(
                model=trained_model,
                filepath=filepath_Model + str(model) + "_" + balancingTechnique + "_" + featureSelection + ".model")

            print("Model: " + str(model) + ", Balancing: " + balancingTechnique + ", Features: " + featureSelection + " done!")

#### Keine Ahnung was das macht
# features = modeller.load_features(
#     filepath=filepath_Model,
#     balancing=True,
#     balancing_tech="TomekLinks",
#     fs_tech="chi2",
#     X_test=X_test
#     )


#######
# Dataframe anlegen das und da alle codes und parameter reinschreiben?
# - Erleichtert die Evaluierung enorm
