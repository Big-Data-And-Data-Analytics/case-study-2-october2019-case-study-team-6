import pickle as pi
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

    def load_X(self, filepath, balancing, balancing_tech):
        if balancing==True:
            X = sc.sparse.load_npz(filepath + balancing_tech + "_x_matrix_fs_chi2.npz")
            return X
        else:
            X = sc.sparse.load_npz(filepath + "_x_matrix_fs_chi2.npz")
            return X

    def train_model(self, model, X, y):
        trained_model = model.fit(X, y)
        return trained_model
    
    def save_model(self, model, filepath):
        pi.dump(model, open(filepath, 'wb'))
        return print("Model saved")

    def load_model(self, filepath):
        model = pi.load(open(filepath))
        return model
    
    def import_X_test(filepath):
        X_test = sc.sparse.load_npz(filepath)
        return X_test

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

    def evaluate(self, model, x_pred, y_true, average, normalize_cm):

        y_pred = model.predict(x_pred)

        print('classification_report\n %s' % classification_report(y_true, y_pred, target_names = self.my_tags))
        print('accuracy\n %s' % accuracy_score(y_true, y_pred))
        print('f1-score\n %s' % f1_score(y_true, y_pred, average=average))
        print('precision_score\n %s' % precision_score(y_true, y_pred, average=average))
        print('recall_score\n %s' % recall_score(y_true, y_pred, average=average))
        print('confusion matrix\n %s' % confusion_matrix(y_true, y_pred, labels=self.my_tags))

        disp = plot_confusion_matrix(model, x_pred, y_pred, display_labels=self.my_tags, cmap=plt.cm.Blues, normalize=normalize_cm)
        disp.ax_.set_title("Normalized confusion matrix")
        plt.xticks(rotation=90)

    def predict_model(self):
        pass

#if __name__ == "__main__":

# Set variables
seed = 69
n_cores = 6
iterations = 10000
estimators = 100
filepath_NPZ = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/NPZs/"
filepath_Model = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/Models_Test/"

# Istanciate classes
modeller = Model()
dt = DecisionTreeClassifier(random_state=seed)
logreg = LogisticRegression(n_jobs=n_cores, max_iter=seed)
nb = MultinomialNB()
rf = RandomForestClassifier(n_jobs=n_cores, n_estimators=estimators, random_state=seed)
svm = SGDClassifier(n_jobs=n_cores, loss='hinge', penalty='l2', alpha=iterations, random_state=seed, max_iter=iterations, tol=None)

# Get y_test from MongoDB
y_test = mc.getCollection("09_TrainingData", "y_test")
y_test = y_test.drop(["_id", "id"], axis=1)

# Get _y
y = mc.getCollection("09_TrainingData", "TomekLinks_y")
y = y.drop(["_id", "id"], axis=1)

# Load x.npz with balancing
X = modeller.load_X(filepath=filepath_NPZ, balancing=True, balancing_tech="TomekLinks")

model = modeller.train_model(model=logreg, X=X, y=y)

modeller.save_model(model=model, filepath=filepath_Model + "Logistic_Regression.model")

X_test = modeller.import_X_test(filepath=filepath_NPZ + "X_test.npz")

features = modeller.load_features(
    filepath=filepath_Model,
    balancing=True,
    balancing_tech="TomekLinks",
    fs_tech="chi2",
    X_test=X_test
    )


#######
## Wir machen das so:
# Egal welche Balancing Technique ausgewählt wird, wir laden X und Y entsprechend gleich mit
#   - Aber wie kann ich Y mitladen? ich glaube wir sollen keine Funktion so innerhalb der Funktion nutzen
# Dann wird das model trainiert
# Dann kann man das model evaluieren
# Dann kann man das model speichern bzw. predicts drauf ausführen