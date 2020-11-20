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

import yaml
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

##TODO Add to the evaluation function something that creates a dataframe where all necessary data for evaluation (accuracy, f1, recall, ...) is stored for easier overview
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

        print('classification_report\n %s' % classification_report(y_true=y_true, y_pred=y_pred, target_names=self.my_tags))
        print('accuracy\n %s' % accuracy_score(y_true=y_true, y_pred=y_pred))
        print('f1-score\n %s' % f1_score(y_true=y_true, y_pred=y_pred, average=average))
        print('precision_score\n %s' % precision_score(y_true=y_true, y_pred=y_pred, average=average))
        print('recall_score\n %s' % recall_score(y_true=y_true, y_pred=y_pred, average=average))
        #print('confusion matrix\n %s' % confusion_matrix(y_true, y_pred, labels=self.my_tags))
        
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average)
        precision = precision_score(y_true=y_true, y_pred=y_pred, average=average)
        recall = recall_score(y_true=y_true, y_pred=y_pred, average=average)

        disp = plot_confusion_matrix(model, x_pred, y_true, display_labels=self.my_tags, cmap=plt.cm.Blues, normalize=normalize_cm)
        disp.ax_.set_title("Normalized confusion matrix")
        plt.xticks(rotation=90)

        if save_cm == True:
            plt.savefig(filepath, bbox_inches='tight', dpi=199)

        return accuracy, f1, precision, recall

    def predict_model(self, filepath_model, balancing_techniques):
        vocab_collection = mc.getCollection(db="09_TrainingData", col="CountVectorVocabulary")
        vocab_list = vocab_collection['dict'].to_list()
        vocab = yaml.safe_load(vocab_list[0])

        modelFiles = [f for f in listdir(filepath_model) if isfile(join(filepath_model, f))]

        # Filter FileNamesList for Models
        def features_files(model):
            return '.tchq' not in model
        
        # Filtering Data using features_files
        exclude_tchq = filter(features_files, modelFiles)
        modelFiles = list(exclude_tchq)
        modelFiles.sort()
        modelCounter = 0

        nltk.download('punkt')
        nltk.download('wordnet')

        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        def removeStopWords(listOfWords):
            filtered_list = [w for w in listOfWords if not w in stop_words]
            lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in filtered_list])
            return lemmatized_output
        def lemmatizeRemoveStopWords(df):
            df = df.str.lower()
            df=df.replace(r'@', '', regex=True)
            df=df.replace(r'http\S+', '', regex=True).replace(r'http \S+', '', regex=True).replace(r'www\S+', '', regex=True)
            df=df.str.replace('[^\w\s]','')
            df=df.str.replace('\s\s+',' ')
            df = df.str.strip()
            df=df.apply(word_tokenize)
            df=df.apply(removeStopWords)
            return df

        # Load Vocab for Non Feature Selected Models
        def loadNewDataForPrediction(vocab):
            validation_set = list()
            while True:
                val = input('Enter n number of validation texts, type "EXIT" to exit\n')
                if val.lower() == 'exit':
                    break
                else:
                    validation_set.append(val)
            valid = pd.Series(validation_set)
            valid = lemmatizeRemoveStopWords(valid)
            return valid

        continueFlag = False
        while True:
            if continueFlag:
                print('\n \n \n \n \n')
                q = input('~ Do you want to continue, type J/N~\n')
                if q.lower() == 'j':
                    pass
                else:
                    break

            modelCounter = 0
            for model in modelFiles:
                if '.tchq' not in model:
                    print(modelCounter, model)
                modelCounter += 1
            val = input('~ Enter Model Number, type "EXIT/exit" to Exit ~\n')
            if val.lower() == 'exit':
                break
            else:
                # Load Model
                val = int(val)
                mdl = pi.load(open(filepath_model + modelFiles[val], 'rb'))
                print('Model Loaded')
                # Load Feature Data / Normal Data
                if 'fs' in modelFiles[val]:
                    print('Feature Data')
                    cnt = 0
                    for tech in balancing_techniques:
                        print(f'Balancing Technique: {tech}')
                        if tech in modelFiles[val]:
                            if 'chi2' in modelFiles[val]:
                                # Load Feature Selection Object
                                fs = pi.load(open(filepath_model + 'Feature_' + balancing_techniques[cnt] + 'fs_chi2.tchq', 'rb'))
                                print(filepath_model + 'Feature_' + balancing_techniques[cnt] + 'fs_chi2.tchq')
                                # Get New Data
                                newData = loadNewDataForPrediction(vocab)

                                # Transform using Count Vectorizer and Entire Vocab
                                cv_pred = CountVectorizer(vocabulary=vocab)
                                x_pred = cv_pred.transform(newData)

                                # Transform to Feature Selected Data
                                X_test_chi2 = fs.transform(x_pred)
                                x_pred = X_test_chi2

                                # Predict
                                y_pred_validation = mdl.predict(x_pred)
                                print(y_pred_validation)
                            else:
                                # Load Feature Selection Object
                                fs = pi.load(open(filepath_model + 'Feature_' + balancing_techniques[cnt] + 'fs_chi2.tchq', 'rb'))
                                print(filepath_model + 'Feature_' + balancing_techniques[cnt] + 'fs_chi2.tchq')
                                # Get New Data
                                newData = loadNewDataForPrediction(vocab)

                                # Transform using Count Vectorizer and Entire Vocab
                                cv_pred = CountVectorizer(vocabulary=vocab)
                                x_pred = cv_pred.transform(newData)

                                # Transform to Feature Selected Data
                                X_test_chi2 = fs.transform(x_pred)
                                x_pred = X_test_chi2

                                # Predict
                                y_pred_validation = mdl.predict(x_pred)
                                print(y_pred_validation)
                            break
                        cnt += 1
                else:
                    print('Normal Data')
                    print(modelFiles[val])

                    newData = loadNewDataForPrediction(vocab)
                    cv_pred = CountVectorizer(vocabulary=vocab)
                    x_pred = cv_pred.transform(newData)
                    y_pred_validation = mdl.predict(x_pred)
                    print(y_pred_validation)
                    ## TODO See if text can be shown as with the predicted value

            continueFlag = True


#if __name__ == "__main__":

# Set paths
filepath_NPZ = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/NPZs/"
filepath_Model = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/Models_Test/"
filepath_Eval = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/Model_Eval_Test/"

# Create empty evaluation dataframe
eval_frame = pd.DataFrame(columns=["model", "Balancing", "Features", "Accuracy", "F1-Score", "Precision", "Recall"])

# Model input parameters
seed = 69
n_cores = -1
iterations = 1000 # Logistic Regression, SVM: Higher value means longer running
estimators = 10 # Random Forest: Higher value means longer running
alpha = 0.1
learning_rate = "optimal"

# Istanciate classes
modeller = Model()
dt = DecisionTreeClassifier(random_state=seed)
logreg = LogisticRegression(n_jobs=n_cores, max_iter=iterations)
nb = MultinomialNB()
rf = RandomForestClassifier(n_jobs=n_cores, n_estimators=estimators, random_state=seed)
svm = SGDClassifier(  # New parameters added -> experimental
    n_jobs=n_cores,
    loss='hinge',
    penalty='l2',
    alpha=alpha,
    random_state=seed,
    max_iter=iterations,
    learning_rate=learning_rate,
    tol=None)

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

            accuracy, f1, precision, recall = modeller.evaluate(
                model=trained_model,
                x_pred=X_test,
                y_test=y_test,
                average="macro",
                normalize_cm="true",
                save_cm=True,
                filepath=filepath_Eval + str(model) + "_" + balancingTechnique + "_" + featureSelection + ".png")
            
            temp_d = {
                "Model": str(model),
                "Balancing": balancingTechnique,
                "Features": featureSelection,
                "Accuracy": accuracy,
                "F1-Score": f1,
                "Precision": precision,
                "Recall": recall}

            temp = pd.DataFrame(data=temp_d, index=[0])
            eval_frame = pd.concat([eval_frame, temp])

            modeller.save_model(
                model=trained_model,
                filepath=filepath_Model + str(model) + "_" + balancingTechnique + "_" + featureSelection + ".model")

            print("Model: " + str(model) + ", Balancing: " + balancingTechnique + ", Features: " + featureSelection + " done!")

print("Modelling done")

eval_frame.reset_index(inplace=True)
eval_frame.to_csv(filepath_Eval + "Eval_Overview.csv")

# modeller.predict_model(
#     filepath_model=filepath_Model,
#     balancing_techniques=balancingTechniques)
