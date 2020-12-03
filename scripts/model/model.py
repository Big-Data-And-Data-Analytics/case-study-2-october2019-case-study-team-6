import nltk
import yaml

import pickle as pi
import scipy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scripts.mongoConnection as mc

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

from os import listdir
from os.path import isfile, join
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime, date


class Model:
    """The Model class provides every function necessary for loading the prepared data, training and evaluating the model and finally predicting
    """    

    my_tags = ['belonging', 'meaning', 'efficacy', 'distinctivness', 'self esteem', 'continuity']

    def __init__(self):
        self.X = None
        self.y = None

    def __import_X_train(self, npz_filepath):
        """Imports the X train data.

        :param npz_filepath: Provide the filepath including "/NPZs/"
        :type npz_filepath: string
        :return: X sparse matrix for training
        :rtype: sparse matrix
        """        
        X = sc.sparse.load_npz(npz_filepath)
        return X
    
    def __import_X_test(self, npz_filepath):
        """Imports the X test data.

        :param npz_filepath: Provide the filepath including "/NPZs/"
        :type npz_filepath: string
        :return: X sparse matrix for testing
        :rtype: sparse matrix
        """       
        X_test = sc.sparse.load_npz(npz_filepath)
        return X_test
    
    def __import_Y_train(self, database, collection):
        """Imports the Y train data.

        :param database: The database where the data is stored
        :type database: string
        :param collection: The collection where the data is stored
        :type collection: string
        :return: Y data frame for training
        :rtype: Pandas data frame
        """        
        y = mc.getCollection(database, collection)
        y = y.drop(["_id", "id"], axis=1)
        return y

    def __import_Y_test(self, database, collection):
        """Imports the Y test data.

        :param database: The database where the data is stored
        :type database: string
        :param collection: The collection where the data is stored
        :type collection: string
        :return: Y data frame for testing
        :rtype: Pandas data frame
        """        
        y_test = mc.getCollection(database, collection)
        y_test = y_test.drop(["_id", "id"], axis=1)
        return y_test
    
    def __import_Y_train_onehot(self, database, collection):
        """Imports the Y train data.

        :param database: The database where the data is stored
        :type database: string
        :param collection: The collection where the data is stored
        :type collection: string
        :return: Y data frame for training
        :rtype: Pandas data frame
        """
        y = mc.getCollection(database, collection)
        y = y.drop(["_id", "id"], axis=1)

        y_train_onehot = onehotencoder.fit_transform(y[['identityMotive']])
        y_train_onehot_ = pd.DataFrame(y_train_onehot[:, 0].tolist()).astype(str)
        return y_train_onehot_

    def __import_Y_test_onehot(self, database, collection):
        """Imports the Y test data.

        :param database: The database where the data is stored
        :type database: string
        :param collection: The collection where the data is stored
        :type collection: string
        :return: Y data frame for testing
        :rtype: Pandas data frame
        """
        y_test = mc.getCollection(database, collection)
        y_test = y_test.drop(["_id", "id"], axis=1)

        y_test_onehot = onehotencoder.fit_transform(y_test[['identityMotive']])
        y_test_onehot_ = pd.DataFrame(y_test_onehot[:, 0].tolist()).astype(str)
        return y_test_onehot_

    def import_train_test_data(self, filepath, database, balancing_technique, feature_selection, fs_function, use_onehot):
        """Runs the functions __import_X_train, __import_X_test, __import_Y_train and __import_Y_test regarding the given input.

        :param filepath: Provide the filepath including "/NPZs/"
        :type filepath: string
        :param database: The database where the data is stored
        :type database: string
        :param balancing_technique: The name of the balancing technique ("SMOTEENN", "NearMiss", "SMOTETomek","SMOTE", "TomekLinks")
        :type balancing_technique: string
        :param feature_selection: If the used data should use any feature selection data
        :type feature_selection: bool
        :param fs_function: The name of the feature selection function ("None", "chi2", "f_classif")
        :type fs_function: string
        :param use_onehot: Should one-hot encoded y should be used
        :type use_onehot: bool
        :return: The data for training and testing the data
        :rtype: Sparse matrix, sparse matrix, Pandas data frame, Pandas data frame
        """        
        if feature_selection == True:
            X = self.__import_X_train(npz_filepath=filepath + balancing_technique + "_x_matrix_fs_" + fs_function + ".npz")

            x_test_fp = filepath + "X_test.npz"
            X_test = self.__import_X_test(x_test_fp)

            fs = pi.load(open(filepath + '/Feature_' + balancing_technique + 'fs_' + fs_function + '.tchq', 'rb'))
            X_test = fs.transform(X_test)
        else:
            X = self.__import_X_train(npz_filepath=filepath + balancing_technique + "_x_matrix.npz")
            x_test_fp = filepath + "X_test.npz"
            X_test = self.__import_X_test(x_test_fp)

        if use_onehot == True:
            y = self.__import_Y_train_onehot(database=database, collection=balancing_technique + "_y")
            y_test = self.__import_Y_test_onehot(database=database, collection="y_test")
        else:
            y = self.__import_Y_train(database=database, collection=balancing_technique + "_y")
            y_test = self.__import_Y_test(database=database, collection="y_test")
        return X, X_test, y, y_test

    def train_model(self, model, X, y):
        """Function for training the model

        :param model: Model class object
        :type model: class object
        :param X: The X train data
        :type X: Sparse matrix
        :param y: The y train data
        :type y: Pandas data frame
        :return: The trained model
        :rtype: Model object
        """        
        trained_model = model.fit(X, y)
        return trained_model

    def save_model(self, model, filepath):
        """Function for saving the model to a specific filepath

        :param model: The model object
        :type model: Model object
        :param filepath: The filepath where the model should be stored including the model name
        :type filepath: string
        :return: Print statement that the model was saved
        :rtype: string
        """        
        pi.dump(model, open(filepath, 'wb'))
        return print("Model saved")

    def load_model(self, filepath):
        """Function for loading a model

        :param filepath: The filepath where the model is stored including the model name
        :type filepath: string
        :return: The loaded model object
        :rtype: Model object
        """        
        model = pi.load(open(filepath))
        return model

    def evaluate(self, model, x_pred, y_test, average, normalize_cm, save_cm, filepath, use_onehot):
        """Function for evaluating a model

        :param model: The model object which should be evaluated
        :type model: Model object
        :param x_pred: X test data
        :type x_pred: Sparse matrix
        :param y_test: Y test data
        :type y_test: Pandas data frame
        :param average: Which average function should be used
        :type average: string
        :param normalize_cm: If the confusion matrix should be normalized ("true", "false")
        :type normalize_cm: string
        :param save_cm: If the confusion matrix should be stored
        :type save_cm: bool
        :param filepath: The filepath where the confusion matrix should be stored including the filename
        :type filepath: string
        :return: accuracy, f1, precision, recall
        :rtype: float, float, float, float
        """        

        y_pred = model.predict(x_pred)
        y_true = y_test

        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average)
        precision = precision_score(y_true=y_true, y_pred=y_pred, average=average)
        recall = recall_score(y_true=y_true, y_pred=y_pred, average=average)
        
        if use_onehot == True:
            y_probas = model.predict_proba(x_pred)
            probs = y_probas[:, 1]
            auc = roc_auc_score(y_true, probs)

        if use_onehot == False:
            disp = plot_confusion_matrix(model, x_pred, y_true, display_labels=self.my_tags, cmap=plt.cm.Blues, normalize=normalize_cm)
            disp.ax_.set_title("Normalized confusion matrix")
            plt.xticks(rotation=90)

            if save_cm == True:
                plt.savefig(filepath, bbox_inches='tight', dpi=199)

        if use_onehot == True:
            return accuracy, f1, precision, recall, auc
        else:
            return accuracy, f1, precision, recall

    def predict_model(self, filepath_model, balancing_techniques):
        """Function for predicting stuff right away on the console

        :param filepath_model: The filepath the model directory
        :type filepath_model: string
        :param balancing_techniques: Which balancing techniques should be included
        :type balancing_techniques: list
        :return: [description]
        :rtype: [type]
        """        
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


if __name__ == "__main__":

    # Set paths
    # filepath_NPZ = "C:/Users/maxim/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/NPZs/"
    # filepath_Model = "C:/Users/maxim/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/Models_Test/"
    # filepath_Eval = "C:/Users/maxim/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/Model_Eval_Test/"

    filepath_NPZ = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/NPZs/"
    filepath_Model = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/Models_Test/"
    filepath_Eval = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/Model_Eval_Test/"

    # Create empty evaluation dataframe
    eval_frame = pd.DataFrame(columns=["Model", "Balancing", "Features", "Accuracy", "F1-Score", "Precision", "Recall", "AUC", "Date", "Timestamp"])

    # Model input parameters
    seed = 69
    np.random.seed(seed)
    n_cores = -1
    iterations = 1000 # Logistic Regression, SVM: Higher value means longer running
    estimators = 10 # Random Forest: Higher value means way longer running
    alpha = 0.1
    learning_rate = "optimal"
    use_onehot=False

    # Istanciate classes # TODO Can go under init method
    modeller = Model()
    onehotencoder = OneHotEncoder(sparse=False)
    nb = MultinomialNB()
    dt = DecisionTreeClassifier(random_state=seed)
    logreg = LogisticRegression(n_jobs=n_cores, max_iter=iterations)
    rf = RandomForestClassifier(n_jobs=n_cores, n_estimators=estimators, random_state=seed)
    svm = SGDClassifier(  # New parameters added -> experimental
        n_jobs=n_cores,
        loss='log',
        penalty='l2',
        alpha=alpha,
        random_state=seed,
        max_iter=iterations,
        learning_rate=learning_rate,
        tol=None)

    # Model selection parameters (Model != One-hot!)
    # TODO Can go under init method
    models = [svm, nb, dt, logreg, rf]
    balancingTechniques = ["SMOTEENN", "NearMiss", "SMOTETomek","SMOTE", "TomekLinks"]
    featureSelections = ["None", "chi2", "f_classif"]

    # Model selection parameters (test)
    # models = [svm]
    # balancingTechniques = ["SMOTEENN", "NearMiss", "SMOTETomek","SMOTE", "TomekLinks"]
    # featureSelections = ["None", "chi2", "f_classif"]

    total_models = len(models) * len(balancingTechniques) * len(featureSelections)
    print(f'Training a total of {total_models} single class models')
    # TODO New Function? for the loops
    for model in models:
        for balancingTechnique in balancingTechniques:
            for featureSelection in featureSelections:
                today = date.today()
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")

                if use_onehot == True:
                    modelname =  "OneHot_" + str(model)
                else:
                    modelname = str(model)
                
                print("Started model: " + modelname + ", Balancing: " + balancingTechnique + ", Features: " + featureSelection)
                # import train and test data
                if featureSelection == "None":
                    feature_selection = False
                else:
                    feature_selection = True

                X, X_test, y, y_test = modeller.import_train_test_data(
                    filepath=filepath_NPZ,
                    database="09_TrainingData",
                    balancing_technique=balancingTechnique,
                    feature_selection=feature_selection,
                    fs_function=featureSelection,
                    use_onehot=use_onehot)

                trained_model = modeller.train_model(model=model, X=X, y=y)

                accuracy, f1, precision, recall = modeller.evaluate(
                    model=trained_model,
                    x_pred=X_test,
                    y_test=y_test,
                    average="macro",
                    normalize_cm="true",
                    save_cm=True,
                    filepath=filepath_Eval + modelname + "_" + balancingTechnique + "_" + featureSelection + ".png",
                    use_onehot=use_onehot)
                
                temp_d = {
                    "Model": modelname,
                    "Balancing": balancingTechnique,
                    "Features": featureSelection,
                    "Accuracy": accuracy,
                    "F1-Score": f1,
                    "Precision": precision,
                    "Recall": recall,
                    "Date": today,
                    "Timestamp": current_time
                    }

                temp = pd.DataFrame(data=temp_d, index=[0])
                eval_frame = pd.concat([eval_frame, temp])

                modeller.save_model(
                    model=trained_model,
                    filepath=filepath_Model + modelname + "_" + balancingTechnique + "_" + featureSelection + ".model")

                print("Done model: " + modelname + ", Balancing: " + balancingTechnique + ", Features: " + featureSelection)
                total_models = total_models - 1
                print(str(total_models) + " models left.")

    ################################################
    ############### ONE HOT ENCODING ###############
    ################################################
    use_onehot=True

    # Model selection parameters (Model = One-hot!)
    models = [logreg]
    balancingTechniques = ["SMOTEENN", "NearMiss", "SMOTETomek","SMOTE", "TomekLinks"]
    featureSelections = ["None", "chi2", "f_classif"]

    total_models = len(models) * len(balancingTechniques) * len(featureSelections)
    print(f'Training a total of {total_models} multiclass models')

    # TODO New Function? for the loops
    for model in models:
        for balancingTechnique in balancingTechniques:
            for featureSelection in featureSelections:
                today = date.today()
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")

                if use_onehot == True:
                    modelname =  "OneHot_" + str(model)
                else:
                    modelname = str(model)
                
                print("Started model: " + modelname + ", Balancing: " + balancingTechnique + ", Features: " + featureSelection)
                # import train and test data
                if featureSelection == "None":
                    feature_selection = False
                else:
                    feature_selection = True

                X, X_test, y, y_test = modeller.import_train_test_data(
                    filepath=filepath_NPZ,
                    database="09_TrainingData",
                    balancing_technique=balancingTechnique,
                    feature_selection=feature_selection,
                    fs_function=featureSelection,
                    use_onehot=use_onehot)

                trained_model = modeller.train_model(model=model, X=X, y=y)

                accuracy, f1, precision, recall, auc = modeller.evaluate(
                    model=trained_model,
                    x_pred=X_test,
                    y_test=y_test,
                    average="macro",
                    normalize_cm="true",
                    save_cm=True,
                    filepath=filepath_Eval + modelname + "_" + balancingTechnique + "_" + featureSelection + ".png",
                    use_onehot=use_onehot)
                
                temp_d = {
                    "Model": modelname,
                    "Balancing": balancingTechnique,
                    "Features": featureSelection,
                    "Accuracy": accuracy,
                    "F1-Score": f1,
                    "Precision": precision,
                    "Recall": recall,
                    "AUC": auc,
                    "Date": today,
                    "Timestamp": current_time
                    }

                temp = pd.DataFrame(data=temp_d, index=[0])
                eval_frame = pd.concat([eval_frame, temp])

                modeller.save_model(
                    model=trained_model,
                    filepath=filepath_Model + modelname + "_" + balancingTechnique + "_" + featureSelection + ".model")

                print("Done model: " + modelname + ", Balancing: " + balancingTechnique + ", Features: " + featureSelection)
                total_models = total_models - 1
                print(str(total_models) + " models left.")

    if isfile(filepath_Eval + "Eval_Overview.csv"):
        existing_eval_frame = pd.read_csv(filepath_Eval + "Eval_Overview.csv")
        eval_frame = pd.concat([existing_eval_frame, eval_frame])
        eval_frame.reset_index(inplace=True)
        eval_frame.to_csv(filepath_Eval + "Eval_Overview.csv")
    else:
        eval_frame.reset_index(inplace=True)
        eval_frame.to_csv(filepath_Eval + "Eval_Overview.csv")

    print("Modelling done")

    # modeller.predict_model(
    #     filepath_model=filepath_Model,
    #     balancing_techniques=balancingTechniques)
