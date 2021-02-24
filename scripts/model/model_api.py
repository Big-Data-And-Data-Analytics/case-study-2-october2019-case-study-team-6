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
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
models = {}
lemmatizer = WordNetLemmatizer()
stop_words = ()
modelFiles = []
vocab = 0
class Prediction:
    
    @app.get('/models')
    def getModels():
        return models

    def init_model(filepath_model):
        global models, lemmatizer, stop_words, modelFiles, vocab
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
        for model in modelFiles:
            if '.tchq' not in model:
                models[modelCounter] = model
            modelCounter += 1
        nltk.download('punkt')
        nltk.download('wordnet')
        stop_words = set(stopwords.words('english'))
        return "Model Initiated; Proceed for Prediction"

    def predict(inp, filepath_model, balancing_techniques):
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

            val = inp['sentence']
            validation_set.append(val)
            valid = pd.Series(validation_set)
            valid = lemmatizeRemoveStopWords(valid)
            return valid

        val = inp['modelNumber']
        val = int(val)
        mdl = pi.load(open(filepath_model + modelFiles[val], 'rb'))
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
                        return (y_pred_validation)
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
                        return (y_pred_validation)
                    break
                cnt += 1
        else:
            newData = loadNewDataForPrediction(vocab)
            cv_pred = CountVectorizer(vocabulary=vocab)
            x_pred = cv_pred.transform(newData)
            y_pred_validation = mdl.predict(x_pred)
            response = {}
            response['prediction'] = y_pred_validation[0]
            return response


class takeInput(BaseModel):
    sentence: str
    modelNumber: str

@app.post('/init')
def initFunction():    
    filepath_Model = "D:/SRH IT/Kinner, Maximilian (SRH Hochschule Heidelberg Student) - Case Study 1/02 Input_Data/03 Model/Models_Test/"
    return Prediction.init_model(filepath_model=filepath_Model)

@app.post('/predict')
def predictInput(takeInput: takeInput):
    filepath_Model = "D:/SRH IT/Kinner, Maximilian (SRH Hochschule Heidelberg Student) - Case Study 1/02 Input_Data/03 Model/Models_Test/"
    balancingTechniques = ["SMOTEENN", "NearMiss", "SMOTETomek","SMOTE", "TomekLinks"]
    inp = takeInput.dict()
    return Prediction.predict(inp, filepath_model=filepath_Model, balancing_techniques=balancingTechniques)

