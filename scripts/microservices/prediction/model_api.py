from flask import Flask, request
import flask
import json
import pickle as pi
from os import listdir
from os.path import isfile, join
import nltk
import pandas as pd
import mongoConnection as mc
import uvicorn
import yaml
from fastapi import FastAPI
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from fastapi import Response
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')

# app = FastAPI()
app = Flask(__name__)
lemmatizer = WordNetLemmatizer()


class Prediction:

    def __init__(self, database, vocab):
        self.database = database
        self.vocab = vocab
        self.models = {}
        self.stop_words = set(stopwords.words('english'))
        self.modelFiles = []

    def getModels(self):
        return self.models

    def init_model(self, filepath_model):
        global lemmatizer
        """Function for predicting stuff right away on the console

        :param filepath_model: The filepath the model directory
        :type filepath_model: string
        :param balancing_techniques: Which balancing techniques should be included
        :type balancing_techniques: list
        :return: [description]
        :rtype: [type]
        """
        vocab_collection = mc.getCollection(db=self.database, col=self.vocab)
        vocab_list = vocab_collection['dict'].to_list()
        self.vocab = yaml.safe_load(vocab_list[0])

        self.modelFiles = [f for f in listdir(filepath_model) if isfile(join(filepath_model, f))]

        # Filter FileNamesList for Models
        def features_files(model):
            return '.tchq' not in model

        # Filtering Data using features_files
        exclude_tchq = filter(features_files, self.modelFiles)
        self.modelFiles = list(exclude_tchq)
        self.modelFiles.sort()
        modelCounter = 0
        for model in self.modelFiles:
            if '.tchq' not in model:
                self.models[modelCounter] = model
            modelCounter += 1
        nltk.download('punkt')
        nltk.download('wordnet')
        return "Model Initiated; Proceed for Prediction"

    def predict(self, inp, filepath_model, balancing_techniques):

        def removeStopWords(listOfWords):
            filtered_list = [w for w in listOfWords if not w in self.stop_words]
            lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in filtered_list])
            return lemmatized_output

        def lemmatizeRemoveStopWords(df):
            df = df.str.lower()
            df = df.replace(r'@', '', regex=True)
            df = df.replace(r'http\S+', '', regex=True).replace(r'http \S+', '', regex=True).replace(r'www\S+', '',
                                                                                                     regex=True)
            df = df.str.replace('[^\w\s]', '', regex=True)
            df = df.str.replace('\s\s+', ' ', regex=True)
            df = df.str.strip()
            df = df.apply(word_tokenize)
            df = df.apply(removeStopWords)
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
        print(f"filepath_model: {filepath_model}")
        mdl = pi.load(open(filepath_model + self.modelFiles[val], 'rb'))
        if 'chi2' in self.modelFiles[val] or 'f_classif' in self.modelFiles[val]:
            print('Feature Data')
            cnt = 0
            for tech in balancing_techniques:
                print(f'Balancing Technique: {tech}')
                if tech in self.modelFiles[val]:
                    if 'chi2' in self.modelFiles[val]:
                        # Load Feature Selection Object
                        fs = pi.load(
                            open(filepath_model + 'Feature_' + balancing_techniques[cnt] + 'fs_chi2.tchq', 'rb'))
                        print(filepath_model + 'Feature_' + balancing_techniques[cnt] + 'fs_chi2.tchq')
                        # Get New Data
                        newData = loadNewDataForPrediction(self.vocab)

                        # Transform using Count Vectorizer and Entire Vocab
                        cv_pred = CountVectorizer(vocabulary=self.vocab)
                        x_pred = cv_pred.transform(newData)

                        # Transform to Feature Selected Data
                        X_test_chi2 = fs.transform(x_pred)
                        x_pred = X_test_chi2

                        # Predict
                        y_pred_validation = mdl.predict(x_pred)
                        return (y_pred_validation)
                    else:
                        # Load Feature Selection Object
                        fs = pi.load(
                            open(filepath_model + 'Feature_' + balancing_techniques[cnt] + 'fs_f_classif.tchq', 'rb'))
                        print(filepath_model + 'Feature_' + balancing_techniques[cnt] + 'fs_f_classif.tchq')
                        # Get New Data
                        newData = loadNewDataForPrediction(self.vocab)

                        # Transform using Count Vectorizer and Entire Vocab
                        cv_pred = CountVectorizer(vocabulary=self.vocab)
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
            newData = loadNewDataForPrediction(self.vocab)
            cv_pred = CountVectorizer(vocabulary=self.vocab)
            x_pred = cv_pred.transform(newData)
            y_pred_validation = mdl.predict(x_pred)
            response = {}
            response['prediction'] = y_pred_validation[0]
            return response

    def initFunction(self, filepath_Model):
        self.init_model(filepath_model=filepath_Model)


class takeInput(BaseModel):
    sentence: str
    modelNumber: str


if __name__ == '__main__':
    filepath_Model_IM = "/models_im/"
    filepath_Model_NI = "/models_ni/"
    # filepath_Model_NI="C:/Users/shubham/SRH IT/Kinner, Maximilian (SRH Hochschule Heidelberg Student) - 06 Case Study I/02 Input_Data/03 Model/Model_Test_NI_vmdhhh/"
    # filepath_Model_NI="C:/Users/shubham/SRH IT/Kinner, Maximilian (SRH Hochschule Heidelberg Student) - 06 Case Study I/02 Input_Data/03 Model/Model_Test_NI_vmdhhh/"

    predictIdentityMotive = Prediction("09_TrainingData", "CountVectorVocabulary")
    predictIdentityMotive.initFunction(filepath_Model=filepath_Model_IM)

    # predictNationalIdentity = Prediction("09_TrainingData_Ni", "CountVectorVocabulary")
    # predictNationalIdentity.initFunction(filepath_Model=filepath_Model_NI)


    @app.route('/models_IM', methods=['GET'])
    def get_models_IM():
        with app.app_context():
            response = flask.jsonify(predictIdentityMotive.getModels())
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response


    @app.route('/models_NI', methods=['GET'])
    def get_models_NI():
        with app.app_context():
            response = flask.jsonify(predictIdentityMotive.getModels())
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response


    @app.route('/predict_id_motive', methods=['POST'])
    def predictInput():
        data = request.json
        filepath_Model = filepath_Model_IM
        balancingTechniques = ["SMOTEENN", "NearMiss", "SMOTETomek", "SMOTE", "TomekLinks"]
        inp = data
        # with app.app_context():
        response = flask.jsonify(predictIdentityMotive.predict(inp, filepath_model=filepath_Model,
                                                               balancing_techniques=balancingTechniques))

        # response.headers.add('Access-Control-Allow-Origin', '*')
        return response


    @app.route('/predict_nat_id', methods=['POST'])
    def predictInput_NI():
        data = request.json
        filepath_Model = filepath_Model_NI
        balancingTechniques = ["SMOTEENN", "NearMiss", "SMOTETomek", "SMOTE", "TomekLinks"]
        inp = data
        response = flask.jsonify(predictIdentityMotive.predict(inp, filepath_model=filepath_Model,
                                                               balancing_techniques=balancingTechniques))

        return response


    app.run(host='0.0.0.0', port=5000)
