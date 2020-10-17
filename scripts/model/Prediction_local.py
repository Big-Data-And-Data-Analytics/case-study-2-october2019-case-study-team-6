import pickle as pi
from os import listdir
from os.path import isfile, join
import pandas as pd
import yaml
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Requirements
"""
Do not need the test data here.
    1. Load Models - Load the correct model based if fs selected or not
    2. Load Feature Selection Data - load the SelectKBest object for correct features
    3. Load Vocabulary Data - for correct vocabulary
"""

# MongoDB connection
client = MongoClient('localhost', 27017)
db = client['09_TrainingData']

# Filepath
filepath = "model/01 Models/Model_Input_Files/"

# Get the Vocabulary
col = db.CountVectorVocabulary
cv_vocab_1 = pd.DataFrame(list(col.find()))
l1 = cv_vocab_1['dict'].to_list()
vocab = yaml.load(l1[0])  # Get first element

# Balancing technique
balancing_technique = [  # "ADASYN",
    "SMOTEENN",
    "NearMiss",
    "SMOTETomek",
    "SMOTE",
    "TomekLinks"]

# Filenames for the model selection/prediction
modelsFilepath = filepath
modelFiles = [f for f in listdir(modelsFilepath) if isfile(join(modelsFilepath, f))]


# Filter FileNamesList for Models
def features_files(model):
    return '.tchq' not in model


# Filtering Data using features_files
exclude_tchq = filter(features_files, modelFiles)
modelFiles = list(exclude_tchq)
modelFiles.sort()
modelCounter = 0


# Load Vocab for Non Feature Selected Models
def loadNewDataForPrediction(vocab):
    validation_set = list()
    while True:
        val = input('Enter n number of validation texts, type "EXIT" to exit\n')
        if val.lower() == 'exit':
            break
        else:
            # print(val)
            validation_set.append(val)
    valid = pd.Series(validation_set)
    valid = lemmatizeRemoveStopWords(valid)
    return valid

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

client = MongoClient('localhost', 27017)
db = client['06_NationalIdentity_Translated']

post = db.ni_post_translated
post = pd.DataFrame(list(post.find({})))
comment = db.ni_comment_translated
comment = pd.DataFrame(list(comment.find({})))
subcomment = db.ni_subcomment_translated
subcomment = pd.DataFrame(list(subcomment.find({})))

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
    #df = df.dropna(subset=)
    df=df.apply(word_tokenize)
    df=df.apply(removeStopWords)
    # del df['_id']
    return df


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
        mdl = pi.load(open(modelsFilepath + modelFiles[val], 'rb'))
        print('Model Loaded')
        # Load Feature Data / Normal Data
        if 'fs' in modelFiles[val]:
            print('Feature Data')
            cnt = 0
            for tech in balancing_technique:
                print(f'Balancing Technique: {tech}')
                if tech in modelFiles[val]:
                    if 'chi2' in modelFiles[val]:
                        # Load Feature Selection Object
                        fs = pi.load(open(filepath + 'Feature_' + balancing_technique[cnt] + 'fs_chi2.tchq', 'rb'))
                        print(filepath + 'Feature_' + balancing_technique[cnt] + 'fs_chi2.tchq')
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
                        fs = pi.load(open(filepath + 'Feature_' + balancing_technique[cnt] + 'fs_chi2.tchq', 'rb'))
                        print(filepath + 'Feature_' + balancing_technique[cnt] + 'fs_chi2.tchq')

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
