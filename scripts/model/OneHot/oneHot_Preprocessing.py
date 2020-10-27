import pandas as pd
from pymongo import MongoClient
import yaml
from sklearn.feature_extraction.text import CountVectorizer
import scipy

filepath = 'D:/OneDrive - SRH IT/Case Study 1/02 Input_Data/03 Model/OneHotPreprocessing'

# Load CSV
d = pd.read_csv(filepath + '/X_test.csv')
# MongoDB connection
client = MongoClient('localhost', 27017)
db = client['09_TrainingData']

# Get the Vocabulary
col = db.CountVectorVocabulary
cv_vocab_1 = pd.DataFrame(list(col.find()))
l1 = cv_vocab_1['dict'].to_list()
vocab = yaml.load(l1[0]) # Get first element
cv_pred = CountVectorizer(vocabulary=vocab)
x_pred = cv_pred.transform(d['x'])
scipy.sparse.save_npz(filepath + '/OneHotPredictionTest.npz', x_pred)

#Continuing work