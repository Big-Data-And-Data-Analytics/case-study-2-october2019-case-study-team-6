import pandas as pd
import scipy
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from pymongo import MongoClient
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split
import threading
from sklearn.model_selection import train_test_split

# Filepath
#filepath = '/home/shubham/PycharmProjects/Case Study 1  Extras/NPZs/'
filepath = "/Users/Shared/Relocated Items/Security/Abhinav's Documents/SRH IT/Kinner, Maximilian (SRH Hochschule Heidelberg Student) - 06 Case Study I/02 Input_Data/03 Model/NPZs/"

# Source Collection
client = MongoClient('localhost', 27017)
db = client['08_PreTrain']
train_data = db.train_data
df_train = pd.DataFrame(list(train_data.find({})))

# Target Collection
client = MongoClient('localhost', 27017)
db = client['09_TrainingData']

print(df_train['identityMotive'].value_counts())
X = df_train[['onlyText']]
y = df_train['identityMotive']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)

# Test Data
# Y
coll = db.test_y
coll.insert_many(y_test.to_frame().to_dict("records"))
# X
coll = db.test_x
coll.insert_many(X_test.to_dict("records"))


# Train - Vectorization
cv = CountVectorizer()
cv.fit(X_train['onlyText'])
X_train = cv.transform(X_train['onlyText'])


# Insert Vector Vocabulary for prediction
coll = db.CountVectorVocabulary
coll.delete_many({})
vocab = dict()
vocab['dict'] = str(cv.vocabulary_)
coll.insert_one(vocab)

# Test - Vectorization
vocab = cv.vocabulary_
cv_test = CountVectorizer(vocabulary=vocab)
cv_test.fit(X_test['onlyText'])
X_test = cv_test.transform(X_test['onlyText'])

# Save test data
## X
scipy.sparse.save_npz(filepath + 'X_test.npz', X_test)
## Y
y_test = y_test.to_frame()
y_test.insert(0, 'id', range(0, len(y_test)))
coll = db.y_test
coll.delete_many({})
coll.insert_many(y_test.to_dict("records"))

feature_names = cv.get_feature_names()
coll = db.feature_names
coll.delete_many({})
coll.insert_many(y_test.to_dict("records"))

#### INSTATIATE BALANCING TECHNIQUES
ada = ADASYN()
sm = SMOTE()
se = SMOTEENN()
st = SMOTETomek()
nm = NearMiss()
tl = TomekLinks()


#### FIT BALANCING #### Multi Threading Part
# Thread1 # ADASYN
def thread1_ADASYN(x, y):
    print("ADASYN started")
    X_res, y_res = ada.fit_resample(x, y)
    print('ADASYN target variable distribution:', Counter(y_res))
    print("ADASYN fitting done")
    ## Save X_res
    scipy.sparse.save_npz(filepath + 'ADASYN_x_matrix.npz', X_res)
    ## Save y_res
    y_res = y_res.to_frame()
    y_res.insert(0, 'id', range(0, len(y_res)))
    coll = db.ADASYN_y
    coll.delete_many({})
    coll.insert_many(y_res.to_dict("records"))
    print("ADASYN saved and done")


# Thread 2 # SMOTE
def thread2_SMOTE(x, y):
    print("SMOTE started")
    X_sm, y_sm = sm.fit_resample(x, y)
    print('SMOTE target variable distribution:', Counter(y_sm))
    print("SMOTE fitting done")
    ## Save X_sm
    scipy.sparse.save_npz(filepath + 'SMOTE_x_matrix.npz', X_sm)
    ## Save y_sm
    y_sm = y_sm.to_frame()
    y_sm.insert(0, 'id', range(0, len(y_sm)))
    coll = db.SMOTE_y
    coll.delete_many({})
    coll.insert_many(y_sm.to_dict("records"))
    print("SMOTE saved and done")


# Thread 3 # SMOTEENN
def thread3_SMOTEENN(x, y):
    print("SMOTEENN started")
    X_se, y_se = se.fit_resample(x, y)
    print('Combined sample(SMOTEENN) target variable distribution:', Counter(y_se))
    print("SMOTEENN fitting done")
    ## Save X_se
    scipy.sparse.save_npz(filepath + 'SMOTEENN_x_matrix.npz', X_se)
    ## Save y_se
    y_se = y_se.to_frame()
    y_se.insert(0, 'id', range(0, len(y_se)))
    coll = db.SMOTEENN_y
    coll.delete_many({})
    coll.insert_many(y_se.to_dict("records"))
    print("SMOTEENN saved and done")


# Thread 4 # SMOTETomek
def thread4_SMOTETomek(x, y):
    print("SMOTETomek started")
    X_st, y_st = st.fit_resample(x, y)
    print('Combined sample(SMOTETomek) target variable distribution:', Counter(y_st))
    print("SMOTETomek fitting done")
    ## Save X_st
    scipy.sparse.save_npz(filepath + 'SMOTETomek_x_matrix.npz', X_st)
    ## Save y_st
    y_st = y_st.to_frame()
    y_st.insert(0, 'id', range(0, len(y_st)))
    coll = db.SMOTETomek_y
    coll.delete_many({})
    coll.insert_many(y_st.to_dict("records"))
    print("SMOTETomek saved and done")


# Thread 5 # NearMiss
def thread5_NearMiss(x, y):
    print("NearMiss started")
    X_nm, y_nm = nm.fit_resample(x, y)
    print('Undersampled near miss target variable distribution:', Counter(y_nm))
    print("NearMiss fitting done")
    ## Save X_nm
    scipy.sparse.save_npz(filepath + 'NearMiss_x_matrix.npz', X_nm)
    ## Save y_nm
    y_nm = y_nm.to_frame()
    y_nm.insert(0, 'id', range(0, len(y_nm)))
    coll = db.NearMiss_y
    coll.delete_many({})
    coll.insert_many(y_nm.to_dict("records"))
    print("NearMiss saved and done")


# Thread 6 # TomekLinks
def thread6_TomekLinks(x, y):
    print("TomekLinks started")
    X_tl, y_tl = tl.fit_resample(x, y)
    print('Undersampled TomekLinks target variable distribution:', Counter(y_tl))
    print("TomekLinks fitting done")
    ## Save X_tl
    scipy.sparse.save_npz(filepath + 'TomekLinks_x_matrix.npz', X_tl)
    ## Save y_tl
    y_tl = y_tl.to_frame()
    y_tl.insert(0, 'id', range(0, len(y_tl)))
    coll = db.TomekLinks_y
    coll.delete_many({})
    coll.insert_many(y_tl.to_dict("records"))
    print("TomekLinks saved and done")


t1 = threading.Thread(name="thread1_ADASYN", target=thread1_ADASYN, args=(X_train, y_train))
t2 = threading.Thread(name="thread2_SMOTE", target=thread2_SMOTE, args=(X_train, y_train))
t3 = threading.Thread(name="thread3_SMOTEENN", target=thread3_SMOTEENN, args=(X_train, y_train))
t4 = threading.Thread(name="thread4_SMOTETomek", target=thread4_SMOTETomek, args=(X_train, y_train))
t5 = threading.Thread(name="thread5_NearMiss", target=thread5_NearMiss, args=(X_train, y_train))
t6 = threading.Thread(name="thread6_TomekLinks", target=thread6_TomekLinks, args=(X_train, y_train))

t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()

print("Balancing Done!")
