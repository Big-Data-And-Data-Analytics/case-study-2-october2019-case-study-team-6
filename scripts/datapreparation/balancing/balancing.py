import os
import sys
import threading
from collections import Counter

import pandas as pd
import scipy
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import NearMiss, TomekLinks
from datapreparation.mongoConnection import getCollection, insertCollection
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import Binarizer


##TODO Currently we are using test_train_split function from sklearn, the model shows that 2 classes are not present in the traindata. We should try StratifiedShuffleSplit, as far as i understood it, keep percentage of each class. Implement besides train_test_split

class BalancingData:
    """BalancingData class represents performance of different balancing data on 'X' and 'y' train data.
        The sequentially used functions  are:
        :func:
        1 :func: `plit_train_test()`
        2 :func: `thread1_ADASYN()`
        3 :func: `thread2_SMOTE()`
        4 :func: `thread3_SMOTEENN()`
        5 :func: `thread4_SMOTETomek()`
        6 :func: `thread5_NearMiss()`
        7 :func: `thread6_TomekLinks()`
        The file path needs to be provided where the output needs to be stored and the entire dataframe collection 
        provided as the input
    """

    def __init__(self, filepath, new_data):
        self.filepath = filepath[0]
        self.filepath_ni = filepath[1]
        self.new_data = new_data

    def split_train_test(self, test_size, random_state, stratified=False):
        """ 'X' variable is assigned 'onlyText' column and 'y' variable has the 'identityMotive'.  The 'X' and 'y' are 
        then divided into test and train data.
        The input for different balancing techniques must be in vectorised form. Thus, count vectoriser is applied on
        'X'. 
        The vocabulary is saved into CountVectorVocabulary. The results of 'X_test' is saved in the provided filepath 
        and 'y_test' is inserted into the database.

        :param new_data: complete collection
        :type new_data: dataframe
        """
        # Filters
        self.new_data = self.new_data[self.new_data["country"] != 0]
        self.new_data = self.new_data[self.new_data[
                                          "country"] != "NORTHEN IRELAND"]  ## TODO Add funtionality that it will automatically filter out countries below threshhold of 4

        X = self.new_data[['onlyText']]
        y = self.new_data[['identityMotive']]
        if not stratified:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size, train_size=1 - test_size,
                                         random_state=random_state)
            for train_index, test_index in sss.split(X, y):
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X.iloc[list(train_index)], X.iloc[list(test_index)]
                y_train, y_test = y.iloc[list(train_index)], y.iloc[list(test_index)]

        print(type(X_train))
        cv = CountVectorizer()
        cv.fit(X_train['onlyText'])
        self.X_train = cv.transform(X_train['onlyText'])
        self.y_train = y_train

        vocab = dict()
        vocab['dict'] = str(cv.vocabulary_)
        vocab = pd.DataFrame(vocab, index=['vocab', ])
        insertCollection('09_TrainingData', 'CountVectorVocabulary', vocab)

        vocab = cv.vocabulary_
        cv_test = CountVectorizer(vocabulary=vocab)
        cv_test.fit(X_test['onlyText'])
        X_test = cv_test.transform(X_test['onlyText'])

        scipy.sparse.save_npz(self.filepath + 'X_test.npz', X_test)
        y_test.insert(0, 'id', range(0, len(y_test)))
        insertCollection('09_TrainingData', 'y_test', y_test)

    def split_train_test_ni(self, test_size, random_state, stratified=False):
        """ 'X' variable is assigned 'onlyText' column and 'y' variable has the 'identityMotive'.  The 'X' and 'y' are
        then divided into test and train data.
        The input for different balancing techniques must be in vectorised form. Thus, count vectoriser is applied on
        'X'.
        The vocabulary is saved into CountVectorVocabulary. The results of 'X_test' is saved in the provided filepath
        and 'y_test' is inserted into the database.

        :param new_data: complete collection
        :type new_data: dataframe
        """

        """ self.new_data = self.new_data[self.new_data["country"] != 0]
        self.new_data["onlyTextMotive"] = self.new_data["onlyText"] + " " + self.new_data["identityMotive"]
        X = self.new_data[['onlyTextMotive']]
        y = self.new_data[['country']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
 """
        self.new_data = self.new_data[self.new_data["country"] != 0]
        self.new_data = self.new_data[self.new_data[
                                          "country"] != "NORTHEN IRELAND"]  ## TODO Add funtionality that it will automatically filter out countries below threshhold of 4
        self.new_data["onlyTextMotive"] = self.new_data["onlyText"] + " " + self.new_data["identityMotive"]
        X = self.new_data[['onlyTextMotive']]
        y = self.new_data[['country']]

        if not stratified:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size, train_size=1 - test_size,
                                         random_state=random_state)
            for train_index, test_index in sss.split(X, y):
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X.iloc[list(train_index)], X.iloc[list(test_index)]
                y_train, y_test = y.iloc[list(train_index)], y.iloc[list(test_index)]

        print(type(X_train))
        cv = CountVectorizer()
        cv.fit(X_train['onlyTextMotive'])
        self.X_train = cv.transform(X_train['onlyTextMotive'])
        self.y_train = y_train

        vocab = dict()
        vocab['dict'] = str(cv.vocabulary_)
        vocab = pd.DataFrame(vocab, index=['vocab', ])
        insertCollection('09_TrainingData_Ni', 'CountVectorVocabulary', vocab)

        vocab = cv.vocabulary_
        cv_test = CountVectorizer(vocabulary=vocab)
        cv_test.fit(X_test['onlyTextMotive'])
        X_test = cv_test.transform(X_test['onlyTextMotive'])

        scipy.sparse.save_npz(self.filepath_ni + 'X_test.npz', X_test)
        y_test.insert(0, 'id', range(0, len(y_test)))
        insertCollection('09_TrainingData_Ni', 'y_test', y_test)

    #### FIT BALANCING #### Multi Threading Part
    # Thread1 # ADASYN
    def thread1_ADASYN(self, x, y):

        """An object of the class ADASYN is created and the dataset is resampled using 'fit_resample'. The output
        'X_res'  of the resampling is saved in the provided path and the output of 'y_res' is stored into the database.

        :param x: X_train
        :param y: y_train

        :return: X_res
        :return: y_res
        """
        ada = ADASYN()
        print("ADASYN started")
        X_res, y_res = ada.fit_resample(x, y)
        print('ADASYN target variable distribution:', Counter(y_res))
        print("ADASYN fitting done")
        y_res.insert(0, 'id', range(0, len(y_res)))
        ## Save X_res
        ## Save y_res
        if 'country' in y_res.columns:
            scipy.sparse.save_npz(self.filepath_ni + 'ADASYN_x_matrix.npz', X_res)
            insertCollection('09_TrainingData_Ni', 'ADASYN_y', y_res)
        else:
            scipy.sparse.save_npz(self.filepath + 'ADASYN_x_matrix.npz', X_res)
            insertCollection('09_TrainingData', 'ADASYN_y', y_res)

        # y_res.insert(0, 'id', range(0, len(y_res)))
        # insertCollection('09_TrainingData', 'ADASYN_y', y_res)

        print("ADASYN saved and done")

    # Thread 2 # SMOTE
    def thread2_SMOTE(self, x, y):

        """An object of the class SMOTE is created and the dataset is resampled using 'fit_resample'. The output
        'X_sm'  of the resampling is saved in the provided path and the output of 'y_sm' is stored into the database.

        :param x: X_train
        :param y: y_train

        :return: X_sm
        :return: y_sm
        """
        sm = SMOTE()
        print("SMOTE started")
        X_sm, y_sm = sm.fit_resample(x, y)
        print('SMOTE target variable distribution:', Counter(y_sm))
        print("SMOTE fitting done")
        y_sm.insert(0, 'id', range(0, len(y_sm)))
        ## Save X_sm
        ## Save y_sm
        if 'country' in y_sm.columns:
            scipy.sparse.save_npz(self.filepath_ni + 'SMOTE_x_matrix.npz', X_sm)
            insertCollection('09_TrainingData_Ni', 'SMOTE_y', y_sm)
        else:
            scipy.sparse.save_npz(self.filepath + 'SMOTE_x_matrix.npz', X_sm)
            insertCollection('09_TrainingData', 'SMOTE_y', y_sm)

        print("SMOTE saved and done")

    # Thread 3 # SMOTEENN
    def thread3_SMOTEENN(self, x, y):

        """An object of the class SMOTEENN is created and the dataset is resampled using 'fit_resample'. The output
        'X_se'  of the resampling is saved in the provided path and the output of 'y_se' is stored into the database.

        :param x: X_train
        :param y: y_train

        :return: X_se
        :return: y_se
        """
        se = SMOTEENN()
        print("SMOTEENN started")
        X_se, y_se = se.fit_resample(x, y)
        print('Combined sample(SMOTEENN) target variable distribution:', Counter(y_se))
        print("SMOTEENN fitting done")
        y_se.insert(0, 'id', range(0, len(y_se)))
        ## Save X_se
        ## Save y_se
        if 'country' in y_se.columns:
            scipy.sparse.save_npz(self.filepath_ni + 'SMOTEENN_x_matrix.npz', X_se)
            insertCollection('09_TrainingData_Ni', 'SMOTEENN_y', y_se)
        else:
            scipy.sparse.save_npz(self.filepath + 'SMOTEENN_x_matrix.npz', X_se)
            insertCollection('09_TrainingData', 'SMOTEENN_y', y_se)

        print("SMOTEENN saved and done")

    # Thread 4 # SMOTETomek
    def thread4_SMOTETomek(self, x, y):

        """An object of the class SMOTETomek is created and the dataset is resampled using 'fit_resample'. The output
        'X_st'  of the resampling is saved in the provided path and the output of 'y_st' is stored into the database.

        :param x: X_train
        :param y: y_train

        :return: X_st
        :return: y_st
        """

        st = SMOTETomek()
        print("SMOTETomek started")
        X_st, y_st = st.fit_resample(x, y)
        print('Combined sample(SMOTETomek) target variable distribution:', Counter(y_st))
        print("SMOTETomek fitting done")
        y_st.insert(0, 'id', range(0, len(y_st)))
        ## Save X_st
        ## Save y_st
        if 'country' in y_st.columns:
            scipy.sparse.save_npz(self.filepath_ni + 'SMOTETomek_x_matrix.npz', X_st)
            insertCollection('09_TrainingData_Ni', 'SMOTETomek_y', y_st)
        else:
            scipy.sparse.save_npz(self.filepath + 'SMOTETomek_x_matrix.npz', X_st)
            insertCollection('09_TrainingData', 'SMOTETomek_y', y_st)

        print("SMOTETomek saved and done")

    # Thread 5 # NearMiss
    def thread5_NearMiss(self, x, y):

        """An object of the class NearMiss is created and the dataset is resampled using 'fit_resample'. The output
        'X_nm'  of the resampling is saved in the provided path and the output of 'y_nm' is stored into the database.

        :param x: X_train
        :param y: y_train

        :return: X_nm
        :return: y_nm
        """

        nm = NearMiss()
        print("NearMiss started")
        X_nm, y_nm = nm.fit_resample(x, y)
        print('Undersampled near miss target variable distribution:', Counter(y_nm))
        print("NearMiss fitting done")
        y_nm.insert(0, 'id', range(0, len(y_nm)))
        ## Save X_nm
        ## Save y_nm
        if 'country' in y_nm.columns:
            scipy.sparse.save_npz(self.filepath_ni + 'NearMiss_x_matrix.npz', X_nm)
            insertCollection('09_TrainingData_Ni', 'NearMiss_y', y_nm)
        else:
            scipy.sparse.save_npz(self.filepath + 'NearMiss_x_matrix.npz', X_nm)
            insertCollection('09_TrainingData', 'NearMiss_y', y_nm)

        print("NearMiss saved and done")

    # Thread 6 # TomekLinks
    def thread6_TomekLinks(self, x, y):

        """An object of the class TomekLinks is created and the dataset is resampled using 'fit_resample'. The output
        'X_tl'  of the resampling is saved in the provided path and the output of 'y_tl' is stored into the database.

        :param x: X_train
        :param y: y_train

        :return: X_tl
        :return: y_tl
        """

        tl = TomekLinks()
        print("TomekLinks started")
        X_tl, y_tl = tl.fit_resample(x, y)
        print('Undersampled TomekLinks target variable distribution:', Counter(y_tl))
        print("TomekLinks fitting done")
        y_tl.insert(0, 'id', range(0, len(y_tl)))
        ## Save X_tl
        if 'country' in y_tl.columns:
            scipy.sparse.save_npz(self.filepath_ni + 'TomekLinks_x_matrix.npz', X_tl)
            insertCollection('09_TrainingData_Ni', 'TomekLinks_y', y_tl)
        else:
            scipy.sparse.save_npz(self.filepath + 'TomekLinks_x_matrix.npz', X_tl)
            insertCollection('09_TrainingData', 'TomekLinks_y', y_tl)
        ## Save y_tl

        print("TomekLinks saved and done")

    def threading_function(self):

        """Different threads are created for each balancing technique having the target as the functions of 
        balancing techniques and arguments are the inputs i.e 'X_train' and 'y_train'. After the creation of threads,
        the threads are executed.
        """
        # t1 = threading.Thread(name="thread1_ADASYN", target=self.thread1_ADASYN, args=(self.X_train, self.y_train))
        t2 = threading.Thread(name="thread2_SMOTE", target=self.thread2_SMOTE, args=(self.X_train, self.y_train))
        t3 = threading.Thread(name="thread3_SMOTEENN", target=self.thread3_SMOTEENN, args=(self.X_train, self.y_train))
        t4 = threading.Thread(name="thread4_SMOTETomek", target=self.thread4_SMOTETomek,
                              args=(self.X_train, self.y_train))
        t5 = threading.Thread(name="thread5_NearMiss", target=self.thread5_NearMiss, args=(self.X_train, self.y_train))
        t6 = threading.Thread(name="thread6_TomekLinks", target=self.thread6_TomekLinks,
                              args=(self.X_train, self.y_train))

        # t1.start()
        t2.start()
        t3.start()
        t4.start()
        t5.start()
        t6.start()

        # t1.join()
        t2.join()
        t3.join()
        t4.join()
        t5.join()
        t6.join()

        print("Balancing Done!")


if __name__ == "__main__":
    df_source_collection = getCollection('08_PreTrain', 'train_data')
    # df_source_collection = getCollection('Train', 'train')
    
    # filepath = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/NPZs/"
    # filepath_ni = "C:/Users/shubham/SRH IT/Kinner, Maximilian (SRH Hochschule Heidelberg Student)" \
    #               " - 06 Case Study I/02 Input_Data/03 Model/NPZs_ni/"
    # filepath = "C:/Users/mavis/Documents/GitHub/case-study-2-october2019-case-study-team-6/NPZs/"
    # filepath_ni = "C:/Users/mavis/Documents/GitHub/case-study-2-october2019-case-study-team-6/NPZs_ni/"

    filepath = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/NPZs/"
    filepath_ni = "D:/OneDrive - SRH IT/06 Case Study I/02 Input_Data/03 Model/NPZs_ni/"

    balancing_input = BalancingData((filepath, filepath_ni), df_source_collection)
    balancing_input.split_train_test(test_size=0.25, random_state=69, stratified=True)
    # balancing_input.split_train_test_ni(test_size=0.25, random_state=69, stratified=True)
    #balancing_input.split_train_test(test_size=0.25, random_state=69) # Stratified as default false
    balancing_input.threading_function()
