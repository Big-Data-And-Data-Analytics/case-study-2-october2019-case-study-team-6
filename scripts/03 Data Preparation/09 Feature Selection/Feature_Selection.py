import pandas as pd
import scipy as sci
import pickle


from pymongo import MongoClient
import pickle as pi
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

# from sklearn.feature_selection import mutual_info_classif

# Parameters
balancing_technique = ["NearMiss", "SMOTEENN", "SMOTETomek", "SMOTE", "TomekLinks"]  # "ADASYN",
counter = 0
number_of_features = 1000

# Filepath
#filepath = '/home/shubham/PycharmProjects/Case Study 1  Extras'
filepath = "D:/OneDrive - SRH IT/Case Study 1/02 Input_Data/03 Model"

# Source Collection
client = MongoClient('localhost', 27017)
db = client['09_TrainingData']

for tech in balancing_technique:
    # Load "y" Train Dataset
    coll_name = balancing_technique[counter] + "_y"
    y = db[coll_name]
    y = pd.DataFrame(list(y.find()))
    y = y.drop(["_id", "id"], axis=1)
    print(f'"y" loaded, {balancing_technique[counter]}')

    # Load "X" Train Dataset
    X = sci.sparse.load_npz(filepath + '/NPZs/' + balancing_technique[
        counter] + '_x_matrix.npz')
    print(f'"x" loaded, {balancing_technique[counter]}')

    #### FEATURE SELECTION ####
    # chi2
    chi2_fs = SelectKBest(chi2, k=number_of_features)
    X_chi2 = chi2_fs.fit_transform(X, y)

    # Saving 'X_chi2' feature selected data after selection  using chi2
    sci.sparse.save_npz(filepath + '/NPZs/' + balancing_technique[
        counter] + '_x_matrix_fs_chi2.npz', X_chi2)
    print(f'Saved chi2 {balancing_technique[counter]}')

    # Save SelectKBest Technique Variable for chi2
    filename = (filepath + "/Models/Feature_" + balancing_technique[counter]
                + 'fs_chi2' + '.tchq')
    pi.dump(chi2_fs, open(filename, 'wb'))
    print(f'Saved chi2 technique - {balancing_technique[counter]}')

    # f_classif
    # Saving 'X' feature selected data after selection using f_classif
    f_classif_fs = SelectKBest(f_classif, k=number_of_features)
    X_f_classif = f_classif_fs.fit_transform(X, y)
    sci.sparse.save_npz(filepath + '/NPZs/' + balancing_technique[
        counter] + '_x_matrix_fs_f_classif.npz', X_f_classif)
    print(f'Saved f_classif {balancing_technique[counter]}')

    # Save SelectKBest Technique Variable for f_classif
    filename = (filepath + "/Models/Feature_" + balancing_technique[counter]
                + 'fs_f_classif' + '.tchq')
    pi.dump(f_classif_fs, open(filename, 'wb'))
    print(f'Saved fs_f_classif technique - {balancing_technique[counter]}')
    counter = counter + 1

print("Feature Selection Done")
