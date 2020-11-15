from os import name
import pickle as pi

import pandas as pd
import scipy as sci
from scripts.mongoConnection import getCollection, insertCollection
from sklearn.feature_selection import (SelectFromModel, SelectKBest, chi2,
                                       f_classif)


class FeatureSelection():

    def __init__(self):
        self.balancing_technique = ["NearMiss", "SMOTEENN", "SMOTETomek", "SMOTE", "TomekLinks"]  # "ADASYN",
        self.counter = 0
        self.number_of_features = 1000
        self.filepath = "D:/OneDrive - SRH IT/Case Study 1/02 Input_Data/03 Model"


    def balancing(self):
        
        for tech in self.balancing_technique:
            # Load "y" Train Dataset
            coll_name = self.balancing_technique[self.counter] + "_y"
            y = getCollection('09_TrainingData', coll_name)
            y = y.drop(["_id", "id"], axis=1)
            print(f'"y" loaded, {self.balancing_technique[self.counter]}')

            # Load "X" Train Dataset
            X = sci.sparse.load_npz(self.filepath + '/NPZs/' + self.balancing_technique[
                self.counter] + '_x_matrix.npz')
            print(f'"x" loaded, {self.balancing_technique[self.counter]}')

            #### FEATURE SELECTION ####
            # chi2
            chi2_fs = SelectKBest(chi2, k = self.number_of_features)
            X_chi2 = chi2_fs.fit_transform(X, y)

            # Saving 'X_chi2' feature selected data after selection  using chi2
            sci.sparse.save_npz(self.filepath + '/NPZs/' + self.balancing_technique[
                self.counter] + '_x_matrix_fs_chi2.npz', X_chi2)
            print(f'Saved chi2 {self.balancing_technique[self.counter]}')

            # Save SelectKBest Technique Variable for chi2
            filename = (self.filepath + "/Models/Feature_" + self.balancing_technique[self.counter]
                        + 'fs_chi2' + '.tchq')
            pi.dump(chi2_fs, open(filename, 'wb'))
            print(f'Saved chi2 technique - {self.balancing_technique[self.counter]}')

            # f_classif
            # Saving 'X' feature selected data after selection using f_classif
            f_classif_fs = SelectKBest(f_classif, k = self.number_of_features)
            X_f_classif = f_classif_fs.fit_transform(X, y)
            sci.sparse.save_npz(self.filepath + '/NPZs/' + self.balancing_technique[
                self.counter] + '_x_matrix_fs_f_classif.npz', X_f_classif)
            print(f'Saved f_classif {self.balancing_technique[self.counter]}')

            # Save SelectKBest Technique Variable for f_classif
            filename = (self.filepath + "/Models/Feature_" + self.balancing_technique[self.counter]
                        + 'fs_f_classif' + '.tchq')
            pi.dump(f_classif_fs, open(filename, 'wb'))
            print(f'Saved fs_f_classif technique - {self.balancing_technique[self.counter]}')
            self.counter = self.counter + 1

if __name__ == 'main':
    featureSelection = FeatureSelection()
    featureSelection.balancing()