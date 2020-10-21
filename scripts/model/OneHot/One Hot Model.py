for tech in balancing_technique:
    # Load "y"
    coll_name = balancing_technique[counter] + "_y"
    y = db[coll_name]
    y = pd.DataFrame(list(y.find()))
    y = y.drop(["_id", "id"], axis=1)
    print(f'"y" loaded, {balancing_technique[counter]}')

class sklearn.preprocessing.OneHotEncoder(*, categories='auto', drop=None, sparse=True, dtype=<class 'numpy.float64'>, handle_unknown='error')

# Load "X"
    if not use_fs_data:
        X = sc.sparse.load_npz(filepath + '/NPZs/' + balancing_technique[counter] +
                               '_x_matrix.npz')
        print(f'"x" loaded, {balancing_technique[counter]}')
    else:
        if score_function == "chi2":
            X = sc.sparse.load_npz(filepath + '/NPZs/' +
                                   balancing_technique[counter] + '_x_matrix_fs_chi2.npz')
            print(f'"x" feature selected with chi2 loaded, {balancing_technique[counter]}')
            fs = pi.load(open(filepath + '/Models/Feature_'
                              + balancing_technique[counter] + 'fs_chi2.tchq', 'rb'))
            print(balancing_technique[counter])
            X_test_chi2 = fs.transform(X_test)
            x_pred = X_test_chi2
        else:
            X = sc.sparse.load_npz(filepath + '/NPZs/' + balancing_technique[counter]
                                   + '_x_matrix_fs_f_classif.npz')
            print(f'"x" feature selected with f_classif loaded, {balancing_technique[counter]}')
            print(balancing_technique[counter])

            fs = pi.load(open(filepath + '/Models/Feature_' + balancing_technique[counter] + 'fs_f_classif.tchq', 'rb'))
            X_test_f_classif = fs.transform(X_test)
            x_pred = X_test_f_classif
