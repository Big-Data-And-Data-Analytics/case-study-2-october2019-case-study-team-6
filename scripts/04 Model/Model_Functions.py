import pickle as pi
import matplotlib.pyplot as plt
from sklearn import metrics

def save_model(use_fs_data, accurcay_score, acc_thresh, filepath, balancing_technique_counter, score_function, model):
        if use_fs_data == False:
            if accurcay_score > acc_thresh:
                filename = filepath + "Case Study 1/02 Input_Data/03 Model/Models/LogiReg_" + balancing_technique[counter] + "_" + str(acc) + ".model"
                pi.dump(model, open(filename, 'wb'))
            else:
                print("Model accuracy below " + acc_thresh + ", model not saved")
        else:
            if score_function == "chi2":
                filename = filepath + "Case Study 1/02 Input_Data/03 Model/Models/LogiReg_" + balancing_technique[counter] + "_" + str(acc) + "_fs_chi2.model"
                pi.dump(model, open(filename, 'wb'))
            else:
                filename = filepath + "Case Study 1/02 Input_Data/03 Model/Models/LogiReg_" + balancing_technique[counter] + "_" + str(acc) + "_fs_f_classif.model"
                pi.dump(model, open(filename, 'wb'))

def save_confusion_matrix(model, x_pred, y_test, my_tags):
    ### Evaluation - Confusion Matrix Plot
    titles_options = [("Confusion matrix, without normalization " + "Acc: " + str(metrics.accuracy_score(y_true, y_pred)), None), ("Normalized confusion matrix " + "Acc: " + str(metrics.accuracy_score(y_true, y_pred)), 'true')]
    conf_counter = 0

    for title, normalize in titles_options:
        disp = metrics.plot_confusion_matrix(model, x_pred, y_test, display_labels=my_tags, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        plt.xticks(rotation=90)
        if conf_counter == 0:
            plt.savefig('LogisticRegression_Confusion_matrix_without_normalization' + "_" + balancing_technique[counter] + '.png', bbox_inches='tight')
        else:
            plt.savefig('LogisticRegression_Confusion_matrix_with_normalization' + "_" + balancing_technique[counter] + '.png', bbox_inches='tight')

        conf_counter = conf_counter + 1

    plt.show()
