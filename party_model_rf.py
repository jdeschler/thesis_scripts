###############################################
# Jack Deschler, Senior Thesis
#   code to fit random forest models on comScore data
#   in order to predict partisanship from browsing history
#################################################

import argparse
import numpy as np
import pandas as pd
import scipy as sp
import gc
import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
parser = argparse.ArgumentParser()
from helper_functions import *

# transforms comscore sessions into usable matrix  
def transform_mat(df):
    # extract demographics, save separately for re-linking later
    print(list(df))
    df_demos = df[['machine_id', 'hoh_most_education', 'census_region',
                   'household_size', 'hoh_oldest_age', 'household_income',
                   'children', 'racial_background','connection_speed',
                   'country_of_origin','zip_code', 'vf_k', 'vf_k_2p',
                   'D_pct', 'D_pct_2p']]
    df_demos = df_demos.drop_duplicates('machine_id')
    # drop columns we don't need, and demos, bc we already saved those
    df = df.drop(['hoh_most_education', 'census_region',
         'household_size', 'hoh_oldest_age', 'household_income',
         'children', 'racial_background','connection_speed',
         'country_of_origin','zip_code', 'vf_k', 'vf_k_2p',
         'D_pct', 'D_pct_2p'], axis = 1)
    # use pivot to get the data in the format we want
    df = df.pivot_table(index='machine_id', columns='domain_name', values='pages_viewed', aggfunc = np.sum)
    df = pd.DataFrame(df.to_records())
    df = df.fillna(0)
    # merge demographics back, and return final
    final = df.merge(df_demos, on = 'machine_id')
    return final 

def fit_rf_model(df_train, df_test, demos = ['machine_id', 'hoh_most_education', 'census_region',
                   'household_size', 'hoh_oldest_age', 'household_income',
                   'children','connection_speed',
                   'country_of_origin', 'racial_background', 'zip_code', 'vf_k', 'vf_k_2p', 'D_pct', 'D_pct_2p'], response = 'democrat', n_estimators = 256, max_depth = 48):
    
    rf_model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)
    print("fitting random forest model")
    preds = list(set(list(df_train)) - set([response] + demos))
    rf_model.fit(df_train[preds], df_train[response])
        
    y_hat = rf_model.predict(df_test[preds])
    print("Overall accuracy: {}".format(classification_accuracy(df_test['democrat'].values, y_hat)))
    conf_mat(df_test['democrat'].values, y_hat)
    # write feature importances console, top 10
    impts = rf_model.feature_importances_
    indices = np.argsort(impts)[::-1]
    print("Top 10 Features")
    for i in range(0,10):
        print('{}. {}'.format(i+1, preds[indices[i]]))
    return rf_model

# create and print our confusion matrix!
# adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_conf_mat(y_true, y_pred, classes=['Nondemocrat','Democrat'],
                  normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=20)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confmat.png')
    plt.clf()

def fit_models(df_final):
    train, test = split_data(df_final)
    rf_model = fit_rf_model(train, test)
    return {'rf': rf_model}

def main():
    parser.add_argument("-tp", "--twoparty",  help='run on 2 party stats', action='store_true')
    parser.add_argument("Sessions", help = "comScore Sessions for NC")
    parser.add_argument("demos", help = "NC users demographic data")
    parser.add_argument("-n", type=int, help = 'number to subsample')
    parser.add_argument("-l", "--threshold", type=float, help = 'threshold if not 0.8 for D classification')
    args = parser.parse_args() 
    n = -1 if not args.n else args.n
    threshold = 0.8 if not args.threshold else args.threshold
    sample = get_subsample(args.Sessions, args.demos, n = n)
    sample = classify_party(sample, threshold = threshold, two_party = args.twoparty)
    fit_models(sample)
    exit(0) 

if __name__ == '__main__':
    main()
