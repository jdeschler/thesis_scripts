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
from nmp_helpers import *

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
    sample = get_subsample(args.Sessions, args.demos, n = n, party = True)
    sample = classify_party(sample, threshold = threshold, two_party = args.twoparty)
    fit_models(sample)
    exit(0) 

if __name__ == '__main__':
    main()
