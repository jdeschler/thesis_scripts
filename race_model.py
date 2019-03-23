###############################################
# Jack Deschler, Senior Thesis
#   code to fit random forest models on comScore data
#   in order to predict race from browser history
#################################################

import numpy as np
import pandas as pd
import scipy as sp
import argparse
import gc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
parser = argparse.ArgumentParser()
from nmp_helpers import *

# Fit the random Forest model!
def fit_rf_model(df_train, df_test, demos = ['machine_id', 'hoh_most_education', 'census_region',
                   'household_size', 'hoh_oldest_age', 'household_income',
                   'children','connection_speed',
                   'country_of_origin','zip_code'], response = 'racial_background', n_estimators = 256, max_depth = 48):
    
    rf_model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)
    print("fitting random forest model")
    preds = list(set(list(df_train)) - set([response] + demos))
    
    rf_model.fit(df_train[preds], df_train[response])
        
    y_hat = rf_model.predict(df_test[preds])
    print("Overall accuracy: {}".format(classification_accuracy(df_test['racial_background'].values, y_hat)))
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
    parser.add_argument('-n', type=int, help='size of subsample')
    parser.add_argument('Sessions', help='comScore Sessions file')
    parser.add_argument('demos', help='comScore demographics file')
    args = parser.parse_args()
    n = 20000 if not args.n else args.n
    sample = get_subsample(args.Sessions, args.demos, n=n)
    fit_models(sample)
    exit(0) 

if __name__ == '__main__':
    main()
