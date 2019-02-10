###############################################
# Jack Deschler, Senior Thesis
#   code to fit models on comScore data
#   in order to predict race from browser history
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
#from joblib import dump, load
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
parser = argparse.ArgumentParser()

# from cleaning.py, fwiw this is the clunky version
def transform_mat(df):
    # extract demographics, save separately for re-linking later
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

# Helper functions from CS109a final
def conf_mat(y_real, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()
    
    print("\nCONFUSION MATIX:")
    print("{0:6} {1:6} {2:6}".format("", "pred +", "pred -"))
    print("{0:6} {1:6} {2:6}".format("true +", tp, fn))
    print("{0:6} {1:6} {2:6}".format("true -", fp, tn))
    
    if tp == 0 and fp == 0:
        tpr = 0
    else:
        tpr = tp / (tp + fp)
        
    if tn == 0 and fn == 0:
        tnr = 0
    else:
        tnr = tn / (tn + fn)
    
    print("\nTRUE POSITIVE:", tpr)
    print("TRUE NEGATIVE:", tnr)


def split_data(df, threshold = 0.8):
    msk = np.random.rand(len(df)) < threshold

    data_train = df[msk]
    data_test = df[~msk]
    
    return (data_train, data_test)

def classification_accuracy(y_true, y_pred):
    plot_conf_mat(y_true, y_pred)
    total_missed = 0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            total_missed += 1
    return 1 - (total_missed/len(y_true))

def fit_rf_model(df_train, df_test, demos = ['machine_id', 'hoh_most_education', 'census_region',
                   'household_size', 'hoh_oldest_age', 'household_income',
                   'children','connection_speed',
                   'country_of_origin','zip_code', 'vf_k', 'vf_k_2p', 'D_pct', 'D_pct_2p'], response = 'democrat', n_estimators = 256, max_depth = 48):
    
    rf_model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)
    print("fitting random forest model")
    preds = list(set(list(df_train)) - set([response] + demos))
    
    rf_model.fit(df_train[preds], df_train[response])
        
    y_hat = rf_model.predict(df_test[preds])
    print("Overall accuracy: {}".format(classification_accuracy(df_test['democrat'].values, y_hat)))
    conf_mat(df_test['democrat'].values, y_hat)
    return rf_model

# create and print our confusion matrix!
# adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_conf_mat(y_true, y_pred, classes=['Democrat','Nondemocrat'],
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
    plt.xticks(tick_marks, classes, rotation=45)
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


############################################################################
# subsample: returns a subsample of a dataframe weighted by certain targets
#  IN: 
#    n:       number of rows to be in subsample
#    df:      dataframe to sample from
#    demos:   list of demographics to weight on (comScore data column names)
#    targets: list of dictionaries of weights (comScore code: weight)
#  OUT:
#    final dataframe subsample
############################################################################

# TODO: is this an ok way? just picking n*w_k instead of how you would think by generating a random every time

def subsample(n, df, demos = ['racial_background'], targets = [{1: .766, 2: .134, 3: .058, 5: .042}]):
    # start with code for just one demographic axis
    demo = demos[0]
    target = targets[0]
    keys = list(target.keys())
    dfs = {k: df[df[demo] == k] for k in keys}
    
    cols = list(df)
    agg = pd.DataFrame(columns = cols)
    
    for k in keys:
        n_k = round(target[k] * n) 
        agg = agg.append(dfs[k].sample(n_k))
    return agg

################################################################
# classify_party
#  Adds column to dataframe for party classification
#  'democrat': 1 if classified as a democrat, 0 if not
#  Inputs:
#   threshold: value at which D_pct[_2p] must be to classify as D
#       default set at 0.8 as per Eitan D. Hersh in 'Hacking the Electorate'
#   2p: boolean, True if to use 2 party values
#  Returns:
#   dataframe with the 'democrat' column
#################################################################
def classify_party(df, threshold = 0.8, two_party = True):
    comp_col = 'D_pct_2p' if two_party else 'D_pct'
    df['democrat'] = df.apply(lambda row: 1 if row[comp_col] > threshold else 0, axis = 1) 
    return df

def get_subsample(sessions, demos, n = -1):
    # get the subsample
    demos = pd.read_csv(demos)
    demos_sample = demos if n == -1 else subsample(n, demos)
    subsample_numbers = demos_sample['machine_id']

    # read in the actual file and clean it here
    df_full = pd.read_csv(sessions)
    sample = df_full[df_full['machine_id'].isin(subsample_numbers)]
    df_final = transform_mat(sample)

    # force memory garbage collection
    demos = None
    df_full = None
    sample = None
    gc.collect()
    return df_final

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
