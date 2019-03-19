# Jack Deschler
# The New Machine Politics
# helper_functions.py
#  contains functions used commonly in modeling code
# 3 sections:
#  - Confusion Matrices and Accuracy
#  - Party Classification
#  - Subsampling
#  - Matrix Transformation
# Dependencies:
import numpy as np
import pandas as pd
import scipy as sp
import gc
import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

####### Confusion Matrices and Accuracy ######

# classification_accuracy(y_true, y_pred)
#  returns the percentage matched between y_pred and y_true
#  which is the overall classfication accuracy of a model
def classification_accuracy(y_true, y_pred):
    plot_conf_mat(y_true, y_pred)
    total_missed = 0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            total_missed += 1
    return 1 - (total_missed/len(y_true))

# conf_mat(y_real, y_pred)
#  prints the confusion matrix to the console
#  only works for 2-class classifiers
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

# Note: plot_conf_mat not included here as the classes and sizes change
#  depending on which QOI is being modeled (race vs party)


####### Party Classification ########
# classify_party
#  Adds column to dataframe for party classification
#  'democrat': 1 if classified as a democrat, 0 if not
#  Inputs:
#   threshold: value at which D_pct[_2p] must be to classify as D
#       default set at 0.8 as per Eitan D. Hersh in 'Hacking the Electorate'
#   2p: boolean, True if to use 2 party values
#  Returns:
#   dataframe with the 'democrat' column
def classify_party(df, threshold = 0.8, two_party = True):
    comp_col = 'D_pct_2p' if two_party else 'D_pct'
    df['democrat'] = df.apply(lambda row: 1 if row[comp_col] > threshold else 0, axis = 1)
    return df

####### Subsampling ########
# get_subsample(sessions, demos, n)
#  returns a subsample of the sessions dataframe of size n, if n is set
#  samples by machine number, so we get n users
#  if n is set, uses subsample() to weight race a certain way (see below)
#  if n is not set, simply uses the entire set of demographics
#  calls respecitve transform mat, collects garbage and returns transformed matrix 
def get_subsample(sessions, demos, n = -1, targets = [{1: .766, 2: .134, 3: .058, 5: .042}], party = False):
    # get the subsample
    demos = pd.read_csv(demos)
    demos_sample = demos if n == -1 else subsample(n, demos, targets = targets)
    subsample_numbers = demos_sample['machine_id']

    # read in the actual file and clean it here
    df_full = pd.read_csv(sessions)
    sample = df_full[df_full['machine_id'].isin(subsample_numbers)]
    df_final = transform_mat_party if party else transform_mat(sample)

    # force memory garbage collection
    demos = None
    df_full = None
    sample = None
    gc.collect()
    return df_final

# splits DataFrame into testing and training sets
def split_data(df, threshold = 0.8):
    msk = np.random.rand(len(df)) < threshold

    data_train = df[msk]
    data_test = df[~msk]

    return (data_train, data_test)

# subsample: returns a subsample of a dataframe weighted by certain targets
#  IN: 
#    n:       number of rows to be in subsample
#    df:      dataframe to sample from
#    demos:   list of demographics to weight on (comScore data column names)
#    targets: list of dictionaries of weights (comScore code: weight)
#       defaults set to bet census weights
#  OUT:
#    final dataframe subsample
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

###### Matrix Transformation #######
# transforms comscore sessions into usable matrix
# transform_mat for race, transform_mat_party for party 
def transform_mat(df):
    # extract demographics, save separately for re-linking later
    df_demos = df[['machine_id', 'hoh_most_education', 'census_region',
                   'household_size', 'hoh_oldest_age', 'household_income',
                   'children', 'racial_background','connection_speed',
                   'country_of_origin','zip_code']]
    df_demos = df_demos.drop_duplicates('machine_id')
    # drop columns we don't need, and demos, bc we already saved those
    df = df.drop(['site_session_id', 'domain_id', 'ref_domain_name', 'duration',
         'tran_flg', 'hoh_most_education', 'census_region',
         'household_size', 'hoh_oldest_age', 'household_income',
         'children', 'racial_background','connection_speed',
         'country_of_origin','zip_code'], axis = 1)
    # use pivot to get the data in the format we want
    df = df.pivot_table(index='machine_id', columns='domain_name', values='pages_viewed', aggfunc = lambda x: np.sum(x).astype(bool).astype(int))
    df = pd.DataFrame(df.to_records())
    df = df.fillna(0)
    print("{} columns in the transformed matrix".format(len(list(df))))
    # merge demographics back, and write final
    final = df.merge(df_demos, on = 'machine_id')
    return final


def transform_mat_party(df):
    # extract demographics, save separately for re-linking later
    # key differnece is we must include the imputed columns (vf_k, etc.)
    df_demos = df[['machine_id', 'hoh_most_education', 'census_region',
                   'household_size', 'hoh_oldest_age', 'household_income',
                   'children', 'racial_background','connection_speed',
                   'country_of_origin','zip_code', 'D_pct','D_pct_2p',
                   'vf_k', 'vf_k_2p']]
    df_demos = df_demos.drop_duplicates('machine_id')
    # drop columns we don't need, and demos, bc we already saved those
    df = df.drop(['hoh_most_education', 'census_region',
         'household_size', 'hoh_oldest_age', 'household_income',
         'children', 'racial_background','connection_speed',
         'country_of_origin','zip_code'], axis = 1)
    # use pivot to get the data in the format we want
    df = df.pivot_table(index='machine_id', columns='domain_name', values='pages_viewed', aggfunc = lambda x: np.sum(x).astype(bool).astype(int))
    df = pd.DataFrame(df.to_records())
    df = df.fillna(0)
    print("{} columns in the transformed matrix".format(len(list(df))))
    # merge demographics back, and write final
    final = df.merge(df_demos, on = 'machine_id')
    return final
