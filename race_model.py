###############################################
# Jack Deschler, Senior Thesis
#   code to fit models on comScore data
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
#from joblib import dump, load
import itertools
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
parser = argparse.ArgumentParser()

# from cleaning.py, fwiw this is the clunky version
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
    df = df.pivot_table(index='machine_id', columns='domain_name', values='pages_viewed', aggfunc = np.sum)
    df = pd.DataFrame(df.to_records())
    df = df.fillna(0)
    print("{} columns in the transformed matrix".format(len(list(df))))
    # merge demographics back, and write final
    final = df.merge(df_demos, on = 'machine_id')
    return final 

# create and print our confusion matrix!
# adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_conf_mat(y_true, y_pred, classes=['White','Black','Asian','Other'],
                  normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    classes = ['White', 'Nonwhite']
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

# Helper functions from CS109a final
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

def fit_log_model(df_train, df_test, response = 'racial_background', demos = ['machine_id', 'hoh_most_education', 'census_region',
                   'household_size', 'hoh_oldest_age', 'household_income',
                   'children','connection_speed',
                   'country_of_origin','zip_code']):
    predictors = list(set(list(df_train)) - set([response] + demos))
    # get test/train sets
    y_train = df_train[response]
    y_test = df_test[response]

    x_train = df_train[predictors]
    x_test = df_test[predictors]
    
    log_model = LogisticRegressionCV(cv = 5, max_iter = 1000)
    
    print("fitting log model")

    log_model.fit(x_train, y_train)

    print("MODEL SCORE:", log_model.score(x_test, y_test))

    y_pred_test = log_model.predict(x_test)

    return log_model

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
        print('{}. {}'.format(i+1, list(train)[indices[i]]))
    return rf_model

################# EXCLUSIVITY INDEX MODELING ###############################

# calc_exclusivity()
# inputs:
#   df: the sample dataframe, with both demographics and matrix tranformation done
#   axis: racial axis to calculate indices on
#   n: number of top exclusivity indices to return
# returns:
#   dictionary of axis codes: top n domains by exclusivity index, over 95%, and by # of visits

def calc_exclusivity(df, axis, n = 100, outfile = 'exclusivity_indices.csv', threshold = 0.9):
    codes = list(np.unique(df[axis].values))
    domains = set(list(df)).difference(set(['machine_id', 'hoh_most_education', 'census_region', 'household_size', 'hoh_oldest_age', 'household_income', 'children', 'racial_background', 'connection_speed', 'country_of_origin', 'zip_code']))
    domains = list(domains)
    ph = [0] * len(domains)
    final_df = pd.DataFrame({'domain': domains, 'visits': ph})
    for c in codes:
        final_df[c] = ph
    counter = 0
    for idx, row in final_df.iterrows():
        d = row['domain']
        visits = df[d].sum()
        fracs = {c: len(df[df[axis] == c][d].nonzero()[0])/len(df[df[axis] == c]) for c in codes}
        denom = sum(fracs.values())
        for c in codes:
            final_df.at[idx, c] = fracs[c]/denom
        final_df.at[idx, 'visits'] = visits
        counter += 1
        if counter % 10000 == 0:
            print("{}/{} domains processed".format(counter, len(domains)))
    final_df.to_csv(outfile, index = False)
    final = {c: list(final_df[final_df[c] > threshold].sort_values(by=['visits'], ascending = False)['domain'])[:n] for c in codes}
    return final

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
# TARGETS NOT CURRENTLY SET TO CENSUS
def subsample(n, df, demos = ['racial_background'], targets = [{1: .5, 2: .35, 3: .1, 5: .05}]):
    # start with code for just one demographic axis
    # targets = [{1:.25, 2:.25, 3:.25, 5:.25}]
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

def get_subsample(sessions, demos, n = 20000):
    # get the subsample
    demos = pd.read_csv(demos)
    demos_sample = subsample(n, demos)
    subsample_numbers = demos_sample['machine_id']

    # read in the actual file and clean it here
    df_full = pd.read_csv(sessions)
    sample = df_full[df_full['machine_id'].isin(subsample_numbers)]
    print("{} unique domains in sample".format(len(sample['domain_name'].unique())))
    df_final = transform_mat(sample)
    df_final['racial_background'] = df_final['racial_background'].apply(lambda x: 1 if x == 1 else 2) 
    # force memory garbage collection
    demos = None
    df_full = None
    sample = None
    gc.collect()
    return df_final

def fit_models(df_final):
    train, test = split_data(df_final)
    # log_model = fit_log_model(train, test)
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
    #print(calc_exclusivity(sample, 'racial_background', n = 10))
    exit(0) 

if __name__ == '__main__':
    main()
