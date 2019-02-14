###############################################
# Jack Deschler, Senior Thesis
#   code to fit models on comScore data
#   based on Zang/Sweeney's Exclusivity Indices
#################################################

import numpy as np
import pandas as pd
import scipy as sp
import argparse
import gc
import matplotlib
import operator
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#from joblib import dump, load
import itertools
from collections import Counter
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
    df = df.pivot_table(index='machine_id', columns='domain_name', values='pages_viewed', aggfunc = lambda x: np.sum(x).astype(bool).astype(int))
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
    if outfile:
        final_df.to_csv(outfile, index = False)
    final = {c: list(final_df[final_df[c] > threshold].sort_values(by=['visits'], ascending = False)['domain'])[:n] for c in codes}
    return final

def calc_exclusivity_v2(df, axis, n = 100, outfile = 'exclusivity_indices.csv', threshold = 0.7):
    codes = list(np.unique(df[axis].values))
    domains = set(list(df)).difference(set(['machine_id', 'hoh_most_education', 'census_region', 'household_size', 'hoh_oldest_age', 'household_income', 'children', 'racial_background', 'connection_speed', 'country_of_origin', 'zip_code']))
    domains = list(domains)
    ph = [0.] * len(domains)
    visits = {c: df[df[axis] == c].sum() for c in codes}
    lengths = {c: len(df[df[axis] == c]) for c in codes}
    lengths['total'] = len(df)
    visits_df = pd.DataFrame({'domain': domains, 'visits': ph})
    for c in codes:
        visits_df[c] = ph
    counter = 0
    for idx, row in visits_df.iterrows():
        tot = 0
        d = row['domain']
        for c in codes:
            tmp = float(visits[c][d])
            tmp2 = tmp/float(lengths[c])
            visits_df.at[idx, c] = tmp/float(lengths[c])
            tot += tmp
        visits_df.at[idx, 'visits'] = tot
        counter += 1
    # have visit fractions, must process them now
    visits_df['tot'] = visits_df.drop(['visits','domain'], axis = 1).sum(axis = 1)
    for c in codes:
        visits_df[c] = visits_df[c] / visits_df['tot']
    if outfile:
        visits_df.to_csv(outfile, index = False)
    final = {c: list(visits_df[visits_df[c] > threshold].sort_values(by=['visits'], ascending = False)['domain'])[:n] for c in codes}
    return final

def ei_classifier(eis, df, outcome, mod = False):
    y_true = df[outcome].values
    df['pred'] = 0
    counter = 0
    y_hat_classified = []
    y_true_classified = []
    for idx, row in df.iterrows():
        counts = {c: 0 for c in list(eis)}
        for c in list(eis):
            domains = eis[c]
            for d in range(len(domains)):
                domain = domains[d]
                try:
                    if row[domain] > 0:
                        counts[c] += (101 - d) if mod else 1
                except KeyError:
                    pass
        if sum(list(counts.values())) == 0:
            counter += 1
        classify = max(counts.items(), key=operator.itemgetter(1))[0]
        classify = int(classify)
        df.at[idx, 'pred'] = classify
        if sum(list(counts.values())) > 0:
            y_hat_classified += [classify]
            y_true_classified += [df.at[idx, outcome]]
    y_hat = df['pred'].values
    print("{} had none of the domains".format(counter))
    print("Overall accuracy (among classified): {}".format(classification_accuracy(y_true_classified, y_hat_classified)))

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
    targets = [{1:.5, 2:.35, 3:.1, 5:.05}]
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
    
    # force memory garbage collection
    demos = None
    df_full = None
    sample = None
    gc.collect()
    return df_final

def main():
    parser.add_argument('-n', type=int, help='size of subsample')
    parser.add_argument('Sessions', help='comScore Sessions file')
    parser.add_argument('demos', help='comScore demographics file')
    parser.add_argument('-o', '--outfile', help='slug for outfile for exclusivity indices')
    parser.add_argument('-f', '--infile', help='file to read in EIs')
    parser.add_argument('axis', help='axis to calculate on')
    parser.add_argument('-m', '--modified', help='use modified criterion')
    args = parser.parse_args()
    mod = False if not args.modified else True
    n = 20000 if not args.n else args.n
    outcome = args.axis
    out = 'exclusivity_indices.csv' if not args.outfile else str(args.outfile)
    excl_indices = None
    if not args.infile:
        eis = []
        ei_df = pd.DataFrame()
        for i in range(10):
            sample = None
            gc.collect()
            print(i+1)
            sample = get_subsample(args.Sessions, args.demos, n=n)
            eis.append(calc_exclusivity_v2(sample, outcome, n = 100, outfile = None))
        for c in eis[0].keys():
            l = [a[c] for a in eis]
            l = [a for b in l for a in b]
            counts = dict(Counter(l))
            counts = sorted(counts.items(), key=lambda kv: kv[1])
            counts = [c for (c,_) in counts[:100]]
            tmp_df = pd.DataFrame({c:counts})
            ei_df = pd.concat([ei_df, tmp_df], axis=1)
        ei_df = ei_df.fillna(value = '')
        ei_df.to_csv(out, index = False)
        excl_indices = ei_df
    else:
        excl_indices = pd.read_csv(args.infile)
    print("Getting testing subsample")
    sample = None
    gc.collect()
    sample = get_subsample(args.Sessions, args.demos, n = n)
    ei_classifier(excl_indices, sample, outcome, mod = mod)
    exit(0) 

if __name__ == '__main__':
    main()
