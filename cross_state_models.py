#############
# Jack Deschler, Senior Thesis
#  cross state modeling code
############

import argparse
import os
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
from party_model_rf import fit_rf_model
from ei_model import *
parser = argparse.ArgumentParser()

def split_by_state(demos1, demos2, df):
    train_machines = demos1['machine_id']
    test_machines = demos2['machine_id']
    train = df[df['machine_id'].isin(train_machines)]
    test = df[df['machine_id'].isin(test_machines)]
    return (train, test) 

def main():
    parser.add_argument("state1", help='first state')
    parser.add_argument("state2", help='second state')
    args = parser.parse_args()
    state1 = args.state1
    state2 = args.state2
    demos1 = pd.read_csv('../demos/' + state1 + '16_demos.csv')
    demos2 = pd.read_csv('../demos/' + state2 + '16_demos.csv') 
    sessions1 = pd.read_csv('../' + state1 + '16_Sessions.csv')
    sessions2 = pd.read_csv('../' + state2 + '16_Sessions.csv')
    sessions = pd.concat([sessions1, sessions2], sort = False)
    sessions1 = None
    sessions2 = None
    gc.collect()
    # add cols so transform_mat doesn't break
    sessions['site_session_id'] = 0
    sessions['domain_id'] = 0
    sessions['ref_domain_name'] = 0
    sessions['duration'] = 0
    sessions['tran_flg'] = 0
    sessions = transform_mat_party(sessions)
    sessions = classify_party(sessions, threshold = 0.5, two_party = True)

    # State 1 on State 2
    print(state1 + ' on ' + state2)
    train, test = split_by_state(demos1, demos2, sessions)
    eis = calc_exclusivity_v2(train, 'democrat', n = 100, outfile = None, threshold = 0.9) 
    print("EI original criterion")
    ei_classifier(eis, test, 'democrat', mod = False)
    os.rename('confmat.png', 'combo_confmat/' + state1 + 'on' + state2 + '_ei_orig.png')
    print("EI modified criterion")
    ei_classifier(eis, test, 'democrat', mod = True)
    os.rename('confmat.png', 'combo_confmat/' + state1 + 'on' + state2 + '_ei_mod.png')
    print("RF Model")
    fit_rf_model(train, test) 
    os.rename('confmat.png', 'combo_confmat/' + state1 + 'on' + state2 + '_rf.png')
    print()

    # State 2 on State 1
    print(state2 + " on " + state1)
    train, test = split_by_state(demos2, demos1, sessions)
    eis = calc_exclusivity_v2(train, 'democrat', n = 100, outfile = None, threshold = 0.9) 
    print("EI original criterion")
    ei_classifier(eis, test, 'democrat', mod = False)
    os.rename('confmat.png', 'combo_confmat/' + state2 + 'on' + state1 + '_ei_orig.png')
    print("EI modified criterion")
    ei_classifier(eis, test, 'democrat', mod = True)
    os.rename('confmat.png', 'combo_confmat/' + state2 + 'on' + state1 + '_ei_mod.png') 
    print("RF Model")
    fit_rf_model(train, test)
    os.rename('confmat.png', 'combo_confmat/' + state2 + 'on' + state1 + '_rf.png') 
    print()
    
    # Combined Model
    print("Combined Model")
    train, test = split_data(sessions)
    eis = calc_exclusivity_v2(train, 'democrat', n = 100, outfile = 'combined_ei.csv', threshold = 0.9)
    print("EI original criterion")
    ei_classifier(eis, test, 'democrat', mod = False)
    os.rename('confmat.png', 'combo_confmat/combo_ei_orig.png') 
    print("EI modified criterion")
    ei_classifier(eis, test, 'democrat', mod = True)
    os.rename('confmat.png', 'combo_confmat/combo_ei_mod.png')
    print("RF Model")
    fit_rf_model(train, test)
    os.rename('confmat.png', 'combo_confmat/combo_rf.png')


if __name__ == '__main__':
    main()
