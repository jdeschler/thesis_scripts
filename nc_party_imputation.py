##################################
# Jack Deschler, Senior Thesis
# code to impute party characteristics into comScore data
#  only for use with North Carolina voter files
##################################

import pandas as pd
import sys

# in NC, we can match on zip code, race, and age
# have to use the comScore coding (complicated wrt age)
def impute_demos(vf, demos):
    # first do some preprocessing on the voter file
    vf['Race'] = vf['Race'].apply(lamda x: 'O' if x in ['I','O','U','M'] else x) 
    # have the upper level of age ranges for the switch case below
    age_codes = {0: 0, 1: 20, 2: 24, 3: 29, 4: 34, 5: 39, 6: 44, 7: 49, 8: 54, 9: 59, 10: 64}
    race_codes = {1: 'W', 2: 'B', 3: 'A', 5: 'O'}
    for idx, row in demos.iterrows():
        # by zip code
        curr = vf[vf['zip_code'] == row['zip_code']]
        # by race
        curr = curr[curr['Race'] == race_codes[row['racial_background']]]
        # by age
        age_code = row['hoh_oldest_age']
        curr = curr[curr['birth_age'] <= age_codes[age_code] \
                    and curr['birth_age'] > age_codes[age_code - 1]]
        # now calculate the statistics
        demos.loc[idx, 'vf_k'] = len(curr)
        d_count = curr['party_cd'].value_counts()['DEM']
        r_count = curr['party_cd'].value_counts()['REP']
        demos.loc[idx, 'vf_k_2p'] = d_count + r_count
        demos.loc[idx, 'D_pct'] = d_count / len(curr)
        demos.loc[idx, 'D_pct_2p'] = d_count / (d_count + r_count) 
    return demos

def main():
    if len(sys.argv) != 3:
        print('Usage: python nc_party_imputation.py NC_voterfile.txt NC_demos.csv')
        exit(1)
    vf = pd.read_csv(sys.argv[1], sep='\t', lineterminator='\n', encoding = "ISO-8859-1")
    demos = pd.read_csv(sys.argv[2])
    demo_cols = list(demos)
    if 'D_pct' not in demo_cols:
        demos['D_pct'] = 0
    if 'D_pct_2p' not in demo_cols:
        demos['D_pct_2p'] = 0
    if 'vf_k' not in demo_cols:
        demos['vf_k'] = 0
    if 'vf_k_2p' not in demo_cols:
        demos['vf_k_2p'] = 0
    # do the imputation in a separate function
    demos = impute_demos(vf, demos)
    demos.to_csv(sys.argv[2])
    exit(0)

if __name__ == '__main__':
    main()
