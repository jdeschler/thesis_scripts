####################
# Jack Deschler
# Data Cleaning for Senior Thesis
#  cleans comScore data
####################
import sys
import pandas as pd
import numpy as np

def process_data(csv):
    token = csv[:-4]
    df = pd.read_csv(csv)
    # extract demographics, save separately for re-linking later
    df_demos = df[['machine_id', 'hoh_most_education', 'census_region', 
                   'household_size', 'hoh_oldest_age', 'household_income', 
                   'children', 'racial_background','connection_speed',
                   'country_of_origin','zip_code']]
    df_demos = df_demos.drop_duplicates('machine_id')
    df_demos.to_csv(token + '_demographics.csv', index = False)
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
    # merge demographics back, and write final
    final = df.merge(df_demos, on = 'machine_id')
    final.to_csv(token + '_cleaned.csv', index = False)

process_data(sys.argv[1])
    
