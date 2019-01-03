####################
# Jack Deschler
# Adds states to comScore demographics tht have been pulled out 
####################
import sys
import pandas as pd
import numpy as np

def process_data(csv, zips):
    token = csv[:-4]
    df = pd.read_csv(csv)
    zips = pd.read_csv(zips, usecols=['Zipcode','State'])
    zips.rename(index=str, columns={'Zipcode':'zip_code'}, inplace=True)
    final = df.merge(zips, on='zip_code', how='inner', copy=False)
    final.to_csv(token + "_states.csv", index=False)

process_data(sys.argv[1], sys.argv[2])
    
