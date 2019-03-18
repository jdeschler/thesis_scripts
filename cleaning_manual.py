# Data Cleaning for Senior Thesis, manual
#  Jack Deschler

import pandas as pd
import sys

def process_data(csv):
    token = csv[:-4]
    cols = ['machine_id', 'domain_name', 'pages_viewed']
    df = pd.read_csv(csv, usecols=cols, encoding = "ISO-8859-1")
    machines = df['machine_id'].unique()
    res = pd.DataFrame(index=machines)
    # process data frame row by row instead of all at once
    # saves memory at the cost of speed
    for m in machines:
        df_m = df.loc[df['machine_id'] == m]
        for _index, row in df_m.iterrows():
            site = row['domain_name']
            if site not in list(res):
                # add the column if necessary
                res[site] = 0
            # now add to the cell
            res.at[m, site] += row['pages_viewed']
    res.columns = res.columns.astype(str)
    res.reindex(sorted(res.columns), axis=1)
    res.to_csv(token + '_cleaned_manual.csv', index_label='machine_id')

process_data(sys.argv[1])


