##################
# Jack Deschler, Senior Thesis
#  code to build dataset of number of domains vs number of users
##################

import pandas as pd
import numpy as np
import gc
import argparse
parser = argparse.ArgumentParser()

def main():
    parser.add_argument('Sessions', help='comScore sessions file')
    parser.add_argument('-t', '--trials', type=int, help='number of trials')
    parser.add_argument('-o', '--outfile', type=str, help='name of outfile')
    args = parser.parse_args()
    outfile = 'users_domains.csv' if not args.outfile else args.outfile
    trials = 10 if not args.trials else args.trials
    df = pd.read_csv(args.Sessions)
    ns = [1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    avgs = []
    users = pd.DataFrame(df['machine_id'].unique())
    for n in ns:
        print(n)
        lst = []
        for _t in range(trials):
            tmp = users.sample(n)
            sample = df[df['machine_id'].isin(tmp[0])]
            lst.append(len(sample['domain_name'].unique()))
        avg = sum(lst) / float(len(lst))
        avgs.append(avg)    
    result = pd.DataFrame({'users': ns, 'domains': avgs})
    result.to_csv(outfile)
    print("Written to: " + outfile)

if __name__ == '__main__':
    main()

