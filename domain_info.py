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
    result = pd.DataFrame(columns = ['users', 'domains'])
    ns = [1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    for n in ns:
        lst = []
        for _t in range(trials):
            tmp = df.sample(n)
            lst.append(len(tmp['domain_name'].unique()))
        avg = sum(l) / float(len(l))
        result.append([n, avg], columns=['users','domains'])
    result.to_csv(outfile)

if __name__ == '__main__':
    main()

