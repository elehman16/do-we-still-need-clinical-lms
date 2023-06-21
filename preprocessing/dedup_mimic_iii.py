import argparse
import numpy as np
import pandas as pd
from datetime import datetime

def main(notes_path: str, out_path: str):
    df = pd.read_csv(notes_path).fillna('NaN')
    groups_to_keep = []
    rows_to_keep = []
    for idx, group in df.groupby(['CHARTTIME', 'SUBJECT_ID', 'CATEGORY', 'CGID']):
        # If accidently grouped together or group of 1, let's keep it all
        store_time_values = set(group.STORETIME.values)
        cgid_values = set(group.CGID.values)
        
        if len(group) == 1 or group.CHARTTIME.iloc[0] == 'NaN' or 'NaN' in cgid_values:
            groups_to_keep.append(group)
        elif len(store_time_values) == 1 and 'NaN' in store_time_values:
            groups_to_keep.append(group)
        elif 'NaN' in store_time_values:
            groups_to_keep.append(group)
        else:
            st = [datetime.strptime(store_time, '%Y-%m-%d %H:%M:%S') for store_time in group.STORETIME.values]
            most_recent = np.argmax(st)
            rows_to_keep.append(group.iloc[most_recent])

    groups_cmb = pd.concat(groups_to_keep)
    rows_cmb = pd.DataFrame(rows_to_keep) #pd.concat(rows_to_keep, axis=1)
    df = pd.concat([groups_cmb, rows_cmb])
    df.to_csv(out_path, index=False)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--notes-path', type=str, required=True, help="")
    parser.add_argument('--out-path', type=str, required=True, help="")
    args = parser.parse_args()    

    main(args.notes_path, args.out_path)
