import glob
import json
import pickle
import argparse
import pandas as pd
from tqdm import tqdm 
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import sparse_dot_topn.sparse_dot_topn as ct
from scipy.sparse import csr_matrix
from collections import Counter

def get_n2c2_ids(data_path: str) -> list:
    """Get the IDs involved in 2010-2012 tasks. """
    ids = set(pd.read_csv(f"{data_path}/n2c2_ids/subject_ids.csv")['subject_id'].values)
    return list(ids)

def get_clip_ids(data_path: str, mimic3_df: pd.DataFrame) -> list:
    """Get the CLIP ids. """
    clip_paths = f"{data_path}/clip_ids/*.csv"

    clip_ids = set()
    for path in glob.glob(clip_paths):
        df = pd.read_csv(path, header=None)
        clip_ids.update(df[0].values)

    return list(set(mimic3_df[mimic3_df['ROW_ID'].isin(clip_ids)].SUBJECT_ID))

def get_radqa_subject_ids(data_path: str, mimic3_df: pd.DataFrame) -> list:
    """Get the RADQA ids. 
    @param data_path is the file extension to where the RADQA ids are stored.
    @param mimic3_df is the mimic3 noteevent dataframe. """
    path = f"{data_path}/radqa_ids/radqa_"
    document_ids = set()
    for rest_path in ['train.json', 'dev.json', 'test.json']:
        jf = json.load(open(path + rest_path))
        for row in jf['data']:
            document_ids.add(int(row['title']))

    return list(set(mimic3_df[mimic3_df['ROW_ID'].isin(document_ids)].SUBJECT_ID))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', type=str, required=True, help="Path to the notes.")
    parser.add_argument('--mimic-iii-ids', type=str, required=True, help="Path to MIMIC-III notes to ignore")
    parser.add_argument('--mimic-iv-ids', type=str, required=True, help="Path to MIMIC-IV notes to ignore")
    parser.add_argument('--output-csv', type=str, required=True, help="Where to store the filtered notes.")
    args = parser.parse_args()

    # Read the CSVs + remove the IDs
    df = pd.read_csv(args.input_csv)
    df3 = pd.read_csv(args.mimic_iii_ids)
    df4 = pd.read_csv(args.mimic_iv_ids)
    ids_to_skip = list(df3.subject_id.values) + list(df4.subject_id_mimic_iv.values)
    removed_pid_df = df[~df['subject_id'].isin(ids_to_skip)]
    removed_pid_df.to_csv(args.output_csv, index=False)
