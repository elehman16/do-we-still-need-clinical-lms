import argparse
import pandas as pd 

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', type=str, required=True, help="Which CSV to load.")
    parser.add_argument('--output-csv', type=str, required=True, help="Where to save the csv.")
    args = parser.parse_args()    

    df = pd.read_csv(args.input_csv)
    df['text'].to_csv(args.output_csv, index=False, header=False)
