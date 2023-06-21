import os
import glob
import argparse
import numpy as np
import pandas as pd
from preprocessing.replace_deid_tags import replace_list_of_notes

def preprocess_answers(df: pd.DataFrame) -> pd.DataFrame:
    """Modify the answers.  """
    answers = []
    for _, row in df.iterrows():
        ans = eval(row.answers)
        
        new_text = replace_list_of_notes(ans['text'], 'mimic-iii')
        try: 
            new_ans_st = [row.context.index(t) for t in new_text]
        except:
            # This means that they accidently cut-off the DEID tag...
            tmp = []
            for a, st in zip(ans['text'], ans['answer_start']):
                end_index = row.orig_context[st + len(a):].index('**]')
                tmp.append(row.orig_context[st:st + len(a) + end_index + 3])

            new_text = replace_list_of_notes(tmp, 'mimic-iii')
            
        answers.append({'text': new_text, 'answer_start': new_ans_st})
            
    df['answers'] = answers
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--radqa_path', type=str, required=True, help="Path to the RadQA data.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to store the modified data.")
    args = parser.parse_args()

    if not(os.path.exists(args.output_dir)):
        os.mkdir(args.output_dir)

    # Iterate over the CSVs and save
    for fp in glob.glob(args.radqa_path + '/*.csv'):
        df = pd.read_csv(fp)
        df['orig_context'] = df['context']
        df['context'] = replace_list_of_notes(df['context'], 'mimic-iii')
        df = preprocess_answers(df)
        
        # Get new fname and save
        f_name = fp.split(args.radqa_path)[-1]
        out_path = os.path.join(args.output_dir, f_name)
        df.to_csv(out_path, index=False)
