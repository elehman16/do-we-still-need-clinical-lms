import os
import re
import pandas as pd
import argparse
from tqdm import tqdm
from preprocessing.convert_deid_to_tag import convert_deid_to_tag, convert_tag_type_to_tag

def replace_single_note(text: str, note_type: str):
    """Find the offsets for a single note."""
    if note_type == 'mimic-iii':
        regex = "\[\*\*(.*?)\*\*\]"
    else:
        regex = "____*"

    offsets = []
    for m in re.compile(regex).finditer(text):
        selected_text = "" if note_type == 'mimic-iv' else m.groups()[0]
        offsets.append((m.start(), m.end(), selected_text))

    return offsets

def replace_list_of_notes(text_to_replace: list[str], note_type: str):
    """Take a list of notes and replace the tags with single tokens. """
    assert(note_type in ['mimic-iii', 'mimic-iv'])

    all_texts = []
    for text in text_to_replace:
        offsets = replace_single_note(text, note_type=note_type)

        new_text, last_offset = [], 0
        for i, (st, end, span) in enumerate(offsets):
            if note_type == 'mimic-iii':
                tag = convert_deid_to_tag(span)
            else:
                if len(row['deid_type'].values) <= i:
                    tag = "[MISC]"
                else:
                    tag = convert_tag_type_to_tag(row['deid_type'].values[i])

            new_text.append(text[last_offset:st])
            new_text.append(tag)
            last_offset = end

        new_text.append(text[last_offset:])
        all_texts.append(' '.join(filter(lambda x: x != '', new_text)))

    return all_texts 

def main(df: pd.DataFrame, note_type: str) -> pd.DataFrame:
    """Replace all of the DEID items with special tags. """
    all_texts = []
    text_key = 'TEXT' if note_type == 'mimic-iii' else 'text'
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        offsets = replace_single_note(row[text_key], note_type)

        new_text, last_offset = [], 0
        for i, (st, end, span) in enumerate(offsets):
            if note_type == 'mimic-iii':
                tag = convert_deid_to_tag(span)
            else:
                if len(row['deid_type'].values) <= i: 
                    tag = "[MISC]"
                else:
                    tag = convert_tag_type_to_tag(row['deid_type'].values[i])

            new_text.append(row[text_key][last_offset:st])
            new_text.append(tag)
            last_offset = end
    
        new_text.append(row[text_key][last_offset:])
        all_texts.append(' '.join(filter(lambda x: x != '', new_text)))

    df[text_key] = all_texts
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=str, help="Where to load in the new notes.")
    parser.add_argument("--input-annotation-csv", type=str, help="Where to load in the MIMIC-IV annotations.")
    parser.add_argument("--output-csv", required=True, type=str, help="Where to store the notes with new tags")
    parser.add_argument("--type", required=True, type=str, choices=['mimic-iii', 'mimic-iv'], help="Running on MIMIC-III or IV text?")
    args = parser.parse_args()
      
    # Load new df, convert notes, save 
    df = pd.read_csv(args.input_csv)
    if args.type == 'mimic-iv':
        annotations = pd.read_csv(args.input_annotation_csv)
               
        ids, deid_type = [], []
        for note_id, group in annotations.groupby('note_id'):
            ids.append(note_id)
            deid_type.append(group['entity_type'])

        ids_offsets = pd.DataFrame({'note_id': ids, 'deid_type': deid_type})
        df = df.merge(ids_offsets, on='note_id')

    new_df = main(df, args.type)
    new_df.to_csv(args.output_csv, index=False)
