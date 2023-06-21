import glob
import json
import argparse
import pandas as pd
from tqdm import tqdm
from convert_deid_to_tag import convert_deid_to_tag

def generated_text_to_idx(generated_text: str, index: int, is_mimic_iii: bool):
    """Given ALL the generated text from a note, get """
    ch_index = generated_text.find(f'<extra_id_{index}') 
    if ch_index == -1: return ""

    ch_end = -1
    ch_index += len(f'<extra_id_{index}') + 1
    for i in range(ch_index, len(generated_text)):
        if generated_text[i] != '<': continue 

        # Check for </s>
        if generated_text[i:i+len('</s>')] == '</s>':
            ch_end = i
            break    

        # Check for <extra_id
        if generated_text[i:i+len('<extra_id')] == '<extra_id':
            ch_end = i
            break

    if ch_end == -1: return -1
    return generated_text[ch_index:ch_end]


def main(df: pd.DataFrame, note_type: str) -> pd.DataFrame:
    """Dataframe with a text field that we are going to replace
    the DEID tags with. If no deid replacement is given, remove
    the data entirely.
    @param df is the dataframe to process. 
    @param note_type is either `mimic-iii` or `mimic-iv`.
    @return a new df with a modified text field. """
    new_texts = []
    text_key = 'TEXT' if note_type == 'mimic-iii' else 'text'
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        text, all_offsets, all_replacements = row[text_key], row['Offsets'], row['Generated_Text']
        new_text, last_index, to_skip, seen = "", 0, 0, 0
        if note_type == 'mimic-iv':
            last_index = text.index('Sex:')

            # Figure out how many ___ are in the header
            to_skip = text[:last_index].count('___')

        for offset_group, replacement_group in zip(all_offsets, all_replacements):
            for i, offset in enumerate(offset_group):
                if seen < to_skip:
                    seen += 1
                    continue
                 
                if note_type == 'mimic-iii':
                    selected_text = text[offset[0] + 3:offset[1] - 3]
                    tag = convert_deid_to_tag(selected_text)

                    # Do not replace dates/numbers. Those are pretty good usually.
                    if tag.strip() == '' or (tag[0] != '[' and tag[-1] != ']'): continue

                # Get the generated text -> incorporated it
                new_text += text[last_index:offset[0]]
                replaced_text = generated_text_to_idx(replacement_group, i, args.type == 'mimic-iii') 
                last_index = offset[1]
                seen += 1 

        # Add the rest
        new_text += text[last_index:]
        new_texts.append(new_text)

    df[text_key] = new_texts    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=str, help="Where to load in the new notes.")
    parser.add_argument("--deid-loc", required=True, type=str, help="Where are the deid tags stored?")
    parser.add_argument("--output-csv", required=True, type=str, help="Where to store the notes with new tags")
    parser.add_argument("--type", required=True, type=str, choices=['mimic-iii', 'mimic-iv'], help="Running on MIMIC-III or IV text?")
    args = parser.parse_args()    

    # Load each of the json shards and combine.
    df = pd.read_csv(args.input_csv)
    generated_texts, row_ids, offsets = [], [], []
    for f in glob.glob(args.deid_loc):
        jf = json.load(open(f))
        generated_texts.extend(jf['generated_text'])
        row_ids.extend(jf['row_id'])
        offsets.extend(jf['offsets'])

    row_id_key = 'ROW_ID' if args.type == 'mimic-iii' else 'note_id'
    gen_df = pd.DataFrame({'Generated_Text': generated_texts, 
                           row_id_key: row_ids,
                           'Offsets': offsets})

    # If MIMIC-III, ROW_IDs are numbers
    if args.type == 'mimic-iii':
        gen_df[row_id_key] = pd.to_numeric(gen_df[row_id_key])

    # We have many rows with the same `row_id` since notes are broken into chunks.
    groups = {row_id_key: [], 'Generated_Text': [], 'Offsets': []}
    for i, group in gen_df.groupby(row_id_key):
        # PRETTY positive that these are already ordered
        groups[row_id_key].append(i)
        groups['Generated_Text'].append(group.Generated_Text.values)
        groups['Offsets'].append(group.Offsets.values)

    # Take the merged DF and process the text.
    grouped_df = pd.DataFrame(groups)

    # Quick sanity check to make sure we aren't missing anything
    df = main(df.merge(grouped_df, on=row_id_key), note_type=args.type)
    df = df.drop(columns=['Offsets', 'Generated_Text'])
    df.to_csv(args.output_csv, index=False)
