import argparse
import pandas as pd

def main(ds_df: pd.DataFrame, rad_df: pd.DataFrame, mimic3_df: pd.DataFrame, carevue_df: pd.DataFrame) -> pd.DataFrame:
    """Take the maximum subset of notes possible. The key is that ALL
    patients in the carevue subset are safe to use. Throw out ALL discharge
    summaries + radiology reports that are in the MetaVision set. """
    mimic4 = pd.concat([ds_df, rad_df])

    # Subtract notes3 - careview = Might overlap with MIMIC-IV
    carevue_subject_id = set(carevue_df['subject_id'].values)
    all_careview_set = mimic3_df[mimic3_df['SUBJECT_ID'].isin(carevue_subject_id)]
    potential_overlap = mimic3_df[~mimic3_df['SUBJECT_ID'].isin(carevue_subject_id)]

    # Take all notes that aren't from MIMIC-IV -> lowercase columns
    all_other_mimic3_notes = potential_overlap[~potential_overlap.CATEGORY.isin(['Radiology', 'Discharge summary'])]
    all_other_mimic3_notes.columns = [x.lower() for x in all_other_mimic3_notes.columns]
    all_careview_set.columns = [x.lower() for x in all_careview_set.columns]

    # Combine all of the notes
    final_df = pd.concat([mimic4, all_careview_set, all_other_mimic3_notes])
    counts = [len(x.split()) for x in final_df.text]
    print(f"We counted {sum(counts)} words.")
    return final_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv-iii', type=str, required=True, help="Location of the MIMIC-III CSV")
    parser.add_argument('--input-csv-ds', type=str, required=True, help="Location of the MIMIC-IV CSV")
    parser.add_argument('--input-csv-rad', type=str, required=True, help="Location of the MIMIC-IV CSV")
    parser.add_argument('--carevue-csv', type=str, required=True, help="Location of the Careview CSV.")
    parser.add_argument('--output-csv', type=str, required=True, help="Where to store the output")
    args = parser.parse_args()

    ds_df = pd.read_csv(args.input_csv_ds)
    rad_df = pd.read_csv(args.input_csv_rad)
    mimic3_df = pd.read_csv(args.input_csv_iii)
    carevue_df = pd.read_csv(args.carevue_csv)

    df = main(ds_df, rad_df, mimic3_df, carevue_df)
    df.to_csv(args.output_csv, index=False)
