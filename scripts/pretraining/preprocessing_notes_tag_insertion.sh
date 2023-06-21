# DE-DEDUPLICATE MIMIC-III NOTES
python preprocessing/dedup_mimic_iii.py --notes-path "data/raw_data/NOTEEVENTS.csv" --out-path "data/NOTEEVENTS_DEDUP.csv"

# REPLACE DEID TAGS IN MIMIC III
python preprocessing/replace_deid_tags.py \
    --type "mimic-iii" \
    --input-csv "data/NOTEEVENTS_DEDUP.csv" \
    --output-csv "data/NOTEEVENTS_DEDUP_REPLACED_TAGS.csv"

# REPLACE DEID TAGS FOR MIMIC IV
python preprocessing/replace_deid_tags.py \
    --type "mimic-iv" \
    --input-csv "data/raw_datasets/discharge.csv" \
    --input-annotation-csv "data/raw_datasets/discharge_annotation.csv" \
    --output-csv "data/DISCHARE_REPLACED_DEID_TAGS.csv"

python preprocessing/replace_deid_tags.py \
    --type "mimic-iv" \
    --input-csv "data/raw_datasets/radiology.csv" \
    --input-annotation-csv "data/raw_datasets/radiology_annotation.csv" \
    --output-csv "data/RADIOLOGY_REPLACED_DEID_TAGS.csv"

# COMBINE ALL OF THE CSVs
python preprocessing/combine_mimics.py \
    --input-csv-iii "data/NOTEEVENTS_DEDUP_REPLACED_TAGS.csv" \
    --input-csv-ds "data/DISCHARE_REPLACED_DEID_TAGS.csv" \
    --input-csv-rad "data/RADIOLOGY_REPLACED_DEID_TAGS.csv" \
    --carevue-csv "data/raw_datasets/MIMIC_III_CAREVIEW_NOTEEVENTS.csv" \
    --output-csv "data/ALL_COMBINED_III_IV_REPLACED_TAGS.csv"

# REMOVE PATIENTS IN OUR FINETUNING DATASETS
python preprocessing/remove_patient_ids.py \
    --input-csv "data/ALL_COMBINED_III_IV_REPLACED_TAGS.csv" \
    --mimic-iii "data/ids_to_remove_mimic_iii_only.csv" \
    --mimic-iv "data/mimic_iv_ids_to_remove.csv" \
    --output-csv "data/COMBINED_III_IV_TRAIN_REPLACED_TAGS.csv"

# STORE THE TEXT ONLY
python preprocessing/convert_csv_to_pretraining_format.py \
    --input-csv "data/COMBINED_III_IV_TRAIN_REPLACED_TAGS.csv" \
    --output-csv "data/COMBINED_III_IV_TRAIN_REPLACED_TAGS_TEXT_ONLY.csv"
