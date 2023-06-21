#!/bin/sh
MIMIC_INIT_CSV=$1
OUT_DIR=$2
FINAL_OUT_DIR=$3
MODEL_TYPE=$4

# Create our own tokenizer.
python preprocessing/t5_from_scratch_tokenizer.py \
    --model t5-base \
    --input-csv $MIMIC_INIT_CSV \
    --out-dir $OUT_DIR \
    --vocab-size 32000 \
    --create-new-model 

# Run MLM on this for t5-base
# Our sequence length is a quarter of what it is in the T5 paper -> 40K warmup steps
python src/run_t5_mlm_flax.py \
    --do_train \
    --do_eval \
    --output_dir=$FINAL_OUT_DIR \
    --model_type=$MODEL_TYPE \
    --config_name=$OUT_DIR \
    --tokenizer_name=$OUT_DIR \
    --max_seq_length="512" \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=128 \
    --inverse_adafactor \
    --warmup_steps="40000" \
    --overwrite_output_dir \
    --save_steps="20000" \
    --eval_steps="20000" \
    --num_train_epochs=28 \
    --seed=42 \
    --train_file $MIMIC_INIT_CSV \
    --cache_dir data/cached_datasets/ 
