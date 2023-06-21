#!/bin/sh
MIMIC_OUT_CSV=$1
OUT_DIR=$2

echo 'Input: ' $MIMIC_OUT_CSV
echo 'Output: ' $OUT_DIR

# Run MLM on this for t5-base
# Our sequence length is a quarter of what it is in the T5 paper -> 40K warmup steps
python src/run_t5_mlm_flax.py \
    --do_train \
    --do_eval \
    --output_dir=$OUT_DIR \
    --tokenizer_name="t5-base" \
    --model_name_or_path="t5-base" \
    --max_seq_length="512" \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --inverse_adafactor \
    --warmup_steps="40000" \
    --overwrite_output_dir \
    --save_steps="20000" \
    --eval_steps="20000" \
    --num_train_epochs=10 \
    --seed=42 \
    --train_file $MIMIC_OUT_CSV \
    --cache_dir data/cached_datasets/ 
