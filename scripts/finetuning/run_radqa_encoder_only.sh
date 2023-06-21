#!/bin/sh
INPUT_MODEL=$1
OUT_DIR=$2
LR=$3
MAX_SEQ=$4
STRIDE=$5
BS=$6
GAS=$7

echo 'Input: ' $INPUT_MODEL
echo 'Output: ' $OUT_DIR

for i in 1 2 3
do
    echo "Saving to: $OUT_DIR/seed_$i/"
    deepspeed --num_gpu 4 src/finetuning/run_qa.py \
        --model_name_or_path $INPUT_MODEL \
        --is_gpt_model \
        --train_file "data/radqa_data/train.csv" \
        --validation_file "data/radqa_data/dev.csv" \
        --test_file "data/radqa_data/test.csv" \
        --version_2_with_negative \
        --do_train \
        --do_eval \
        --do_predict \
        --per_device_train_batch_size $BS \
        --per_device_eval_batch_size $BS \
        --gradient_accumulation_steps $GAS \
        --learning_rate $LR \
        --num_train_epochs 10 \
        --save_total_limit 1 \
        --metric_for_best_model "eval_f1" \
        --report_to "wandb" \
        --evaluation_strategy "epoch" \
        --save_strategy "epoch" \
        --seed $i \
        --load_best_model_at_end \
        --overwrite_output_dir \
        --max_seq_length $MAX_SEQ \
        --doc_stride $STRIDE \
        --output_dir "$OUT_DIR/seed_$i/" \
        --deepspeed deep_speed_config.json
done
