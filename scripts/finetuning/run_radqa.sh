#!/bin/sh
INPUT_MODEL=$1
OUT_DIR=$2
LR=$3

echo 'Input: ' $INPUT_MODEL
echo 'Output: ' $OUT_DIR
echo 'LR: ' $LR

for i in 1 2 3
do
    echo "Storing in $OUT_DIR/seed_$i/"
    python -m torch.distributed.launch --nproc_per_node 4 src/finetuning/run_seq2seq_qa.py \
      --model_name_or_path $INPUT_MODEL \
      --train_file "data/radqa_data/train.csv" \
      --validation_file "data/radqa_data/dev.csv" \
      --test_file "data/radqa_data/test.csv" \
      --context_column context \
      --question_column question \
      --answer_column answers \
      --do_train \
      --do_eval \
      --do_predict \
      --save_total_limit 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 16 \
      --per_device_eval_batch_size 1 \
      --num_train_epochs 15 \
      --version_2_with_negative \
      --max_seq_length 1024 \
      --load_best_model_at_end \
      --predict_with_generate \
      --metric_for_best_model "f1" \
      --report_to "wandb" \
      --evaluation_strategy "epoch" \
      --learning_rate $LR \
      --lr_scheduler_type "constant" \
      --optim "adafactor" \
      --save_strategy "epoch" \
      --overwrite_output_dir \
      --seed $i \
      --output_dir $OUT_DIR/seed_$i/
done
