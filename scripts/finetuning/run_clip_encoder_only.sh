MODEL_PATH=$1
OUT_PATH=$2
LR=$3
BS=$4
GAS=$5

echo 'Loading model from' $MODEL_PATH
echo 'Saving to ' $OUT_PATH
echo 'LR is set to ' $LR
echo 'BS is set to ' $BS
echo 'GAS is set to ' $GAS

for i in {1..3}
do
    echo "Storing in $OUT_PATH/seed-$i/"
    python src/finetuning/preprocess_clip_encoder_only.py \
        --clip-dir ./data/clip_data/ \
        --model-path $MODEL_PATH \
        --output-dir "$OUT_PATH/seed-$i/" \
        --lr $LR \
        --seed $i \
        --batch-size $BS \
        --gradient-accumulation-steps $GAS 
done
