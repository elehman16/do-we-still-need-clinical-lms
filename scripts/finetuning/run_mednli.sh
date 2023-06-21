for i in {1..3}
do
    echo "Storing in models/finetuning/mednli-t5-base-seed-$i/"
    python src/finetuning/preprocess_mednli.py \
        --mednli-dir ./data/mednli/ \
        --model-path t5-base \
        --output-dir "models/finetuning/mednli-t5-base-seed-$i/" \
        --use-constant-adafactor \
        --seed $i
done 

array=( "models/mimic-t5-model-replace-deid-tags-11-23-22/model_outputs_20000/" 
       "models/mimic-t5-model-replace-deid-tags-11-23-22/model_outputs_40000/"
       "models/mimic-t5-model-replace-deid-tags-11-23-22/model_outputs_60000/"
       "models/mimic-t5-model-replace-deid-tags-11-23-22/model_outputs_80000/"
       "models/mimic-t5-model-replace-deid-tags-11-23-22/model_outputs_100000/" )

array2=( "models/finetuning/mednli-t5-deid-tags-20K_reg/" 
        "models/finetuning/mednli-t5-deid-tags-40K_reg/"
        "models/finetuning/mednli-t5-deid-tags-60K_reg/"
        "models/finetuning/mednli-t5-deid-tags-80K_reg/"
        "models/finetuning/mednli-t5-deid-tags-100K_reg/" )


for i in ${!array[*]}; do
   echo "Using: ${array[$i]}"

   python src/finetuning/preprocess_mednli.py \
       --mednli-dir data/mednli/ \
       --model-path ${array[$i]} \
       --output-dir ${array2[$i]} \
       --use-constant-adafactor \
       --replace-text-with-tags
done
