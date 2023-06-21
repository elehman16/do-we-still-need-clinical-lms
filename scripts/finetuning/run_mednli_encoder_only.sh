

for lr in "2e-5" "3e-5" "5e-5"
do 
   for i in {1..5}
   do
       echo "models/finetuning/mednli-clinical-bert-$lr-seed-$i/"
       python src/finetuning/preprocess_mednli_encoder_only.py \
           --mednli-dir ./data/mednli/ \
           --lr $lr \
           --model-path "emilyalsentzer/Bio_ClinicalBERT" \
           --output-dir "models/finetuning/mednli-clinical-bert-$lr-seed-$i/" \
           --seed $i
   done
done

for lr in "2e-5" "3e-5" "5e-5"
do
   for i in {1..5}
   do
       echo "models/finetuning/mednli-clinical-longformer-$lr-seed-$i/"
       python src/finetuning/preprocess_mednli_encoder_only.py \
           --mednli-dir ./data/mednli/ \
           --lr $lr \
           --model-path "yikuan8/Clinical-Longformer" \
           --output-dir "models/finetuning/mednli-clinical-longformer-$lr-seed-$i/" \
           --seed $i
   done
done


for lr in "2e-5" "3e-5" "5e-5"
do
    for i in {1..5}
    do
        echo "models/finetuning/mednli-gatortron-$lr-seed-$i/"
        python -m torch.distributed.launch --nproc_per_node 4 src/finetuning/preprocess_mednli_encoder_only.py \
            --mednli-dir ./data/mednli/ \
            --lr $lr \
            --model-path "models/gatortron/" \
            --output-dir "models/finetuning/mednli-gatortron-$lr-seed-$i/" \
            --seed $i
    done
done


