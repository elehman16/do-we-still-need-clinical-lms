for i in {1..3}
do
    echo "Storing in models/finetuning/clip-t5-sci-1e-4/seed-$i/"
    python src/finetuning/preprocess_clip.py \
        --clip-dir ./data/clip_data/ \
        --model-path razent/SciFive-base-Pubmed \
        --output-dir "models/finetuning/clip-t5-sci-1e-5/seed-$i/" \
        --lr 1e-5 \
        --seed $i 
done

for i in {1..3}
do
    echo "Storing in models/finetuning/clip-t5-sci-1e-5/seed-$i/"
    python src/finetuning/preprocess_clip.py \
        --clip-dir ./data/clip_data/ \
        --model-path t5-base \
        --output-dir "models/finetuning/clip-t5-base-1e-5/seed-$i/" \
        --lr 1e-5 \
        --seed $i
done
