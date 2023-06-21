# ClinicalBERT
./scripts/run_clip_encoder_only.sh "emilyalsentzer/Bio_ClinicalBERT" ./models/finetuning/clip_encoder_only_cbert_2e5/ 2e-5 16 1
./scripts/run_clip_encoder_only.sh "emilyalsentzer/Bio_ClinicalBERT" ./models/finetuning/clip_encoder_only_cbert_3e5/ 3e-5 16 1
./scripts/run_clip_encoder_only.sh "emilyalsentzer/Bio_ClinicalBERT" ./models/finetuning/clip_encoder_only_cbert_5e5/ 5e-5 16 1

# ClinicalLongformer
./scripts/run_clip_encoder_only.sh "yikuan8/Clinical-Longformer" ./models/finetuning/clip_encoder_only_clongformer_2e5/ 2e-5 8 2
./scripts/run_clip_encoder_only.sh "yikuan8/Clinical-Longformer" ./models/finetuning/clip_encoder_only_clongformer_3e5/ 3e-5 8 2
./scripts/run_clip_encoder_only.sh "yikuan8/Clinical-Longformer" ./models/finetuning/clip_encoder_only_clongformer_5e5/ 5e-5 8 2

# Finally, gatortron!
./scripts/run_clip_encoder_only.sh "models/gatortron/" ./models/finetuning/clip_encoder_only_gatortron_2e5/ 2e-5 8 2
./scripts/run_clip_encoder_only.sh "models/gatortron/" ./models/finetuning/clip_encoder_only_gatortron_3e5/ 3e-5 8 2
./scripts/run_clip_encoder_only.sh "models/gatortron/" ./models/finetuning/clip_encoder_only_gatortron_5e5/ 5e-5 8 2
