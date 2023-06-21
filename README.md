# clinical_llm

You may need to install jax and jaxlib separately.

`preprocessing_notes_tag_insertion.sh` -> Get data. 
- You will need to get the radqa.csv + discharge.csv from MIMIC-IV. 
- `noteevents.csv` is from MIMIC-III

Recreating the experiments:
- I used the hyperparameters described in the paper.
- Seeds used were all between 1-10. 
- If trying to recreate the experiments exactly, please email `lehmer16@mit.edu`, and I can send you all of my Wandb files, which contain the seeds used.