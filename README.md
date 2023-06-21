# Do We Still Need Clinical Language Models?
Paper Link: https://arxiv.org/abs/2302.08091 <br>
PhysioNet Link: https://physionet.org/content/clinical-t5/1.0.0/ <br>
Citation: 
```
@article{Lehman2023DoWS,
  title={Do We Still Need Clinical Language Models?},
  author={Eric P. Lehman and Evan Hernandez and Diwakar Mahajan and Jonas Wulff and Micah J. Smith and Zachary M. Ziegler and Daniel Nadler and Peter Szolovits and Alistair E. W. Johnson and Emily Alsentzer},
  journal={ArXiv},
  year={2023},
  volume={abs/2302.08091}
}
```

## Setup
For this paper, I used Python 3.9. Install the `requirements.txt`. You may need to install jax and jaxlib separately.

## Pre-training the Models
Most of the code for this is from Huggingface.
`preprocessing_notes_tag_insertion.sh` -> Get data. 
- You will need to get the radqa.csv + discharge.csv from MIMIC-IV. 
- `noteevents.csv` is from MIMIC-III.

## Re-Creating Experiments
The hyperparameters are described fully in the paper. However, for the T5 models, we used a learning rate of 1e-4. For BioClinRoBERTa, we also explore the hyperparameters suggested in their paper. We did a similar thing for PubMedGPT.

## Exact Re-creation
Seeds used were all between 1-10, but mostly were 1-3. There might have been some cases in which 4-9 were used, but I would need to go back and check. If trying to recreate the experiments exactly, please email `lehmer16@mit.edu`, and I can send you all of my Wandb files, which contain the seeds used. I have them all in my folder, but just don't have the time right now to go through and record.

## Misc Issues
Feel free to email `lehmer16@mit.edu` with any questions / concerns.
