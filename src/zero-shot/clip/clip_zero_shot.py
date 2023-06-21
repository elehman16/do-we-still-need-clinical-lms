import json
import argparse
import pandas as pd
import numpy as np 
from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from tqdm import tqdm
import random

def conversion(prediction_str: str) -> str:
    mapper = {'N/A': ''}    
    if prediction_str in mapper:
        return mapper[prediction_str]
    else:
        return prediction_str


LABEL_TYPES = ['Appointment-related followup',
               'Medication-related followups',
               'Other helpful contextual information',
               'Lab-related followup',
               'Case-specific instructions for patient',
               'Procedure-related followup',
               'Imaging-related followup']


LOWERCASED_TYPES = [x.lower() for x in LABEL_TYPES]


def str_tags_to_binary_tensor(los):
    """Convert list of strings to indexed positions. """
    arr = np.zeros(len(LABEL_TYPES))
    for str_ in los:
        if not str_ in LABEL_TYPES and not str_ in LOWERCASED_TYPES: continue

        # It's in our list. Get the label. Mark as 1 in our label list.
        if str_ in LOWERCASED_TYPES: arr[LOWERCASED_TYPES.index(str_)] = 1
        else: arr[LABEL_TYPES.index(str_)] = 1

    return arr


def load_ids(clip_path):
    """Load the training/val/test ids. """
    tr_id = pd.read_csv(clip_path + '/train_ids.csv', header=None)
    vl_id = pd.read_csv(clip_path + '/val_ids.csv', header=None)
    te_id = pd.read_csv(clip_path + '/test_ids.csv', header=None)
    return set(tr_id[0].values), set(vl_id[0].values), set(te_id[0].values)

LABEL_TYPES = ['Appointment-related followup',
               'Medication-related followups',
               'Other helpful contextual information',
               'Lab-related followup',
               'Case-specific instructions for patient',
               'Procedure-related followup',
               'Imaging-related followup']
def prompt_label_type(s, label_type, prompt):
    if prompt == '0':
        return f"Context: {s}\nIs above sentence a {label_type}? Options:\n-Yes\n-No"        
    elif prompt == '1':
        return f'Does this sentence contain any information about a {label_type}: "{s}"? Options:\n-Yes\n-No'
    elif prompt == '2':
        if label_type == LABEL_TYPES[0]:
            return f'Context: {s}\n\nDoes the above sentence contain any information about an appointment? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[1]:
            return f'Context: {s}\n\nDoes the above sentence contain any information about medication instructions? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[2]:
            return f'Context: {s}\n\nDoes the above sentence contain any information about any very general followups? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[3]:
            return f'Context: {s}\n\nDoes the above sentence contain any information about any lab followups? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[4]:
            return f'Context: {s}\n\nDoes the above sentence contain any information about any case-specific instructions? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[5]:
            return f'Context: {s}\n\nDoes the above sentence contain any information about any future procedures? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[6]:
            return f'Context: {s}\n\nDoes the above sentence contain any information about any future imagings? Options:\n-Yes\n-No'
    elif prompt == '3':
        if label_type == LABEL_TYPES[0]:
            return f'Context: {s}\n\nDoes the above sentence match this description: "Appointments to be made by the PCP, or monitored to ensure the patient attends them."? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[1]:
            return f'Context: {s}\n\nDoes the above sentence match this description: "Medications that the PCP either needs to ensure that the patient is taking correctly, e.g. time-limited medications or new medications that may need dose adjustment."? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[2]:
            return f'Context: {s}\n\nDoes the above sentence match this description: "Other actionable information that is important to relay to the PCP but does not fall under existing aspects"? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[3]:
            return f'Context: {s}\n\nDoes the above sentence match this description: "Laboratory tests that either have results pending or need to be ordered by the PCP."? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[4]:
            return f'Context: {s}\n\nDoes the above sentence match this description: "Post-discharge instructions that are directed to the patient, so the PCP can ensure the patient understands and performs them."? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[5]:
            return f'Context: {s}\n\nDoes the above sentence match this description: "Procedures that the PCP needs to either order, ensure another caregiver orders, or ensure the patient undergoes."? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[6]:
            return f'Context: {s}\n\nDoes the above sentence match this description: "Imaging studies that either have results pending or need to be ordered by the PCP."? Options:\n-Yes\n-No'
    elif prompt == '3':
        if label_type == LABEL_TYPES[0]:
            return f'Context: {s}\n\nDoes the above sentence match this description: "Appointments to be made by the PCP, or monitored to ensure the patient attends them."? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[1]:
            return f'Context: {s}\n\nDoes the above sentence match this description: "Medications that the PCP either needs to ensure that the patient is taking correctly, e.g. time-limited medications or new medications that may need dose adjustment."? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[2]:
            return f'Context: {s}\n\nDoes the above sentence match this description: "Other actionable information that is important to relay to the PCP but does not fall under existing aspects"? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[3]:
            return f'Context: {s}\n\nDoes the above sentence match this description: "Laboratory tests that either have results pending or need to be ordered by the PCP."? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[4]:
            return f'Context: {s}\n\nDoes the above sentence match this description: "Post-discharge instructions that are directed to the patient, so the PCP can ensure the patient understands and performs them."? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[5]:
            return f'Context: {s}\n\nDoes the above sentence match this description: "Procedures that the PCP needs to either order, ensure another caregiver orders, or ensure the patient undergoes."? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[6]:
            return f'Context: {s}\n\nDoes the above sentence match this description: "Imaging studies that either have results pending or need to be ordered by the PCP."? Options:\n-Yes\n-No'

    elif prompt == '4':
        if label_type == LABEL_TYPES[0]:
            return f'Context: {s}\n\nDoes the above sentence contain information about appointments to be made or monitored by the doctor? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[1]:
            return f'Context: {s}\n\nDoes the above sentence contain information about medications that the docto needs to ensure the patient needs to take? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[2]:
            return f'Context: {s}\n\nDoes the above sentence contain any important actional information? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[3]:
            return f'Context: {s}\n\nDoes the above sentence contain any information about laboratory tests that either have results pending or need to be ordered? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[4]:
            return f'Context: {s}\n\nDoes the above sentence contain any information about post-discharge instructions? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[5]:
            return f'Context: {s}\n\nDoes the above sentence contain any information about procedures? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[6]:
            return f'Context: {s}\n\nDoes the above sentence contain any information about imaging studies that are pending or need to be ordered? Options:\n-Yes\n-No' 
    elif prompt == '5':
        if label_type == LABEL_TYPES[0]:
            return f'Context: {s}\n\nDoes the above sentence contain information about current or future appointments? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[1]:
            return f'Context: {s}\n\nDoes the above sentence contain information about medications? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[2]:
            return f'Context: {s}\n\nDoes the above sentence contain any important actional information? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[3]:
            return f'Context: {s}\n\nDoes the above sentence contain any information about laboratory tests? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[4]:
            return f'Context: {s}\n\nDoes the above sentence contain any information about what to do post-discharge? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[5]:
            return f'Context: {s}\n\nDoes the above sentence contain any information about procedures (e.g., surgeries)? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[6]:
            return f'Context: {s}\n\nDoes the above sentence contain any information about radiology or imaging? Options:\n-Yes\n-No' 

    elif prompt == '6':
        if label_type == LABEL_TYPES[0]:
            prompt = "Context: The patient requires a neurology consult at XYZ for evaluation.\n\nDoes the above sentence contain information about current or future appointments? Options:\n-Yes\n-No\nYes\n\n"
            return prompt + f'Context: {s}\n\nDoes the above sentence contain information about current or future appointments? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[1]:
            prompt = 'Context: The patient was instructed to hold ASA and refrain from NSAIDs for 2 weeks.\n\nDoes the above sentence contain information about medications? Options:\n-Yes\n-No\nYes\n\n'
            return prompt + f'Context: {s}\n\nDoes the above sentence contain information about medications? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[2]:
            tmp = "Since the patient has been struggling to gain weight this past year, we will monitor his nutritional status and trend weights closely"
            prompt = f'Context: {tmp}\n\nDoes the above sentence contain any important actional information? Options:\n-Yes\n-No\nYes\n\n'
            return prompt + f'Context: {s}\n\nDoes the above sentence contain any important actional information? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[3]:
            tmp = "We ask that the patients’ family physician repeat these tests in 2 weeks to ensure resolution."
            prompt = f'Context: {tmp}\n\nDoes the above sentence contain any information about laboratory tests? Options:\n-Yes\n-No\nYes\n\n'
            return prompt + f'Context: {s}\n\nDoes the above sentence contain any information about laboratory tests? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[4]:
            tmp = "No driving until post-op visit and you are no longer taking pain medications."
            prompt = f"Context: {tmp}\n\nDoes the above sentence contain any information about what to do post-discharge? Options:\n-Yes\n-No\nYes\n\n"
            return prompt + f'Context: {s}\n\nDoes the above sentence contain any information about what to do post-discharge? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[5]:
            tmp = "Please follow-up for EGD with GI."
            prompt += f'Context: {tmp}\n\nDoes the above sentence contain any information about procedures (e.g., surgeries)? Options:\n-Yes\n-No\nYes\n\n'
            return f'Context: {s}\n\nDoes the above sentence contain any information about procedures (e.g., surgeries)? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[6]:
            tmp = "Superior segment of the left lower lobe: rounded density which could have been related to infection, but follow-up for resolution recommended to exclude possible malignancy"
            prompt = f'Context: {tmp}\n\nDoes the above sentence contain any information about an imaging followup? Options:\n-Yes\n-No\nYes\n\n'
            return f'Context: {s}\n\nDoes the above sentence contain any information about an imaging followup? Options:\n-Yes\n-No'
        else:
            raise ValueError("Not implemented")

    elif prompt == '7':
        if label_type == LABEL_TYPES[0]:
            prompt = "Context: The patient requires a neurology consult at XYZ for evaluation.\n\nDoes the above sentence contain information about current or future appointments? Options:\n-Yes\n-No\nYes\n\n"
            return prompt + f'Context: {s}\n\nDoes the above sentence contain information about current or future appointments? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[1]:
            tmp = "The patient was instructed to hold ASA and refrain from NSAIDs for 2 weeks."
            prompt = f'Context: {tmp}\n\nDoes the above sentence contain information about medication dosing instructions? Options:\n-Yes\n-No\nYes\n\n'
            return prompt + f'Context: {s}\n\nDoes the above sentence contain information about medication dosing instructions? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[2]:
            tmp = "Since the patient has been struggling to gain weight this past year, we will monitor his nutritional status and trend weights closely"
            prompt = f'Context: {tmp}\n\nDoes the above sentence contain any important actional information? Options:\n-Yes\n-No\nYes\n\n'
            return prompt + f'Context: {s}\n\nDoes the above sentence contain any important actional information? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[3]:
            tmp = "We ask that the patients’ family physician repeat these tests in 2 weeks to ensure resolution."
            prompt = f'Context: {tmp}\n\nDoes the above sentence contain any information about laboratory tests and the results? Options:\n-Yes\n-No\nYes\n\n'
            return prompt + f'Context: {s}\n\nDoes the above sentence contain any information about laboratory tests and the results? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[4]:
            tmp = "No driving until post-op visit and you are no longer taking pain medications."
            prompt = f"Context: {tmp}\n\nDoes the above sentence contain any instructions for after hospital discharge? Options:\n-Yes\n-No\nYes\n\n"
            return prompt + f'Context: {s}\n\nDoes the above sentence contain any instructions for after hospital discharge? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[5]:
            tmp = "Please follow-up for EGD with GI."
            prompt += f'Context: {tmp}\n\nDoes the above sentence contain any information about future procedures? Options:\n-Yes\n-No\nYes\n\n'
            return f'Context: {s}\n\nDoes the above sentence contain any information about future procedures? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[6]:
            tmp = "Superior segment of the left lower lobe: rounded density which could have been related to infection, but follow-up for resolution recommended to exclude possible malignancy"
            prompt = f'Context: {tmp}\n\nDoes the above sentence contain any information about a medical imaging followup? Options:\n-Yes\n-No\nYes\n\n'
            return f'Context: {s}\n\nDoes the above sentence contain any information about a medical imaging followup? Options:\n-Yes\n-No'
        else:
            raise ValueError("Not implemented")


    elif prompt == '8':
        if label_type == LABEL_TYPES[0]:
            prompt = "The patient requires a neurology consult at XYZ for evaluation.\n\nDoes the above sentence contain information about current or future appointments? Options:\n-Yes\n-No\nYes\n\n"
            return prompt + f'{s}\n\nDoes the above sentence contain information about current or future appointments? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[1]:
            prompt = 'The patient was instructed to hold ASA and refrain from NSAIDs for 2 weeks.\n\nDoes the above sentence contain information about medications? Options:\n-Yes\n-No\nYes\n\n'
            return prompt + f'{s}\n\nDoes the above sentence contain information about medications? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[2]:
            tmp = "Since the patient has been struggling to gain weight this past year, we will monitor his nutritional status and trend weights closely"
            prompt = f'{tmp}\n\nDoes the above sentence contain any important actional information? Options:\n-Yes\n-No\nYes\n\n'
            return prompt + f'{s}\n\nDoes the above sentence contain any important actional information? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[3]:
            tmp = "We ask that the patients’ family physician repeat these tests in 2 weeks to ensure resolution."
            prompt = f'{tmp}\n\nDoes the above sentence contain any information about laboratory tests? Options:\n-Yes\n-No\nYes\n\n'
            return prompt + f'{s}\n\nDoes the above sentence contain any information about laboratory tests? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[4]:
            tmp = "No driving until post-op visit and you are no longer taking pain medications."
            prompt = f"{tmp}\n\nDoes the above sentence contain any information about what to do post-discharge? Options:\n-Yes\n-No\nYes\n\n"
            return prompt + f'{s}\n\nDoes the above sentence contain any information about what to do post-discharge? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[5]:
            tmp = "Please follow-up for EGD with GI."
            prompt += f'{tmp}\n\nDoes the above sentence contain any information about procedures (e.g., surgeries)? Options:\n-Yes\n-No\nYes\n\n'
            return f'{s}\n\nDoes the above sentence contain any information about procedures (e.g., surgeries)? Options:\n-Yes\n-No'
        elif label_type == LABEL_TYPES[6]:
            tmp = "Superior segment of the left lower lobe: rounded density which could have been related to infection, but follow-up for resolution recommended to exclude possible malignancy"
            prompt = f'{tmp}\n\nDoes the above sentence contain any information about an imaging followup? Options:\n-Yes\n-No\nYes\n\n'
            return f'{s}\n\nDoes the above sentence contain any information about an imaging followup? Options:\n-Yes\n-No'
    
    elif prompt == '9':
        if label_type == LABEL_TYPES[0]:
            examples = [('The patient requires a neurology consult at MGH for evaluation', 'Yes'), (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain information about current or future appointments? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])                
 
        elif label_type == LABEL_TYPES[1]:
            examples = [('The patient was instructed to hold ASA and refrain from NSAIDs for 2 weeks', 'Yes'), (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain information about medications? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[2]:
            examples = [('Since the patient has been struggling to gain weight this past year, we will monitor his nutritional status and trend weights closely', 'Yes'), (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain any important actional information? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[3]:
            examples = [('We ask that the patients’ family physician repeat these tests in 2 weeks to ensure resolution', 'Yes'), (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain any information about laboratory tests? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])
        
        elif label_type == LABEL_TYPES[4]:
            examples = [('No driving until post-op visit and you are no longer taking pain medications', 'Yes'), (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain any information about what to do post-discharge? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[5]:
            examples = [('Please follow-up for EGD with GI.', 'Yes'), (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain any information about current or future procedures? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[6]:
            examples = [('Superior segment of the left lower lobe: rounded density which could have been related to infection, but follow-up for resolution recommended to exclude possible malignancy', 'Yes'), (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain any information about an imaging followup? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        else:
            raise ValueError("Not implemented")

    elif prompt == '9':
        if label_type == LABEL_TYPES[0]:
            examples = [('The patient requires a neurology consult at MGH for evaluation', 'Yes'), (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain information about current or future appointments? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[1]:
            examples = [('The patient was instructed to hold ASA and refrain from NSAIDs for 2 weeks', 'Yes'), (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain information about medication instructions? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[2]:
            examples = [('Since the patient has been struggling to gain weight this past year, we will monitor his nutritional status and trend weights closely', 'Yes'), (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain any important actional information? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[3]:
            examples = [('We ask that the patients’ family physician repeat these tests in 2 weeks to ensure resolution', 'Yes'), (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain any information about laboratory tests? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[4]:
            examples = [('No driving until post-op visit and you are no longer taking pain medications', 'Yes'), (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain any information about what to do post-discharge? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[5]:
            examples = [('Please follow-up for EGD with GI.', 'Yes'), (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain any information about current or future procedures? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[6]:
            examples = [('Superior segment of the left lower lobe: rounded density which could have been related to infection, but follow-up for resolution recommended to exclude possible malignancy', 'Yes'), (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain any information about an imaging followup? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        else:
            raise ValueError("Not implemented")

    elif prompt == '10':
        if label_type == LABEL_TYPES[0]:
            examples = [('This can be done when she is seen at the Norton Audubon Hospital Clinic', 'Yes'), 
                        ('The patient requires a neurology consult at MGH for evaluation', 'Yes'), 
                        (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain information about current or future appointments? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[1]:
            examples = [('Furosemide 20 mg Tablet Sig : One ( 1 ) Tablet PO Q12H ( every 12 hours ) for 7 days .', 'Yes'), 
                        ('The patient was instructed to hold ASA and refrain from NSAIDs for 2 weeks', 'Yes'), 
                        (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain information about medication dosing instructions? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[2]:
            examples = [('The patient will be transferred to the care of Dr. Lisa Larsen who can be reached at telephone # 765-307-9409', 'Yes'), 
                        ('Since the patient has been struggling to gain weight this past year, we will monitor his nutritional status and trend weights closely', 'Yes'),
                        (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain any important actional information? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[3]:
            examples = [('8 . Outpatient Lab Work Theophylline level on 1996-09-23 ; Please call level to Dr. Chapman at Phone : 333-335-6909 Fax : 19-993-6386', 'Yes'), 
                        ('We ask that the patients’ family physician repeat these tests in 2 weeks to ensure resolution', 'Yes'),
                        (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain any information about laboratory tests? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[4]:
            examples = [('No driving until post-op visit and you are no longer taking pain medications', 'Yes'), 
                        ('This can be done when she is seen at the Norton Audubon Hospital Clinic', 'Yes'),
                        (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain any information about what to do post-discharge? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[5]:
            examples = [('Neurosurgery felt the patient was stable enough to f/u as an opt for a shunt .', 'Yes'), 
                        ('Please follow-up for EGD with GI.', 'Yes'),
                        (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain any information about current or future procedures? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        elif label_type == LABEL_TYPES[6]:
            examples = [('Please make this appointment by calling 492-230-1 . ? ? ? ? ? ? You have an appointment in the Brain Erie County Medical Center Clinic on 03-13 , MRI at 11:15 am Cardiology 4 and clinic at 1 pm .', 'Yes'), 
                        ('Superior segment of the left lower lobe: rounded density which could have been related to infection, but follow-up for resolution recommended to exclude possible malignancy', 'Yes'),
                        (s, "")]
            prompt = 'Context: {}\n\nDoes the above sentence contain any information about an imaging followup? Options:\n-Yes\n-No\n{}'
            return '\n\n'.join([prompt.format(x[0], x[1]) for x in examples])

        else:
            raise ValueError("Not implemented")

def tokenize_data(tokenizer, df: pd.DataFrame, max_seq_length: int, prompt: str) -> pd.DataFrame:
    """Split the data into chunks of `max_seq_length`.  Attempt to take an equal number
    of tokens before and after the text.
    @param tokenizer 
    @param replace_text_with_tags
    @param max_seq_length """
    inputs, labels, ids, sentence_ids = [], [], [], []

    j = 0 
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = row.text 
        
        # 
        for s in row.labels:
            sent = row.text[s[0]:s[1]]
            all_prompts = [prompt_label_type(sent, l, prompt) for l in LABEL_TYPES]
            idx_labels = str_tags_to_binary_tensor(s[2].split(", "))
            inputs.extend(all_prompts)
            labels.extend([idx_labels] + [[] for x in range(len(LABEL_TYPES) - 1)])
            ids.extend([row.note_id for x in range(len(LABEL_TYPES))])  
            sentence_ids.extend([j for x in range(len(LABEL_TYPES))])

            j += 1

    tokenized_inputs = tokenizer(inputs, text_target=["" for x in inputs])
    tokenized_inputs['idx_labels'] = labels
    tokenized_inputs['id'] = ids
    tokenized_inputs['sentence_ids'] = sentence_ids
    return Dataset.from_dict(tokenized_inputs).to_pandas()


def load_data(clip_path) -> pd.DataFrame:
    """Load the data from the sentences.csv. """
    df = pd.read_csv(clip_path + '/sentence_level.csv')
    df['sentence'] = [eval(x) for x in df['sentence']]
    df['labels'] = [eval(x) for x in df['labels']]

    # Combine the text, remember sentence offsets. 
    docs = {'note_id': [], 'text': [], 'labels': []}
    for id, group in df.groupby('doc_id'):
        text, sentence_offsets = "", []

        for _, row in group.iterrows():
            sent = ' '.join(row['sentence'])

            # Remove the weird `I-` in front of the labels
            row['labels'] = [x[2:] for x in row['labels']]
            row['labels'].sort()
            labels = ', '.join(row['labels'])
            sentence_offsets.append((len(text), len(text) + len(sent), labels))

            # Now join the text
            text += sent + ' '

        docs['note_id'].append(id)
        docs['text'].append(text)
        docs['labels'].append(sentence_offsets)

    return pd.DataFrame(docs)


def get_data(tokenizer, clip_path, max_seq_length, prompt) -> DatasetDict:
    """Get the CLIP data. 
    @param tokenizer is a Huggingface tokenizer.
    @param clip_path is the path to the clip data.
    @param replace_text_with_tags determines whether or not we modify the text to remove PHI.
    @param max_seq_length is the maximum sequence length."""
    df = load_data(clip_path)
    tr_id, vl_id, te_id = load_ids(clip_path)

    # Split the data into chunks + into train/val/test
    model_inputs = tokenize_data(tokenizer, df, max_seq_length, prompt)
    train = model_inputs[model_inputs['id'].isin(tr_id)]
    val = model_inputs[model_inputs['id'].isin(vl_id)]
    test = model_inputs[model_inputs['id'].isin(te_id)]

    # Create the dataset objects and return 
    input_dict = {'train': Dataset.from_pandas(train), 'val': Dataset.from_pandas(val), 'test': Dataset.from_pandas(test)}
    return DatasetDict(input_dict)


def compute_metrics_prompt(predictions, examples, prompt):
    """Given some predictions, calculate the F1. """
    # Decode the predictions + labels 
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
 
    # It's generating too much! Take the first word. 
    print(set(decoded_predictions))
    preds = np.asarray([1 if p == 'Yes' else 0 for p in decoded_predictions])

    labels = np.asarray([x for x in examples['idx_labels'] if x])
    preds = preds.reshape(labels.shape)
    return {'macro_f1': f1_score(labels, preds, average='macro'), 'micro_f1': f1_score(labels, preds, average='micro'), 'per_class': f1_score(labels, preds, average=None)}



def get_tokenized_data(dataset_dict: DatasetDict, tokenizer, max_seq_length: int, prompt: str):
    """Tokenize stuff. """
    tokenized_dict = {}
    for name, dataset_ in dataset_dict.items():
        processed_item = preprocess_function(dataset_, tokenizer, max_seq_length, prompt)
        processed_item['answers'] = dataset_['answers']
        tokenized_dict[name] = Dataset.from_dict(processed_item)
        tokenized_dict['answers'] = None        

    return DatasetDict(tokenized_dict)


def test_model(model,
               tokenizer,
               output_dir: str,                            
               tokenized_data, 
               seed: int,
               local_rank: int,
               prompt: str):

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            do_train=False,
            do_eval=False,
            do_predict=True,
            local_rank=local_rank,
            per_device_eval_batch_size=1,
            predict_with_generate=True,
            overwrite_output_dir=True,
            seed=seed,
    )    

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        #compute_metrics=lambda x: compute_metrics_prompt(x, args.prompt)
    )
    
    if args.set == 'dev':
        ds = tokenized_data['val']
        random.seed(0)
        ids = list(set(ds['sentence_ids']))
        random.shuffle(ids)
        ids = ids[:100]        

        filtered_ds = ds.filter(lambda x: x['sentence_ids'] in ids)
        outputs = trainer.predict(filtered_ds)

        metrics = compute_metrics_prompt(outputs.predictions, filtered_ds, args.prompt)
    else:
        ds = tokenized_data['val']
        random.seed(0)
        ids = list(set(ds['sentence_ids']))
        random.shuffle(ids)
        ids = ids[:int(len(ids) * 0.25)]

        filtered_ds = ds.filter(lambda x: x['sentence_ids'] in ids)
        outputs = trainer.predict(filtered_ds)
        metrics = compute_metrics_prompt(outputs.predictions, filtered_ds, args.prompt)       
 
    print(outputs.metrics)
    print(metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-dir', type=str, required=True)
    parser.add_argument('--model-path', type=str, default="google/flan-t5-xxl")
    parser.add_argument('--max-seq-length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--set', required=True, choices=['dev', 'test'])
    args = parser.parse_args()
    print(f"Running with {args.seed}")

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
 
    # Get data and use the tokenizer on the data 
    tokenized_datasets = get_data(tokenizer, args.clip_dir, args.max_seq_length, args.prompt)

    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

    # Train the model
    test_model(model, tokenizer, args.output_dir, tokenized_datasets, args.seed, args.local_rank, args.prompt)  
