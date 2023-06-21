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


def prompt_label_type(s, prompt):
    if prompt == '0':
        return f"Context: {s}\nLabel the above sentence as one of the following:\n\nOptions:\n-Appointment-related followup\n-Medication-related followup\n-Lab-related followup\n-Case-specific instructions for the patient\n-Procedure-related followup\n-Imaging-related followup\n-Other helpful contextual information for the patient\n-None of the above"        
    elif prompt == '1':
        return f"Context: {s}\nLabel the above sentence as one of the following:\n\nOptions:\n-Appointment-related followup\n-Medication-related followup\n-Lab-related followup\n-Case-specific instructions for the patient\n-Procedure-related followup\n-Imaging-related followup\n-None of the above"
    elif prompt == '2':
        return f"Context: {s}\nLabel the above sentence as one of the following:\n\nOptions:\n-Appointment-related followup\n-Medication-related Information\n-Lab-related Information\n-Case-specific instructions for the patient\n-Procedure-related followup\n-Imaging-related followup\n-None of the above"
    elif prompt == '3':
        return f"Context: {s}\nLabel the above sentence as one or more of the following, separated by a comma:\n\nOptions:\n-Appointment-related followup\n-Medication-related Information\n-Lab-related Information\n-Case-specific instructions for the patient\n-Procedure-related followup\n-Imaging-related followup\n-None of the above"
    elif prompt == '4':
        return f"Context: {s}\nLabel the above sentence as one or more of the following, delimited by comma:\n\nOptions:\n-Appointment-related followup information\n-Medication-related followup information\n-Lab-related followup information\n-Case-specific instructions for the patient\n-Procedure-related followup information\n-Imaging-related followup information\n-None of the above"
    elif prompt == '5':
        examples = ['He has a follow-up neck CTA and appointment with [ **Month/Year ( 2 ) 1106** ] surgery on 1978-10-18 , with possible subsequent carotid stenting procedure to follow . .'] + [s]
        labels = ['Appointment-related followup, Imaging-related followup, Procedure-related followup'] + [""]
        in_context = [f"Context: {s}\nLabel the above sentence as one or more of the following, delimited by comma:\n\nOptions:\n-Appointment-related followup information\n-Medication-related followup information\n-Lab-related followup information\n-Case-specific instructions for the patient\n-Procedure-related followup information\n-Imaging-related followup information\n-None of the above\n{l}" for s, l in zip(examples, labels)]
        return "\n\n".join(in_context)
    elif prompt == '6':
        examples = ['He has a follow-up neck CTA and appointment with [ **Month/Year ( 2 ) 1106** ] surgery on 1978-10-18 , with possible subsequent carotid stenting procedure to follow . .', 'He is also to be seen by his primary care physician to regulate medications and to restart Plavix approximately two weeks after his surgery .', 'JP drains were left in place under his wound , with planned removal by Thoracics surgery once there was trace to no output .'] + [s]
        labels = ['Appointment-related followup, Imaging-related followup, Procedure-related followup', 'Appointment-related followup, Case-specific instructions for patient, Medication-related followups', 'Appointment-related followup, Other helpful contextual information'] + [""]
        in_context = [f"Context: {s}\nLabel the above sentence as one or more of the following, delimited by comma:\n\nOptions:\n-Appointment-related followup information\n-Medication-related followup information\n-Lab-related followup information\n-Case-specific instructions for the patient\n-Procedure-related followup information\n-Imaging-related followup information\n-None of the above\n{l}" for s, l in zip(examples, labels)]
        return "\n\n".join(in_context) 
    
    elif prompt == '7':
        examples = ['He has a follow-up neck CTA and appointment with [ **Month/Year ( 2 ) 1106** ] surgery on 1978-10-18 , with possible subsequent carotid stenting procedure to follow . .', 'He is also to be seen by his primary care physician to regulate medications and to restart Plavix approximately two weeks after his surgery .', 'JP drains were left in place under his wound , with planned removal by Thoracics surgery once there was trace to no output .'] + [s]
        labels = ['Appointment-related followup, Imaging-related followup, Procedure-related followup', 'Appointment-related followup, Case-specific instructions for patient, Medication-related followups', 'Appointment-related followup, Other helpful contextual information'] + [""]
        in_context = [f"Context: {s}\nLabel the above sentence as one or more of the following, delimited by comma:\n\nOptions:\n-Appointment-related followup information\n-Medication-related followup information\n-Lab-related followup information\n-Case-specific instructions for the patient\n-Medical procedure-related followup information\n-Medical imaging-related followup information\n-None of the above\n{l}" for s, l in zip(examples, labels)]
        return "\n\n".join(in_context)

    elif prompt == '8':
        examples = ['He has a follow-up neck CTA and appointment with [ **Month/Year ( 2 ) 1106** ] surgery on 1978-10-18 , with possible subsequent carotid stenting procedure to follow . .', 'He is also to be seen by his primary care physician to regulate medications and to restart Plavix approximately two weeks after his surgery .', 'JP drains were left in place under his wound , with planned removal by Thoracics surgery once there was trace to no output .'] + [s]
        labels = ['Appointment-related followup, Imaging-related followup, Procedure-related followup', 'Appointment-related followup, Case-specific instructions for patient, Medication-related followups', 'Appointment-related followup, Other helpful contextual information'] + [""]
        in_context = [f"Context: {s}\nLabel the above sentence as one or more of the following, delimited by comma:\n\nOptions:\n-Appointment-related followup information\n-Medication-related followup information\n-Lab-related followup information\n-Case-specific instructions for the patient\n-Medical procedure-related followup information\n-Medical imaging-related followup information\n-None of the above\n{l}" for s, l in zip(examples, labels)]
        return "\n\n".join(in_context)
 
def tokenize_data(tokenizer, df: pd.DataFrame, max_seq_length: int, prompt: str) -> pd.DataFrame:
    """Split the data into chunks of `max_seq_length`.  Attempt to take an equal number
    of tokens before and after the text.
    @param tokenizer 
    @param replace_text_with_tags
    @param max_seq_length """
    inputs, labels, ids = [], [], []

    j = 0 
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = row.text 
        
        # 
        for s in row.labels:
            sent = row.text[s[0]:s[1]]
            all_prompts = prompt_label_type(sent, prompt)
            idx_labels = str_tags_to_binary_tensor(s[2].split(", "))
            inputs.append(all_prompts)
            labels.append(idx_labels)
            ids.append(row.note_id)

            j += 1

    tokenized_inputs = tokenizer(inputs, text_target=["" for x in inputs])
    tokenized_inputs['idx_labels'] = labels
    tokenized_inputs['id'] = ids
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
    # Appointment related followup\n
    #-Medication-related followup\n
    #-Lab-related followup\n
    #-Case-specific instructions for the patient\n
    #-Procedure-related followup\n
    #-Imaging-related followup\n
    #-Other helpful contextual information for the patient\n
    #-None of the above
    # ['Appointment-related followup',
    #    'Medication-related followups',
    #    'Other helpful contextual information',
    #    'Lab-related followup',
    #           'Case-specific instructions for patient',
    #           'Procedure-related followup',
    #           'Imaging-related followup']

    mapper = {
              'Appointment-related followup': LABEL_TYPES[0],
              'Appointment-related followup information': LABEL_TYPES[0], 
              'Medication-related followup': LABEL_TYPES[1], 
              'Medication-related Information': LABEL_TYPES[1],
              'Medication-related followup information': LABEL_TYPES[1],
              'Lab-related Information': LABEL_TYPES[3],
              'Lab-related followup information': LABEL_TYPES[3],
              'Lab-related instructions for the patient': LABEL_TYPES[3],
              'Case-specific instructions for the patient': LABEL_TYPES[4],
              'Procedure-related followup': LABEL_TYPES[5],
              'Imaging-related followup': LABEL_TYPES[6],
              'Procedure-related followup information': LABEL_TYPES[5],
              'Imaging-related followup information': LABEL_TYPES[6],
              'Medical procedure-related followup information': LABEL_TYPES[5],
              'Medical imaging-related followup information': LABEL_TYPES[6],
              'Other helpful contextual information': LABEL_TYPES[2]
             }

    preds = []
    for d in decoded_predictions:
        str_pred = []
        for p in d.split(', '):
            if p in mapper:
                str_pred.append(mapper[p])
            else:
                str_pred.append(p)

        preds.append(", ".join(str_pred)) 

    preds = [str_tags_to_binary_tensor([x]) for x in preds]
    labels = examples['idx_labels']
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
        ds.shuffle()        

        ds = Dataset.from_dict(ds[:200])
        outputs = trainer.predict(ds)

        metrics = compute_metrics_prompt(outputs.predictions, ds, args.prompt)
    else:
        outputs = trainer.predict(tokenized_data['test'])
        
    print(outputs.metrics)
    print(metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-dir', type=str, required=True)
    parser.add_argument('--model-path', type=str, default="google/flan-t5-xxl")
    parser.add_argument('--max-seq-length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--output-dir', type=str, default="testing-testing/")
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
