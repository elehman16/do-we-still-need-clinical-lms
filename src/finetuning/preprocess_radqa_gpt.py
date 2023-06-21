import torch
import json
import argparse
import pandas as pd
import numpy as np 
from sklearn.metrics import f1_score, accuracy_score
import datasets
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorWithPadding, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer

from collections.abc import Mapping

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from preprocessing.replace_deid_tags import replace_list_of_notes
# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)


def compute_metrics(predictions):
    """Given some predictions, calculate the F1. """
    predictions.label_ids[predictions.label_ids == -100] = tokenizer.pad_token_id
    predictions.predictions[predictions.predictions == -100] = tokenizer.pad_token_id    
    
    # Decode the predictions + labels 
    decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
    decoded_predictions = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    return {
            'f1': np.mean([compute_f1(x, y) for x, y in zip(decoded_predictions, decoded_labels)]),
            'exact': np.mean([compute_exact_match(x, y) for x, y in zip(decoded_predictions, decoded_labels)])
            #'f1': f1_score(decoded_predictions, decoded_labels, average='macro'), 
            #'accuracy': accuracy_score(decoded_predictions, decoded_labels)
           }

def format_single_example(tokenizer, example) -> (str, str, str, int):
    """Format a single example. """
    context = example['context']
    question = example['question']
    answer_lens = [len(x) for x in example['answers']['text']]
    if answer_lens:
        best_option = np.argmax(answer_lens)
        answer = example['answers']['text'][best_option]
        answer_st = example['answers']['answer_start'][best_option] + len("Context: ")
    else:
        answer = ""
        answer_st = -1
    
    return f"Context: {context}", f"\nQuestion: {question}\nAnswer: ", answer, answer_st

def preprocess_function(examples, tokenizer, args, is_test: bool = False):
    """Format the examples and then tokenize them. """
    inputs = [format_single_example(tokenizer, e) for e in examples]
    context, question = [x[0] for x in inputs], [x[1] for x in inputs]
    answers, answer_index = [x[2] for x in inputs], [x[3] for x in inputs]
    
    tokenized_inputs = tokenizer(context,
              question,
              truncation="only_first",
              max_length=1024 if is_test else args.max_seq_length,
              #stride=data_args.doc_stride,
              #return_overflowing_tokens=True,
              return_offsets_mapping=True,
              padding=False)

    inputs, attn_mask, untokenized_labels = [], [], []
    for i in range(len(answers)):
        offset_mapping = tokenized_inputs['offset_mapping'][i]
        
        st = answer_index[i]
        end = answer_index[i] + len(answers[i])
        label_idx_st = [o[0] <= st <= o[1] for o in offset_mapping]
        label_idx_end = [o[0] <= end <= o[1] for o in offset_mapping]
            
        # Nothing cut off   
        if any(label_idx_end):
            inputs.append(tokenized_inputs['input_ids'][i])
            attn_mask.append(tokenized_inputs['attention_mask'][i])
            untokenized_labels.append(answers[i] + tokenizer.eos_token)
        
        # The start is in but not the end! Discard! 
        elif any(label_idx_st) and not is_test:
            continue
        
        # We fucked up
        elif any(label_idx_st) and is_test:
            import pdb; pdb.set_trace()
        else:
            inputs.append(tokenized_inputs['input_ids'][i])
            attn_mask.append(tokenized_inputs['attention_mask'][i])
            untokenized_labels.append(tokenizer.eos_token)
    
    assert(len(untokenized_labels) == len(attn_mask) == len(inputs))
    targets = tokenizer(untokenized_labels)['input_ids']
    target_lens = [len(x) for x in targets]
    lens_ = [len(x) for x in inputs]

    # Make sure they are all the same length
    labels = [[-100] * to_pad + x for to_pad, x in zip(lens_, targets)]

    if not is_test:
        inputs = [x + [tokenizer.pad_token_id] * to_pad for to_pad, x in zip(target_lens, inputs)]
        attn_mask = [x + [0] * to_pad for to_pad, x in zip(target_lens, attn_mask)]

    return {'labels': labels, 'input_ids': inputs, 'attention_mask': attn_mask}

def get_data(radqa_path: str):
    """Get the RADQA data. """
    types = ['train', 'dev', 'test']
    data_files = {}

    for name in types:
        file_ = radqa_path + name + '.csv'
        df = pd.read_csv(file_)
        df['answers'] = [eval(x) for x in df['answers'].values]
        data_files[name] = datasets.Dataset.from_pandas(df)

    raw_datasets = datasets.DatasetDict(data_files)
    return raw_datasets

def get_tokenized_data(dataset_dict: DatasetDict, tokenizer, args):
    """Tokenize stuff. """
    tokenized_dict = {}
    for name, dataset_ in dataset_dict.items():
        print(name)
        processed_item = preprocess_function(dataset_, tokenizer, args, is_test=(name == 'test'))
        tokenized_dict[name] = Dataset.from_dict(processed_item)
        
    return DatasetDict(tokenized_dict)

def preprocess_logits(preds, labels):
    return torch.argmax(preds, dim=-1)

def train_model(model,
                tokenizer,
                output_dir: str,                            
                tokenized_data, 
                args):

    #DataCollatorForSeq2Seq(
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,

    )

    #data_collator = DataCollatorForLMQA(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
            output_dir=output_dir,
            do_train=True,
            do_eval=True,
            do_predict=True,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            save_total_limit=1,
            load_best_model_at_end=True,
            num_train_epochs=10,
            metric_for_best_model='f1',
            overwrite_output_dir=True,
            seed=args.seed,
            report_to='wandb',
            deepspeed=args.deep_speed,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits
    )

    trainer.train()
    #outputs = trainer.predict(tokenized_data["test"])
    #with open(output_dir + '/predict_results.json', 'w') as f:
    #    json.dump(outputs.metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--radqa-dir', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--max-seq-length', type=int, default=512)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--deep-speed', type=str)
    args = parser.parse_args()
    print(f"Running with {args.seed}")

    # Load tokenizer + model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
 
    # Get data and use the tokenizer on the data 
    dataset_dict = get_data(args.radqa_dir)
    tokenized_datasets = get_tokenized_data(dataset_dict, tokenizer, args)

    # Train the model
    train_model(model, tokenizer, args.output_dir, tokenized_datasets, args)  
