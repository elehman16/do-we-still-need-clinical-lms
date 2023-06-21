import json
import argparse
import pandas as pd
import numpy as np 
from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

from preprocessing.replace_deid_tags import replace_list_of_notes

def compute_metrics(predictions):
    """Given some predictions, calculate the F1. """
    predictions.label_ids[predictions.label_ids == -100] = 0
    
    # Decode the predictions + labels 
    decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
    decoded_predictions = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    return {
            'f1': f1_score(decoded_predictions, decoded_labels, average='macro'), 
            'accuracy': accuracy_score(decoded_predictions, decoded_labels)
           }

def format_single_example(sentence1: str, sentence2: str) -> str:
    """Format a single example. """
    prefix = f"mednli premise: {sentence1} hypothesis: {sentence2.strip()}"
    if not(prefix[-1] == '.'):
        prefix += '.'
    
    return prefix

def preprocess_function(examples, tokenizer, max_seq_length: int, replace_text_with_tags: bool):
    """Format the examples and then tokenize them. """
    inputs = [format_single_example(s1, s2) for s1, s2 in zip(examples['sentence1'], examples['sentence2'])]
    targets = examples['gold_label']

    if replace_text_with_tags:
        inputs = replace_list_of_notes(inputs, 'mimic-iii')

    # 1 token for the SEP between 
    to_remove = [len(tokenizer.tokenize(x)) <= (max_seq_length - 2) for x in inputs]    
    inputs = np.asarray(inputs)[to_remove].tolist()
    targets = np.asarray(targets)[to_remove].tolist()

    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_seq_length, truncation=True)
    return model_inputs

def read_jsonl(file_path: str):
    """Read the given JSONL file."""
    with open(file_path) as f:
        data = [json.loads(line) for line in f]
    
    return data

def get_data(mednli_path: str):
    """Get the mednli data. """
    # mli_dev_v1.jsonl  mli_test_v1.jsonl  mli_train_v1.jsonl
    train = Dataset.from_list(read_jsonl(mednli_path + '/mli_train_v1.jsonl'))
    val = Dataset.from_list(read_jsonl(mednli_path + '/mli_dev_v1.jsonl'))
    test = Dataset.from_list(read_jsonl(mednli_path + '/mli_test_v1.jsonl'))
    return DatasetDict({"train": train, "val": val, "test": test})

def get_tokenized_data(dataset_dict: DatasetDict, tokenizer, max_seq_length: int, replace_text_with_tags: bool):
    """Tokenize stuff. """
    tokenized_dict = {}
    for name, dataset_ in dataset_dict.items():
        processed_item = preprocess_function(dataset_, tokenizer, max_seq_length, replace_text_with_tags)
        tokenized_dict[name] = Dataset.from_dict(processed_item)
        
    return DatasetDict(tokenized_dict)

def train_model(model,
                tokenizer,
                output_dir: str,                            
                tokenized_data, 
                args):

    if args.sample_train_percent != -1:
        tokenized_data['train'] = tokenized_data['train'].train_test_split(args.sample_train_percent)['test']


    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model
    )

    if args.use_constant_adafactor:
        print("Using constant Adafactor as learning rate.")
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            do_train=True,
            do_eval=True,
            do_predict=True,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=1e-4,
            optim="adafactor",
            local_rank=args.local_rank,
            lr_scheduler_type="constant",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            num_train_epochs=15,
            metric_for_best_model='accuracy',
            predict_with_generate=True,
            overwrite_output_dir=True,
            generation_num_beams=3,
            seed=args.seed,
            report_to='wandb'
        )    

    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            do_train=True,
            do_eval=True,
            do_predict=True,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            save_total_limit=1,
            load_best_model_at_end=True,
            num_train_epochs=15,
            metric_for_best_model='accuracy',
            predict_with_generate=True,
            overwrite_output_dir=True,
            generation_num_beams=3,
            seed=args.seed,
            report_to='wandb'
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    outputs = trainer.predict(tokenized_data["test"])
    with open(output_dir + '/predict_results.json', 'w') as f:
        json.dump(outputs.metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mednli-dir', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--max-seq-length', type=int, default=256)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--use-constant-adafactor', action='store_true')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--replace-text-with-tags', action='store_true')
    parser.add_argument('--sample-train-percent', type=float, default=-1)
    args = parser.parse_args()
    print(f"Running with {args.seed}")

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    try: model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    except: model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, from_flax=True)
 
    # Get data and use the tokenizer on the data 
    dataset_dict = get_data(args.mednli_dir)
    tokenized_datasets = get_tokenized_data(dataset_dict, tokenizer, args.max_seq_length, args.replace_text_with_tags)

    # Train the model
    train_model(model, tokenizer, args.output_dir, tokenized_datasets, args)  
