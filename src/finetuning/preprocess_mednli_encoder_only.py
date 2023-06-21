import torch
import json
import argparse
import pandas as pd
import numpy as np 
from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, DataCollatorWithPadding

def compute_metrics(predictions):
    """Given some predictions, calculate the F1. """
    argmaxed_predictions = np.argmax(predictions.predictions, axis=-1)
    #argmaxed_predictions = predictions.predictions
    return {
            'accuracy': accuracy_score(argmaxed_predictions, predictions.label_ids)
           }

def format_single_example(sentence1: str, sentence2: str) -> str:
    """Format a single example. """
    prefix = f"mednli premise: {sentence1} hypothesis: {sentence2.strip()}"
    if not(prefix[-1] == '.'):
        prefix += '.'
    
    return prefix

def preprocess_function(examples, tokenizer, max_seq_length: int, is_train=True):
    """Format the examples and then tokenize them. """
    inputs = [format_single_example(s1, s2) for s1, s2 in zip(examples['sentence1'], examples['sentence2'])]
    targets = examples['gold_label']

    # 1 token for the SEP between
    nli_mapping = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    to_remove = [len(tokenizer.tokenize(x)) <= (max_seq_length - 2) for x in inputs]
    if is_train:
        to_remove = [len(tokenizer.tokenize(x)) <= (max_seq_length - 2) for x in inputs]    
        inputs = np.asarray(inputs)[to_remove].tolist()
    else:
        print(f"Warning: Would've removed {sum([len(tokenizer.tokenize(x)) > (max_seq_length - 2) for x in inputs])} instances")

    targets = [nli_mapping[x] for x in np.asarray(targets)[to_remove].tolist()]
    model_inputs = tokenizer(inputs, max_length=max_seq_length, truncation=True)
    model_inputs['labels'] = targets 
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

def get_tokenized_data(dataset_dict: DatasetDict, tokenizer, max_seq_length: int):
    """Tokenize stuff. """
    tokenized_dict = {}
    for name, dataset_ in dataset_dict.items():
        is_train = (name == 'train')
        processed_item = preprocess_function(dataset_, tokenizer, max_seq_length, is_train=is_train)
        tokenized_dict[name] = Dataset.from_dict(processed_item)
        
    return DatasetDict(tokenized_dict)

def train_model(model,
                tokenizer,
                output_dir: str,                            
                tokenized_data, 
                args):

    if args.sample_train_percent != -1:    
        tokenized_data['train'] = tokenized_data['train'].train_test_split(args.sample_train_percent)['test']

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=256, padding='max_length')
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        do_predict=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size,
        save_total_limit=1,
        load_best_model_at_end=True,
        num_train_epochs=50,
        metric_for_best_model='accuracy',
        overwrite_output_dir=True,
        seed=args.seed,
        report_to='wandb',
        deepspeed=args.deepspeed,
        logging_steps=100,
    )

    #def post_processing(x, y):
    #    return torch.argmax(x[0], dim=-1)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        #preprocess_logits_for_metrics=post_processing
    )

    if not args.test_model_only:
        trainer.train()
    
    outputs = trainer.predict(tokenized_data["test"])
    with open(output_dir + '/predict_results.json', 'w') as f:
        json.dump(outputs.metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-model-only', action='store_true')
    parser.add_argument('--mednli-dir', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--max-seq-length', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--deepspeed', type=str)
    parser.add_argument('--modify-tokenizer', action='store_true')
    parser.add_argument('--sample-train-percent', type=float, default=-1)
    args = parser.parse_args()
    print(f"Running with {args.seed}")

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path, num_labels=3)
    if 'gpt-j' in args.model_path:
        from transformers import GPTJForSequenceClassification
        model = GPTJForSequenceClassification.from_pretrained(
           "EleutherAI/gpt-j-6B", config=config #, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
        ) #.half()

    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path, config=config)

    # Get data and use the tokenizer on the data 
    dataset_dict = get_data(args.mednli_dir)
    tokenized_datasets = get_tokenized_data(dataset_dict, tokenizer, args.max_seq_length)
    
    # For PubmedGPT
    if args.modify_tokenizer:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Train the model
    train_model(model, tokenizer, args.output_dir, tokenized_datasets, args)

