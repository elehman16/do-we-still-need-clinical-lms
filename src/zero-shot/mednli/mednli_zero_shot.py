import json
import argparse
import pandas as pd
import numpy as np 
from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

def get_mapping(prompt: str) -> dict:
    mapper1 = {'yes': 'entailment', 'no': 'contradiction', 'maybe': 'neutral', 'it is not possible to tell': 'neutral'}
    mapper2 = {'true': 'entailment', 'false': 'contradiction', 'inconclusive': 'neutral'}
    mapper3 = {'always': 'entailment', 'never': 'contradiction', 'sometimes': 'neutral'}
    mapper4 = {'entailment': 'entailment', 'contradiction': 'contradiction', 'neutral': 'neutral'}
    mapper5 = {'true': 'entailment', 'false': 'contradiction', 'neither': 'neutral'}  
    mapper6 = {'yes': 'entailment', 'no': 'contradiction', "it's impossible to say": 'neutral', 'it is not possible to say': 'neutral'} 
    
    combiner = {}
    mappers = [mapper1, mapper2, mapper3, mapper4, mapper5, mapper6]
    for m in mappers:
        combiner = combiner | m

    print(combiner)
    return combiner


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

def compute_metrics_prompt(predictions, prompt: str):
    """Given some predictions, calculate the F1. """
    predictions.label_ids[predictions.label_ids == -100] = 0

    # Decode the predictions + labels 
    decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
    decoded_predictions = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    mapper = get_mapping(prompt)
    
    # It's generating too much! Take the first word. 
    print(set(decoded_predictions))
    predictions = [mapper.get(x.lower().strip(), '') for x in decoded_predictions]
    if len(set(decoded_predictions)) > 3:
        predictions = [mapper.get(x.split()[0].lower().strip(), "") for x in decoded_predictions]
        
    return {
            'f1': f1_score(predictions, decoded_labels, average='macro'),
            'accuracy': accuracy_score(predictions, decoded_labels)
           }

def format_single_example(sentence1: str, sentence2: str, prompt: int) -> str:
    if prompt == '0':
        return f"Answer entailment, neutral or contradiction.\n\nPremise: {sentence1}\nHypothesis: {sentence2}\nAnswer:"
    elif prompt == '1':
        return f'Suppose {sentence1} Can we infer that "{sentence2}"? Yes, no, or maybe?'
    elif prompt == '2':
        return f'{sentence1} Based on that information, is the claim: "{sentence2}" true, false, or inconclusive?'
    elif prompt == '3':
        return f'Given that "{sentence1}". Does it follow that "{sentence2}" Yes, no, or maybe?'
    elif prompt == '4':
        return f'Suppose it\'s true that {sentence1} Then, is "{sentence2}" always, sometimes, or never true?'
    elif prompt == '5':
        prompt = 'Answer entailment, contradiction or neutral.\nPremise: Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.\nHypothesis: Patient has elevated Cr\nAnswer: entailment\n\nPremise: Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.\nHypothesis: Patient has elevated BUN\nAnswer: neutral\n\nPremise: Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.\nHypothesis: Patient has normal Cr\nAnswer: contradiction\n\n'
        prompt += f'Premise: {sentence1}\nHypothesis: {sentence2}\nAnswer:'
        return prompt
    elif prompt == '6':
        prompt = 'Premise: Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.\nHypothesis: Patient has elevated Cr\nAnswer entailment, contradiction or neutral: entailment\n\nPremise: Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.\nHypothesis: Patient has elevated BUN\nAnswer entailment, contradiction or neutral: neutral\n\nPremise: Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.\nHypothesis: Patient has normal Cr\nAnswer entailment, contradiction or neutral: contradiction\n\n'
        prompt += f'Premise: {sentence1}\nHypothesis: {sentence2}\nAnswer entailment, contradiction or neutral:'
        return prompt
    elif prompt == '7':
        prompt = f"{sentence1} Question: {sentence2} True, False, or Neither?"
        return prompt
    elif prompt == '8':
        prompt = 'Given that "Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.". Does it follow that "Patient has elevated Cr" Yes, no, or maybe? Yes.\n\nGiven that "Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.". Does it follow that "Patient has elevated BUN" Yes, no, or maybe? Maybe.\n\nGiven that "Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.". Does it follow that "Patient has normal Cr" Yes, no, or maybe? No.\n\n' 
        prompt += f'Given that "{sentence1}". Does it follow that "{sentence2}" Yes, no, or maybe?'
        return prompt
    elif prompt == '9':
        prompt = 'Given that "It was not associated with any shortness of breath, nausea, vomiting, or she tried standing during this episode.” Does it follow that "She had vomiting and dyspnea with this episode”? Yes, no, or maybe? No.\n\nGiven that "He has been followed by Dr. [**Last Name (STitle) 21267**] of Podiatry for chronic first toe MTP ulcer, which is now resolving.” Does it follow that "He had a chronic wound on his toe”? Yes, no, or maybe? Yes.\n\nGiven that "She had no fevers/chills/sweats prior to coming to the hosptial.” Does it follow that "Patient has a normal abdominal CT”? Yes, no, or maybe? Neutral.'
        prompt += f'Given that {sentence1} Does it follow that {sentence2.strip()}? Yes, no, or maybe?'
        return prompt 
    elif prompt == '10':
        prompt = f"Does {sentence1} mean that {sentence2}?\n\n Options:\n-Yes\n-No\n-It's impossible to say"
        return prompt
    elif prompt == '11':
        if sentence1.strip()[-1] == '.':
            sentence1 = sentence1[:-1]

        prompt = f'Does "{sentence1}" mean that "{sentence2.strip()}"?\n\nOptions:\n-Yes\n-No\n-It\'s impossible to say'
        return prompt

    elif prompt == '12':
        if sentence1.strip()[-1] == '.':
            sentence1 = sentence1[:-1]

        prompt = 'Does “Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4” mean that "Patient has elevated Cr"?\n\nOptions:\n-Yes\n-No\n-It\'s impossible to say \nYes\n\nDoes “Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4” mean that "Patient has elevated BUN"?\n\nOptions:\n-Yes\n-No\n-It\'s impossible to say \nNo\n\nDoes “Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4” mean that "Patient has normal Cr"?\n\nOptions:\n-Yes\n-No\n-It\'s impossible to say \nIt’s impossible to say\n\n'
        actual_prompt = f'Does "{sentence1}" mean that "{sentence2.strip()}"?\n\nOptions:\n-Yes\n-No\n-It\'s impossible to say'
        return prompt + actual_prompt
    elif prompt == '13':
        if sentence1.strip()[-1] == '.':
            sentence1 = sentence1[:-1]

        prompt = 'Does “It was not associated with any shortness of breath, nausea, vomiting, or she tried standing during this episode” mean that "She had vomiting and dyspnea with this episode"?\n\n Options:\n-Yes\n-No\n-It\'s impossible to say \nNo\n\nDoes “He has been followed by Dr. [**Last Name (STitle) 21267**] of Podiatry for chronic first toe MTP ulcer, which is now resolving” mean that "He had a chronic wound on his toe"?\n\n Options:\n-Yes\n-No\n-It\'s impossible to say \nYes\n\nDoes “She had no fevers/chills/sweats prior to coming to the hosptial” mean that "Patient has a normal abdominal CT"?\n\n Options:\n-Yes\n-No\n-It\'s impossible to say \nIt’s impossible to say'
        actual_prompt = f'Does "{sentence1}" mean that "{sentence2.strip()}"?\n\nOptions:\n-Yes\n-No\n-It\'s impossible to say'
        return prompt + actual_prompt
    
    elif prompt == '14':
        prompt = f"Does Discharge Summary: {sentence1} mean that {sentence2}?\n\n Options:\n-Yes\n-No\n-It's impossible to say"
        return prompt

def preprocess_function(examples, tokenizer, max_seq_length: int, prompt: int):
    """Format the examples and then tokenize them. """
    inputs = [format_single_example(s1, s2, prompt) for s1, s2 in zip(examples['sentence1'], examples['sentence2'])]
    targets = examples['gold_label']

    import pdb; pdb.set_trace()
    model_inputs = tokenizer(inputs, text_target=targets) #, max_length=max_seq_length, truncation=True)
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

def get_tokenized_data(dataset_dict: DatasetDict, tokenizer, max_seq_length: int, prompt: str):
    """Tokenize stuff. """
    tokenized_dict = {}
    for name, dataset_ in dataset_dict.items():
        processed_item = preprocess_function(dataset_, tokenizer, max_seq_length, prompt)
        tokenized_dict[name] = Dataset.from_dict(processed_item)
        
    return DatasetDict(tokenized_dict)

def test_model(model,
                tokenizer,
                output_dir: str,                            
                tokenized_data, 
                seed: int,
                local_rank: int):

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
        compute_metrics=lambda x: compute_metrics_prompt(x, args.prompt), #compute_metrics,
    )

    
    if args.force_words:
        allowed_words = [tokenizer.encode(x) for x in get_mapping(args.prompt)]
        outputs = trainer.predict(tokenized_data["val"], force_words_ids=allowed_words, num_beams=3)
    else:
        tmp = tokenized_data["val"].shuffle()
        
        outputs = trainer.predict(Dataset.from_dict(tmp[:200]))
        #outputs = trainer.predict(tokenized_data['test'])

    print(outputs.metrics)
    with open(output_dir + '/predict_results.json', 'w') as f:
        outputs = {
                    'label_ids': outputs.label_ids.tolist(),
                    'metrics': outputs.metrics,
                    'predictions': outputs.predictions.tolist()
                  }

        json.dump(outputs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mednli-dir', type=str, required=True)
    parser.add_argument('--model-path', type=str, default="google/flan-t5-xxl")
    parser.add_argument('--max-seq-length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--force-words', action='store_true')
    args = parser.parse_args()
    print(f"Running with {args.seed}")

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
 
    # Get data and use the tokenizer on the data 
    dataset_dict = get_data(args.mednli_dir)
    tokenized_datasets = get_tokenized_data(dataset_dict, tokenizer, args.max_seq_length, args.prompt)

    # Train the model
    test_model(model, tokenizer, args.output_dir, tokenized_datasets, args.seed, args.local_rank)  
