import json
import argparse
import pandas as pd
import numpy as np 
from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from evaluate import load
squad_metric = load("squad_v2")

def conversion(prediction_str: str) -> str:
    mapper = {'N/A': ''}    
    if prediction_str in mapper:
        return mapper[prediction_str]
    else:
        return prediction_str

def compute_metrics_prompt(predictions, answers, prompt: str):
    """Given some predictions, calculate the F1. """
    # Decode the predictions + labels 
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
 
    # It's generating too much! Take the first word. 
    print(set(decoded_predictions))
    predictions = [{'prediction_text': conversion(x), 'id': str(i), 'no_answer_probability': -1} for i, x in enumerate(decoded_predictions)]
    references = [{'id': str(i), 'answers': l} for i, l in enumerate(answers)]

    results = squad_metric.compute(predictions=predictions, references=references)
    return results
 

def format_single_example(e, prompt: str) -> str:
    if prompt == '0':
        return f"Context: {e['context']}\nQuestion:{e['question']}\nAnswer:"
    elif prompt == '1':
        return f"Context: {e['context']}\nGiven the above context, {e['question']}? Answer N/A if there is no answer."
    elif prompt == '2':
        return f"Answer N/A if there is no answer.\n\nContext: {e['context']}\nQuestion:{e['question']}\nAnswer:"
    elif prompt == '3':
        return f"Context: {e['context']}\nGiven the above context, {e['question']}? Answer N/A if there is no answer or give a quote from the context: "
    elif prompt == '4':
        return f"Context: {e['context']}\nGiven the above context, {e['question']}?\nAnswer: \""
    elif prompt == '5':
        return f"Context: {e['context']}\nGiven the above context, {e['question']}? Give a quote from the text: "
    elif prompt == '6':
        prompt = "Context: IMPRESSION:  Subdural hematomas with blood products of different ages.\n Question vescular abnormality in left suprasellar space.  Findings were\n discussed with Dr. [**Last Name (STitle) 8620**] at 9:25 am on [**2191-8-5**].  An MRI of the brain and MRA\n of the COW is recommended.\nGiven the above context, Is there any significant change in bleeding? Answer N/A if there is no answer or give a quote from the context: N/A\n\n"
        return prompt + f"Context: {e['context']}\nGiven the above context, {e['question']}? Answer N/A if there is no answer or give a quote from the context: "
    elif prompt == '7':
        prompt = "Context: FINAL REPORT\nCHEST SINGLE AP FILM:\n\nHISTORY: ICD placement and shortness of breath.\n\nThere is a right sided dual chamber ICD with leads unchanged in location in\n this single view compared with the prior film of [**2101-5-25**]. No pneumothorax.\n There is cardiomegaly with upper zone redistribution, bilateral pleural\n effusions and associated bibasilar atelectases. Consolidation at the lung\n bases cannot be ruled out.\nGiven the above context, Is there any significant change from prior visit? Answer N/A if there is no answer or give a quote from the context: ICD with leads unchanged in location\n\n"
        prompt += f"Context: {e['context']}\nGiven the above context, {e['question']}? Answer N/A if there is no answer or give a quote from the context: "
        return prompt 
    elif prompt == '8':
        prompt = "Context: IMPRESSION:  Subdural hematomas with blood products of different ages.\n Question vescular abnormality in left suprasellar space.  Findings were\n discussed with Dr. [**Last Name (STitle) 8620**] at 9:25 am on [**2191-8-5**].  An MRI of the brain and MRA\n of the COW is recommended.\nGiven the above context, Is there any significant change in bleeding? Answer N/A if there is no answer or give a quote from the context: N/A\n\nContext: FINAL REPORT\nCHEST SINGLE AP FILM:\n\nHISTORY: ICD placement and shortness of breath.\n\nThere is a right sided dual chamber ICD with leads unchanged in location in\n this single view compared with the prior film of [**2101-5-25**]. No pneumothorax.\n There is cardiomegaly with upper zone redistribution, bilateral pleural\n effusions and associated bibasilar atelectases. Consolidation at the lung\n bases cannot be ruled out.\nGiven the above context, Is there any significant change from prior visit? Answer N/A if there is no answer or give a quote from the context: ICD with leads unchanged in location\n\n"
        prompt += f"Context: {e['context']}\nGiven the above context, {e['question']}? Answer N/A if there is no answer or give a quote from the context: "
        return prompt
    elif prompt == '9':
        prompt = "Context: IMPRESSION:  Subdural hematomas with blood products of different ages.\n Question vescular abnormality in left suprasellar space.  Findings were\n discussed with Dr. [**Last Name (STitle) 8620**] at 9:25 am on [**2191-8-5**].  An MRI of the brain and MRA\n of the COW is recommended.\nGiven the above context, Is there any significant change in bleeding? Answer N/A if there is no answer or give a quote from the context: N/A\n\n"
        prompt += "Context: FINAL REPORT\nCHEST SINGLE AP FILM:\n\nHISTORY: ICD placement and shortness of breath.\n\nThere is a right sided dual chamber ICD with leads unchanged in location in\n this single view compared with the prior film of [**2101-5-25**]. No pneumothorax.\n There is cardiomegaly with upper zone redistribution, bilateral pleural\n effusions and associated bibasilar atelectases. Consolidation at the lung\n bases cannot be ruled out.\nGiven the above context, Is there any significant change from prior visit? Answer N/A if there is no answer or give a quote from the context: ICD with leads unchanged in location\n\n"
        prompt += f"Context: {e['context']}\nGiven the above context, {e['question']}? Answer N/A if there is no answer or give a quote from the context: "
        return prompt
    elif prompt == '10':
        prompt = f"Context: {e['context']}\nGiven the above context, {e['question']}?\nGive an answer from the text, but do not reply if there is no answer: "
        return prompt
    elif prompt == '11':
        if e['question'][-1] == '?':
            e['question'] = e['question'][:-1]


        return f"Radiology Report: {e['context']}\nGiven the above radiology report, {e['question']}? Answer N/A if there is no answer or give a quote from the context: "
    else:
        raise ValueError("Prompt id not implemented")


def preprocess_function(examples, tokenizer, max_seq_length: int, prompt: str):
    """Format the examples and then tokenize them. """
    inputs = [format_single_example(e, prompt) for e in examples]
    model_inputs = tokenizer(inputs, text_target=["" for i in range(len(examples))])
    return model_inputs

def format_csv(df: pd.DataFrame):
    df['answers'] = [eval(x) for x in df['answers']] #type: ignore
    return Dataset.from_pandas(df)

def get_data(radqa_path: str):
    """Get the mednli data. """
    train = format_csv(pd.read_csv(radqa_path + '/train.csv'))
    val = format_csv(pd.read_csv(radqa_path + '/dev.csv'))
    test = format_csv(pd.read_csv(radqa_path + '/test.csv'))
    return DatasetDict({"train": train, "val": val, "test": test})

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
    )
    
    if args.set == 'dev':
        ds = tokenized_data['val']
        ds.shuffle()
        ds = Dataset.from_dict(ds[:200])
        answers = ds['answers']
        outputs = trainer.predict(ds)

        output_metrics = compute_metrics_prompt(outputs.predictions, answers, prompt)
    else:
        answers = tokenized_data['test']['answers']
        outputs = trainer.predict(tokenized_data['test'])
        output_metrics = compute_metrics_prompt(outputs.predictions, answers, prompt)
        

    print(output_metrics)
    #with open(output_dir + '/predict_results.json', 'w') as f:
    #    outputs = {
    #                'label_ids': outputs.label_ids.tolist(),
    #                'metrics': outputs.metrics,
    #                'predictions': outputs.predictions.tolist()
    #              }
    #
    #    json.dump(outputs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--radqa-dir', type=str, required=True)
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
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
 
    # Get data and use the tokenizer on the data 
    dataset_dict = get_data(args.radqa_dir)
    tokenized_datasets = get_tokenized_data(dataset_dict, tokenizer, args.max_seq_length, args.prompt)

    # Train the model
    test_model(model, tokenizer, args.output_dir, tokenized_datasets, args.seed, args.local_rank, args.prompt)  
