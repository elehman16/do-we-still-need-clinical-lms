import os
import argparse
from transformers import AutoModel, AutoTokenizer
from preprocessing.convert_deid_to_tag import DEID_TO_TAG, TYPE_TO_TAG

def main(model_path: str):
    """Load the model, and save the model. """
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Add these to the tokenizer + model as special tokens
    tags_to_add = list(set(DEID_TO_TAG.values()).union(set(TYPE_TO_TAG.values())))

    before_update = tokenizer.tokenize(' '.join(tags_to_add))
    #tags_to_add = {'additional_special_tokens': list(set(DEID_TO_TAG.values()))}
    tokens_added = tokenizer.add_tokens(tags_to_add)
    model.resize_token_embeddings(len(tokenizer))
    after_update = tokenizer.tokenize(' '.join(tags_to_add))

    assert len(before_update) != len(after_update)
    print(f"We added {len(after_update) - len(before_update)} tokens")
    return tokenizer, model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help="Where to load the initial model from")
    parser.add_argument('--output-model-dir', type=str, required=True, help="Where to store the new model.")
    args = parser.parse_args()

    if not(os.path.exists(f"{args.output_model_dir}/config.json")):
        tokenizer, model = main(args.model_path)
        tokenizer.save_pretrained(args.output_model_dir)
        model.save_pretrained(args.output_model_dir)
    else:
        print("** Skipping Model Vocab **")
