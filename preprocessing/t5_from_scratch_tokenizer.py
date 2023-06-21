import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from preprocessing.t5_tokenizer_model import SentencePieceUnigramTokenizer
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration

def create_tokenizer(input_csv: str, vocab_size: int):
    """Create a tokenizer from data and an """
    tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

    # Train tokenizer
    tokenizer.train(
        files=input_csv,
        vocab_size=vocab_size,
        show_progress=True
    )

    return tokenizer

def update_tokenizer(old_tokenizer, new_tokenizer):
    """Add the tokenizers together.
        `old_tokenizer`: the old tokenizer to update.
        `new_tokenizer`: the new tokenizer to take vocab from.
    """
    missing = list(set(new_tokenizer.get_vocab()) - set(old_tokenizer.get_vocab()))
    num_added_tokens = old_tokenizer.add_tokens(missing)
    
    return old_tokenizer, new_tokenizer

def update_model(model, tokenizer):
    model.resize_token_embeddings(len(tokenizer)) 
    return model 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Vocab/Model to base our `new` model off of.")
    parser.add_argument("--input-csv", required=True, type=str, help="Where to load the input text from?")
    parser.add_argument("--out-dir", required=True, type=str, help="Where to save the tokenizer?")
    parser.add_argument("--vocab-size", default=32000, type=int, help="How many tokens to keep in the vocab?")
    parser.add_argument("--create-new-model", action='store_true', help="")
    args = parser.parse_args()

    # Load the old tokenizer + model.
    old_tokenizer = T5Tokenizer.from_pretrained(args.model)

    # Load the new tokenizer, config, and model 
    new_tokenizer = create_tokenizer(args.input_csv, args.vocab_size)

    if args.create_new_model:
        config = T5Config.from_pretrained(args.model, vocab_size=new_tokenizer.get_vocab_size())
        config.save_pretrained(args.out_dir)
        new_tokenizer.save(args.out_dir + '/tokenizer.json')
    else:
        old_model = T5ForConditionalGeneration.from_pretrained(args.model)
        updated_tokenizer = update_tokenizer(old_tokenizer, new_tokenizer)
        model = update_model(old_model, updated_tokenizer)
        model.save_pretrained(args.out_dir)
        updated_tokenizer.save_pretrained(args.out_dir)

