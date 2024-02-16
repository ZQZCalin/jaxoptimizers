from datasets import load_dataset, disable_caching
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange
import random
import functools


def shift_labels(batch):
    batch["labels"] = F.pad(batch["labels"][:, 1:], (0, 1), value=-100)
    return batch

def split_sequences(batch, max_length):
    # assumes that batch has shape [B, L] where L is a multiple of max_length
    batch["size"] = batch["input_ids"].size()
    batch["input_ids"] = rearrange(batch["input_ids"], 'b (l c) -> (b l) c')
    batch["labels"] = rearrange(batch["labels"], 'b (l c) -> (b l) c')
    return batch

def postprocess_collate_fn(collate_fn, post_fn):
    old_torch_call = collate_fn.torch_call
    def new_torch_call(self, *args, **kwargs):
        batch = old_torch_call(self, *args, **kwargs)
        return post_fn(batch)
    collate_fn.torch_call = new_torch_call
    return collate_fn


def get_c4_loader_next_token(tokenizer, split, batch_size, max_length=None, shuffle_buffer_size=0,
                             pad_to_multiple_of=None, mlm=False, mlm_probability=0, random_start=False, 
                             ds_path="/projectnb/aclab/datasets/c4/en/", num_workers=2, output_format='torch', **collator_args):
    '''
    Produces a pytorch dataloader object to train C4 on a "next token prediction" task
    in which each example is some length L tokenized text, and the model must
    predict the nth token using the first n-1 tokens for each n from 1 to L+1.

    For a batch size of B, the each entry in the dataloader will have columns:
        'size': a list of B integers describing the number of tokens in each example.
        'input_ids': a BxL matrix of token ids where L is the maximum sequence length.
        'labels': a BxL matrix of target token ids. Each row of 'labels' is usually
            simply a shifted version of 'input_ids', so that actually only the
            last entry in the row is not computable from 'input_ids'. Despite this
            redundancy, it is simpler in downstream code to have the labels available
            in this format.

    Arguments:
        tokenizer: the tokenizer to use (should be Huggingface tokenizer).
        split: 'train' or 'test'.
        batch_size: integer, the batch size.
        max_length: restrict sequences to this max length (if None, then no restriction).
        shuffle_buffer_size: if >0, will "shuffle" the data using a lookahead buffer of this size.
            Larger values require more memory but provide a better shuffle.
        pad_to_multiple_of: pad sequences to a multiple of this value, if provided.
        mlm: if true, blank out some tokens at random in the input.
        mlm_probability: probability to blank out any given token.
        random_start: start each sequence in a random point in the original C4 sequence.
        num_worker: number of CPU threads to use for this.
        **collator args: extra arguments to pass to DataCollatorForLanguageModeling


    Returns:
        pytorch DataLoader for the C4 dataset.
    '''
    collate_fn = DataCollatorForLanguageModeling(tokenizer,
                                                 mlm=mlm,
                                                 mlm_probability=mlm_probability,
                                                 pad_to_multiple_of=pad_to_multiple_of,
                                                 **collator_args)

    collate_fn = postprocess_collate_fn(collate_fn, shift_labels)
    return get_c4_loader_from_collate_fn(tokenizer=tokenizer,
                                         split=split,
                                         batch_size=batch_size,
                                         shuffle_buffer_size=shuffle_buffer_size,
                                         max_length=max_length,
                                         random_start=random_start,
                                         collate_fn=collate_fn,
                                         num_workers=num_workers,
                                         output_format=output_format)
                                    


def get_c4_loader_from_collate_fn(
    tokenizer, 
    split, 
    batch_size, 
    max_length, 
    shuffle_buffer_size, 
    random_start, 
    collate_fn, 
    ds_path="/projectnb/aclab/datasets/c4/en/", 
    num_workers=2,
    output_format='torch'):
    disable_caching()
    c4 = load_dataset('c4', 'en', data_dir=ds_path, streaming=True, split=split)
    c4 = c4.filter(lambda x: len(x['text']) > 1)
    if shuffle_buffer_size > 0:
        c4 = c4.shuffle(buffer_size=shuffle_buffer_size)
    if random_start:
        c4 = c4.map(lambda examples: {"text": examples["text"][random.randint(0,len(examples["text"])):]})
    c4 = c4.map(lambda examples: tokenizer(examples["text"], 
                                            padding=True,
                                            truncation=True,
                                            max_length=max_length),
                                remove_columns=["text", "timestamp", "url"],
                                batched=True,
                                batch_size=batch_size)
    c4 = c4.with_format(output_format)


    dataloader = DataLoader(c4,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            num_workers=num_workers)
    return dataloader
