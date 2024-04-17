from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import transformers


def debug_pile_loader(ds_path, tokenizer, max_length):
    pile = load_dataset("json", data_files=f"{ds_path}*.jsonl", streaming=True, split='train')
    pile = pile.filter(lambda x: len(x['text']) > 1)  # Make sure texts are longer than 1 character
    pile = pile.map(lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length), batched=True)
    
    # Print a few samples to see what's being generated
    print(">>> Testing data loader...")
    for i, sample in enumerate(pile.take(5)):
        print(f"Sample {i}: {sample}")
        

class SimpleIterableDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            yield item


def test_dataloader(ds_path, tokenizer, batch_size, max_length):
    from datasets import load_dataset

    pile = load_dataset("json", data_files=f"{ds_path}*.jsonl", streaming=True, split='train')
    pile = pile.filter(lambda x: len(x['text']) > 1)
    pile = pile.map(lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length), batched=True)

    iterable_dataset = SimpleIterableDataset(pile)
    dataloader = DataLoader(iterable_dataset, batch_size=batch_size)

    for i, batch in enumerate(dataloader):
        if i > 5:  # Limit to checking the first few batches to see what's being generated
            break
        print(f"Batch {i}: {batch.keys()}")  # Check structure of each batch


if __name__ == "__main__":
    # Call this with appropriate parameters to see output
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    # debug_pile_loader("/projectnb/aclab/datasets/pile/raw_data/train/", tokenizer, 128)
    test_dataloader("/projectnb/aclab/datasets/pile/raw_data/train/", tokenizer, 2, 128)
