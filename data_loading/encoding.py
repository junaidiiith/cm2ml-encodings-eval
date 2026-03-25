from typing import List, Union
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer

def get_max_length(tokenizer):
    tokenizer_name = tokenizer.name_or_path.lower()
    if 'modernbert' in tokenizer_name:
        return 8000
    return 512

# Create your dataset
class EncodingDataset(Dataset):
    def __init__(
            self, 
            tokenizer: AutoTokenizer, 
            texts: List[str], 
            labels:List[Union[str, int]]=None, 
            max_length=512,
            remove_duplicates=False
        ):
        
        max_length = get_max_length(tokenizer)

        if remove_duplicates:
            print(f'Dataset with {len(texts)} samples before removing duplicates')
            texts_to_id = {text: i for i, text in enumerate(texts)}
            texts = list(texts_to_id.keys())
            labels = [labels[i] for i in texts_to_id.values()] if labels else None            
        
        # print(f'Encoding started with {len(texts)} samples')

        self.inputs = tokenizer(
            texts, 
            return_tensors='pt', 
            truncation=True, 
            padding='max_length', 
            max_length=max_length
        )

        if labels is not None:
            self.inputs['labels'] = torch.tensor(labels, dtype=torch.long) if labels is not None else None

        # print("Embedding shape: ", self.inputs['input_ids'].shape)
        # print("Encoding Dataset created with {} samples".format(len(self.inputs['input_ids'])))
        # print("\n".join([f"Label: {l}, Text: {i}" for i, l in zip(texts, labels)]))
        # import code; code.interact(local=locals())
        
 
    def __len__(self):
        return len(self.inputs['input_ids'])


    def __getitem__(self, index):
        return {k: v[index] for k, v in self.inputs.items()}


class GPTTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all the texts upon initialization
        self.encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,  # Pads to the longest sequence in the batch
            max_length=max_length,
            return_tensors="pt"  # Return PyTorch tensors
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = self.encodings["input_ids"][idx]
        attention_mask = self.encodings["attention_mask"][idx]
        
        # Labels for language modeling are the same as input_ids
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }