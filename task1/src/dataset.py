import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src import config

class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)   
        self.texts = texts
        self.labels = labels 
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        # use tokenizer to process text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True, # add [CLS] and [SEP] tokens 
            max_length = config.MAX_LEN, # max length from config
            padding = 'max_length', # pad short text to max length
            truncation = True, # truncate text>128 to max length
            return_attention_mask = True, # attention mask
            return_tensors = 'pt' # return tensors
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long) # 1-5 to 0-4
        }

