import os

from datasets import load_from_disk
from transformers import AutoTokenizer

from src.textsummarizer.entity import DataTransformationConfig
from src.textsummarizer.logging import logger


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)
        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(example_batch['summary'], max_length=1024, truncation=True)
        return {
            'input_ids':input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encoding['input_ids']
        }
    
    def convert(self):
        dataset_samsun = load_from_disk(self.config.data_path)
        dataset_samsun_pt = dataset_samsun.map(self.convert_examples_to_features, batched=True)
        dataset_samsun_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))
