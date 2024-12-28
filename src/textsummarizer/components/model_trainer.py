import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from datasets import load_from_disk

from src.textsummarizer.entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_pegasus)
        
        # loading the data
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        config = self.config
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=config.num_train_epochs, warmup_steps=config.warmup_steps,
            per_device_train_batch_size=config.per_device_train_batch_size, per_device_eval_batch_size=1,
            weight_decay=config.weight_decay, logging_steps=config.logging_steps,
            eval_strategy=config.eval_strategy, eval_steps=config.eval_steps, save_steps=config.save_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps 
        )
        trainer = Trainer(model=model_pegasus, args=trainer_args,
                          tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                          train_dataset=dataset_samsum_pt["train"],
                          eval_dataset=dataset_samsum_pt["validation"])
        # 4GB GPU not enough to train this
        # trainer.train()
        
        # save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        # save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
        