from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, default_data_collator, TrainerCallback, set_seed
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset, concatenate_datasets
import torch
import os
import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DA_DIR = '/teamspace/studios/this_studio/amiya_data'

model_id = "google/gemma-3-12b-pt"
tokenizer_id = "google/gemma-3-12b-pt" 
lr = 3e-5
TRAIN_BATCH = 8
VAL_BATCH = 4
GRAD_ACCUM = 8
NUM_EPOCHS = 1
WEIGHT_DECAY = 0.01
SAVE_STEPS = 500
EVAL_STEPS = 1000
LOGGING_STEPS = 100



PEAK_LR_PHASE1 = lr
WARMUP_RATIO_PHASE1 = 0.03
PHASE1_EPOCHS = 1



OUTPUT_DIR_PHASE1 = "phase1_output"
 


def load_all_csvs_from_dir(directory):
    csv_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")])
    datasets = [load_dataset("csv", data_files=f)['train'] for f in csv_files]
    return concatenate_datasets(datasets)


def filter_data(example):
    return example.get("Country") == "Egypt" and example.get("Text") not in (None, "")


def run_phase(model, train_dataset, output_dir, peak_lr, warmup_ratio, epochs,
              tokenizer=None, resume_from_checkpoint=None, callbacks=None, seq_len=256):
    
    os.makedirs(output_dir, exist_ok=True)

    def tokenize(batch):
        texts = [(text or "") + tokenizer.eos_token for text in batch["Text"]]
        tokens = tokenizer(
            texts,               
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
        input_ids = torch.tensor(tokens["input_ids"])
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        tokens["input_ids"] = input_ids
        tokens["labels"] = labels
        return tokens

    train_dataset = train_dataset.map(lambda x: tokenize(x), batched=True, batch_size=5000, num_proc=4, remove_columns=["Text"])

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=TRAIN_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=epochs,
        learning_rate=peak_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=50,
        save_strategy="steps",
        save_total_limit=1,
        save_steps=500,
        report_to="wandb",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model = trainer.model
    trainer.save_model('./full_model')
    tokenizer.save_pretrained('./full_model')
    return model



if __name__ == "__main__":
    dataset_phase1 = load_all_csvs_from_dir(DA_DIR)


    dataset_phase1 = dataset_phase1.filter(filter_data)

    print("Phase 1 dataset length after filtering:", len(dataset_phase1))
    
    model = AutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(tokenizer_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    for name, param in model.named_parameters():
        if any(k in name for k in ["vision_tower", "vision_model", "image", "visual"]):
           param.requires_grad = False

    run_phase(
        model=model,
        train_dataset=dataset_phase1,
        output_dir=OUTPUT_DIR_PHASE1,
        peak_lr=PEAK_LR_PHASE1,
        warmup_ratio=WARMUP_RATIO_PHASE1,
        epochs=PHASE1_EPOCHS,
        tokenizer=tokenizer,
    )

