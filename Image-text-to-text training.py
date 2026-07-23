
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, TaskType, PeftModel          
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


login(token=os.environ.get(""))


ds = load_dataset("ranwakhaled/EgyM3AV")
dataset = ds["train"]
print(dataset)
print(dataset.features)
print(dataset[0])

MODEL_ID = "google/gemma-3n-E4B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)
base_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, "RanaGaber/Gemma_Egy")

model = model.merge_and_unload()


GEMMA3_IMAGE_TOKEN = "<start_of_image>"  

PROMPT_TEMPLATE = (
    "Explain this slide in **Egyptian Arabic** using **clear** and **simple** sentences.\n"
    "Cover all visible elements in the slides including text, formulas, images and diagrams.\n"
    "Do NOT add new information, examples, definitions, or assumptions "
    "that are not explicitly shown on the slide."
)

class SingleCaptionDataset(Dataset):
    def __init__(self, hf_dataset, processor, max_length=256):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_length = max_length
        self.pad_id = processor.tokenizer.pad_token_id or 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        image = example["image"].convert("RGB").resize((448, 448))
        caption = example["target"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": PROMPT_TEMPLATE},
                ],
            }
        ]

        prompt_str = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,   
            tokenize=False,
        )

        batch = processor(
            images=[image],
            text=[prompt_str],
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in batch.items()}

        label_enc = processor.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        prompt_ids = inputs["input_ids"]          
        cap_ids    = label_enc["input_ids"].squeeze(0)  

        labels = torch.full((prompt_ids.shape[0] + cap_ids.shape[0],), -100, dtype=torch.long)
        labels[prompt_ids.shape[0]:] = cap_ids

        inputs["input_ids"] = torch.cat([prompt_ids, cap_ids], dim=0)
        inputs["attention_mask"] = torch.cat([
            inputs["attention_mask"],
            label_enc["attention_mask"].squeeze(0),
        ], dim=0)
        inputs["labels"] = labels

        return inputs


train_size = int(0.9 * len(dataset))
val_size   = len(dataset) - train_size
print(f"Train: {train_size}  |  Val: {val_size}")

train_data, val_data = random_split(dataset, [train_size, val_size])
train_dataset = SingleCaptionDataset(train_data, processor)
val_dataset   = SingleCaptionDataset(val_data,   processor)

def data_collator(features):
    batch = {}
    pad_id = processor.tokenizer.pad_token_id or 0

    if "pixel_values" in features[0]:
        batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])

    if "image_grid_thw" in features[0]:
        max_len = max(f["image_grid_thw"].shape[0] for f in features)
        padded = []
        for f in features:
            t = f["image_grid_thw"]
            pad_rows = max_len - t.shape[0]
            if pad_rows > 0:
                t = torch.cat([t, torch.zeros(pad_rows, t.shape[1], dtype=t.dtype)], dim=0)
            padded.append(t)
        batch["image_grid_thw"] = torch.stack(padded)

    for key, pad_val in [("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)]:
        if key in features[0]:
            batch[key] = torch.nn.utils.rnn.pad_sequence(
                [f[key] for f in features],
                batch_first=True,
                padding_value=pad_val,
            )

    return batch



training_args = TrainingArguments(
    output_dir="./ain-captioning-lora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,    
    num_train_epochs=5,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=25,
    eval_steps=100,
    save_steps=200,
    save_strategy="steps",
    eval_strategy="steps",
    report_to="none",
    bf16=True,
    dataloader_pin_memory=True,
    save_total_limit=2,
    remove_unused_columns=False,
    resume_from_checkpoint="last",     
    gradient_checkpointing=True,
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./full_model")
processor.save_pretrained("./full_model")

