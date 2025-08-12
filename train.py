import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

# Model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for LoRA + int8 training
model = prepare_model_for_int8_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load dataset (replace with your own CSV files)
dataset = load_dataset("csv", data_files={"train": "data/train.csv", "validation": "data/valid.csv"})

def tokenize(batch):
    inputs = tokenizer(batch["input"], truncation=True, padding="max_length", max_length=512)
    outputs = tokenizer(batch["output"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    learning_rate=3e-4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    fp16=True,
    push_to_hub=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

trainer.train()
trainer.save_model("./lora_finetuned_model")
