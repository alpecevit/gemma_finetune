from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    BitsAndBytesConfig
)
from datasets import load_dataset
import os
import torch
import gc
import accelerate
from peft import LoraConfig
from trl import SFTTrainer
from functions.functions import generate_inputs

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

hf_token = os.environ['HF_TOKEN']

model_name = "google/gemma-2b-it"

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             quantization_config=bnb_config,
                                             device_map={"":0},
                                             token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          add_eos_token=True,
                                          token=hf_token)

# load in dataset for tuning
ds = load_dataset("iamtarun/code_instructions_120k_alpaca", split="train")

# these were some of the na values I was able to find in the dataset
# I will use this to clean the data, might not be all of them
na_list = [
    'Not applicable'
    ,'No input'
    ,'Not Applicable'
    ,'Not applicable.'
    ,'<no input>'
    ,'No input.'
    ,'not applicable'
    ,'No Input'
    ,'N/A'
    ,'"<noinput>"'
    ,'None'
    ,'Noinput'
    ,'noinput'
    ,'<No input>'
    ,'<No Input>'
    ,'No input required'
    ,'No input required.'
    ,'No input needed.'
    ,'<noinput>.'
    ,'< noinput >'
    ,'NoInput'
    ,'"No input"'
    ,'none'
    ,'<noinput'
    ,'`<noinput>`'
    ,'[No input]'
    ,'no input'
    ,'Not appliable'
    ,'None.'
    ,'Nothing'
    ,'Not Required'
    ,'No input necessary'
    ,'Not appliccable'
    ,'Not applicaple'
]

# generate the input text
text_col = [generate_inputs(row, na_list) for row in ds]
dataset = ds.add_column("text", text_col)

print(dataset['text'][1])

dataset = dataset.shuffle(seed=42)
dataset = dataset.map(lambda records: tokenizer(records["text"]),
                batched=True,
                remove_columns=['instruction','input', 'output', 'prompt'])

dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

del ds, dataset, na_list, text_col
torch.cuda.empty_cache()
gc.collect()

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    dataset_text_field="text",
    peft_config=lora_config,
    tokenizer=tokenizer,
    max_seq_length = 1024,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=0.03,
        max_steps=35,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    )
)
trainer.train()

save_model_name = "gemma-fintune-code"
trainer.model.save_pretrained(save_model_name)
tokenizer.save_pretrained(save_model_name)
