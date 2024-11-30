from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    #BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from transformers import DataCollatorForSeq2Seq

from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from accelerate import Accelerator

from trl import SFTTrainer, setup_chat_format
from dotenv import load_dotenv
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import login
# Use a pipeline as a high-level helper
from transformers import pipeline

from transformers.utils.logging import set_verbosity_debug
from huggingface_hub import hf_hub_download



os.environ["HF_HUB_ENABLE_PARALLEL_DOWNLOAD"] = "1"
load_dotenv()


from transformers.utils.logging import set_verbosity_debug
import transformers.utils.logging as logging

# Enable debug logging
#set_verbosity_debug()

# Log output will display download details
#logging.enable_default_handler()
#logging.enable_explicit_format()


# Define the model path
model_path = "/home/zahee/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/"
#model_path= "meta-llama/Llama-3.2-1B"

#pipe = pipeline("text-generation", model="/home/zahee/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B")

hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    print("Token loaded successfully.")
else:
    print("Failed to load token. Please check your .env file.")
login(token = hf_token)

wb_token = os.getenv("WEIGHTS_BIASES_TOKEN")

wandb.login(key=wb_token)
run = wandb.init(
    project='Fine-tune Llama 3 1B on Medical Dataset',
    job_type="training",
    anonymous="allow"
)


dataset_name = "ruslanmv/ai-medical-chatbot"
new_model = "llama-3-1b-chat-doctor_10sample"

torch_dtype = torch.float16
attn_implementation = "eager"

# QLoRA config (Reduce sie and memory requirements of the model only for GPU)
#bnb_config = BitsAndBytesConfig(
 #   load_in_4bit=True,
  #  bnb_4bit_quant_type="nf4",
   # bnb_4bit_compute_dtype=torch_dtype,
    #bnb_4bit_use_double_quant=True,
#)
# Allows for distributed training. Manages memory more efficiently
accelerator = Accelerator()

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    #quantization_config=bnb_config,
    #load_in_8bit=True,
    device_map={"": "cpu"},  # Load the model on CPU
    #offload_folder="./offload",
    attn_implementation=attn_implementation,
)

# Now load the pipeline with the model
#pipe = pipeline("text-generation", model=model)


# Enable disk offloading
#model.disk_offload()  # Offload model layers to disk for memory efficiency


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
# Define the data collator for preprocessing/tokenizing
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest",
    max_length=10,
    return_tensors="pt"
)
# Use accelerator for model and tokenizer
model, tokenizer = accelerator.prepare(model, tokenizer)

model, tokenizer = setup_chat_format(model, tokenizer)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)

#Importing the dataset
dataset0 = load_dataset(dataset_name, split="all")
dataset01 = dataset0.shuffle(seed=65).select(range(10)) # Only use 1000 samples for quick demo

def format_chat_template(row):
    row_json = [{"role": "user", "content": row["Patient"]},
               {"role": "assistant", "content": row["Doctor"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset01.map(
    format_chat_template,
    num_proc=2,
)

print(dataset['text'][3])

dataset = dataset.train_test_split(test_size=0.1)

training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="adamw_torch",
    num_train_epochs=0.05,
    eval_strategy="steps",
    eval_steps=1,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length=10,
    dataset_text_field="text",
    data_collator=data_collator,
    args=training_arguments,
    packing= False,
)

trainer.train()

wandb.finish()
model.config.use_cache = True


