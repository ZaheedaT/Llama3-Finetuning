from huggingface_hub import login
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import torch
from trl import setup_chat_format

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, LlamaConfig
from peft import PeftModel
import torch
from trl import setup_chat_format

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

if hf_token:
    print("------------Token loaded successfully----------------")
else:
    print("--------------Failed to load token. Please check your .env file-----------------")
login(token = hf_token)



base_model_path = "./llama-3-1b-chat-doctor/checkpoint-5/"
new_model = "llama-3-1b-chat-doctor/"
checkpoint = torch.load('./llama-3-1b-chat-doctor/checkpoint-5/rng_state.pth', weights_only=False)


#Set up the chat format using the trl library.
# Load and merge the adapter to the base model using the PEFT library.


# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
print(" Tokenizer length before", len(tokenizer))
# Step 2: Adjust the model configuration
config = LlamaConfig.from_pretrained(base_model_path)  # Use your model's config class


config.hidden_size=2048
config.vocab_size = 128256 # Ensure vocab size matches the tokenizer's vocab size

print("This is the config from the Checkpoint model", config)
base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model_path,
ignore_mismatched_sizes=True,
   # config = config,
        #return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
)


# Resize the token embeddings to match the tokenizer's vocabulary size
#base_model_reload.resize_token_embeddings(128256)

base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)
#base_model_reload = base_model_reload.load_state_dict(checkpoint['model_state_dict'], strict=False)
print(" Tokenizer length after", len(tokenizer))

# Merge adapter with base model
model = PeftModel.from_pretrained(new_model,model=base_model_reload)

model = model.merge_and_unload()
print("------------------Model Merged!------------------------")































