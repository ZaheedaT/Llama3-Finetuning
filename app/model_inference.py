# To verify if our model has been merged correctly, we'll perform a simple inference using pipeline from the transformers library.


messages = [{"role": "user", "content": "Hello doctor, I have bad acne. How do I get rid of it?"}]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto", # 'auto'- to use multiple GPUs
)

# Perform inference
outputs = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])

model.save_pretrained("llama-3-1b-chat-doctor")
tokenizer.save_pretrained("llama-3-1b-chat-doctor")

#We can push all the files to the Hugging Face Hub using the push_to_hub() function.
model.push_to_hub("llama-3-1b-chat-doctor", use_temp_dir=False)
tokenizer.push_to_hub("llama-3-1b-chat-doctor", use_temp_dir=False)

# Save and push the merged model
model.save_pretrained("llama-3-1b-chat-doctor")
tokenizer.save_pretrained("llama-3-1b-chat-doctor")

model.push_to_hub("llama-3-1b-chat-doctor", use_temp_dir=False)
tokenizer.push_to_hub("llama-3-1b-chat-doctor", use_temp_dir=False)