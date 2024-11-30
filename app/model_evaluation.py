from main import tokenizer, model,trainer, new_model

messages = [
    {
        "role": "user",
        "content": "Hello doctor, I have bad acne. How do I get rid of it?"
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors='pt', padding=True,
                   truncation=True)#.to("cuda")

outputs = model.generate(**inputs, max_length=150,
                         num_return_sequences=1, batch_size=2)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(text.split("assistant")[1])

trainer.model.save_pretrained(new_model)
trainer.model.push_to_hub(new_model, use_temp_dir=False)

print("---------------------MODEL SAVED TO: -----------------------", f"Model saved to PATH: {new_model}")