# Fine-Tuning Llama 3 and Using It Locally
## In this project Llama 3 is fine tuned on a dataset of patient-doctor conversations, to create a model for medical dialogue.
Frameworks/Libraries used: 
*  Torch
*  Hugging Face Hub
*  PEFT: LoRA
*  trl

We will:
1)  Fine-tune a Llama 3 model using a medical dataset.
2)  Integrate the adapter with the base model and upload the complete model to the Hugging Face Hub.
3)  Convert the model files to the Llama.cpp GGUF format.
4)  Quantize the GGUF model and upload it to the Hugging Face Hub.
5)  Use the fine-tuned model locally with the Jan application.
6)  After merging, converting, and quantizing the model, it will be ready for private local use via the Jan application.

The model can also be deployed and used as a WebApp

![AI Medical Chatbot](doc-Jan.png)
