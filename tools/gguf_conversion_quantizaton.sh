# Clone the llama.cpp repository with shallow copy (depth=1)
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git

# Modify the Makefile for CUDA library paths
sed -i 's|MK_LDFLAGS   += -lcuda|MK_LDFLAGS   += -L/usr/local/nvidia/lib64 -lcuda|' Makefile

# Build the project with CUDA enabled
LLAMA_CUDA=1 make -j > /dev/null


# Quantify the GGUF model, reducing the 16 GB model to around 4-5 GB.
./llama.cpp/llama-quantize /kaggle/input/hf-llm-to-gguf/llama-3-8b-chat-doctor.gguf llama-3-8b-chat-doctor-Q4_K_M.gguf Q4_K_M