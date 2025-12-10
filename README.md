# LoRA-Fine-Tuning-LLMs

LoRA is currently one of the most popular and effective parameter-efficient fine-tuning (PEFT) techniques for large language models.

This project provides clean, step-by-step implementations of LoRA so you can truly understand how it works under the hood. Everything is presented in easy-to-follow Jupyter notebooks.

The examples are designed to run on modest hardware with a very small memory footprint — a single Tesla T4, consumer-grade RTX cards, Colab, or even some higher-end laptops — so you can experiment immediately.

| Notebook                                                                 | LLM Model                                                                      |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| lora-finetune-opt.ipynb  | Fine-tune Meta’s OPT-125M |
| lora-finetune-gpt2.ipynb     | Fine-tune OpenAI’s GPT-2 (124M)


## 1. Environment Setup

You can run these notebooks on any machine with a modern NVIDIA GPU.  
The instructions below were tested on an Ubuntu 24.04 LTS VM

### Step 1: Install NVIDIA CUDA Driver

```bash
# Basic build tools
sudo apt-get update
sudo apt-get install -y gcc make

# Download and install CUDA
wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda_13.1.0_590.44.01_linux.run
sudo sh cuda_13.1.0_590.44.01_linux.run

# Add CUDA to your PATH and library path
echo 'export PATH=/usr/local/cuda-13.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 2: Install Python Packages

```bash
# Upgrade pip
sudo apt-get install -y python3-pip
sudo -H pip3 install --upgrade pip

# Core libraries
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu131
pip3 install transformers datasets pandas matplotlib tqdm

# For running the notebooks
pip3 install jupyterlab   # or just jupyter if you prefer the classic interface
```

## 2. Run the Examples

```bash
# Clone the repository
git clone https://github.com/tsmatz/finetune_llm_with_lora
cd finetune_llm_with_lora
```

Start Jupyter (choose whichever you installed):

```bash
jupyter lab          # recommended, modern interface
# or
jupyter notebook     # classic notebook interface
```

Open your browser (the URL will be shown in the terminal), navigate to one of the notebooks, and run the cells step-by-step.

That’s it — you’re now fine-tuning large models with LoRA using only a few hundred MB of GPU memory!

Enjoy experimenting, and feel free to open issues or contribute improvements. Happy fine-tuning! 
