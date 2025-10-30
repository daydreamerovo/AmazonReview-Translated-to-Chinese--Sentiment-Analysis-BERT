# AmazonReview-Chinese-Sentiment-Analysis

## Overview

This project firstly translates the 2023 Amazon Review of Digital_Music into Chinese using Qwen API, then performs a 3-class sentiment analysis based on the data (around 10,000 comments) in LLaMA Factory.

## Key Results
2 models are fine-tuned: Qwen3-0.6B-base and Qwen3-4B-instruct-2057. 
Qwen3-4B-Instruct-2507: with 5-shot prompting on 2,000 samples, the model achieves **78.25%** accuracy with an average latency of **1.466 s/sample**. The same model in **zero-shot** yields **77.25%** (**0.523 s/sample**), in **5-shot** yields **78.25%** (**1.47 s/sample**), and in **10-shot** yields **82.75%** (**14.517 s/sample**).

For smaller models with locally merged weights, the two **Qwen3-0.6B** zero-shot runs achieve **57.15%** (**0.075 s/sample**) and **54.35%** (**0.061 s/sample**), with the corresponding **5-shot** results **79.66%** (**0.26 s/sample**).

Overall, the 4B instruction model delivers the best result this round at **82.75%** with a reasonable few-shot size, but the time delay is far beyond industrial demand. Perform a LoRA fine-tune on 0.6B's 5-shot may be a satisfactory plan.

## Repository Structure
TO DO

## Getting Started
To reproduce the results in this repository, please follow the steps below.
### Prerequisites
This project requires modules listed in the `requirements.txt` file, or you can follow the steps below. Do pay attention that new versions of **Hugging Face**'s **datasets** module DOES NOT support running data's author's remote script to download the dataset. One way (in which I used) is downgrading **datasets** to a certain version, listed in `requirements.txt`, then using the script to download.

### Installation
1. **Data download**:
Go to HuggingFace's dataset part and find Amazon-Review 2023. Or you can click this link directly: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
Choose any review you like, just change other variable names as well. Using `data_retrive.py` to retrieve data:
```bash
python sripts/data_retrive.py
```

3. **Install PyTorch with CUDA**:
   
Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) and get the correct installation command for your specific CUDA version. For example:
```bash
pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

3. **Install LLama Factory**
   
Clone and install the LLaMA-Factory framework, further train log, evaluated predicitons, etc will be saved in this file accordingly.
```bash
git clone [https://github.com/hiyouga/LLaMA-Factory.git](https://github.com/hiyouga/LLaMA-Factory.git)
cd LLaMA-Factory
pip install -e ".[torch,bitsandbytes,webui,api]"
cd ..
```

4. Create and Activate Conda Environment (Optional but recommended)
   
Use a clean environment named `llama_env` (or any name you prefer) with Python 3.10.
```bash
conda create -n llama_env python=3.10 -y
conda activate llama_env
```

5. **Install Project Dependencies**
TO DO

6. **Set Up Environmental Variables:**:
Set up your API key (choose one you prefer) using
```bash
QWEN_API_KEY="your_api_key_here"

```
API Key setup for English users: [API guide here :)](https://www.immersivelimit.com/tutorials/adding-your-openai-api-key-to-system-environment-variables)

API key setup for Chinese users: [API guide here :)](https://blog.csdn.net/tenc1239/article/details/133040806#:~:text=%E6%9C%AC%E6%96%87%E4%BB%8B%E7%BB%8D%E4%BA%86%E5%A6%82%E4%BD%95%E9%80%9A%E8%BF%87%E8%AE%BE%E7%BD%AE%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E5%AD%98%E5%82%A8OpenAIAPI%E5%AF%86%E9%92%A5%EF%BC%8C%E4%BB%A5%E5%87%8F%E5%B0%91%E7%9B%B4%E6%8E%A5%E4%BB%A3%E7%A0%81%E4%B8%AD%E7%9A%84%E7%A1%AC%E7%BC%96%E7%A0%81%EF%BC%8C%E4%BB%8E%E8%80%8C%E6%8F%90%E9%AB%98%E5%AE%89%E5%85%A8%E6%80%A7%E3%80%82,%E6%AD%A5%E9%AA%A4%E5%8C%85%E6%8B%AC%E6%96%B0%E5%BB%BA%E7%B3%BB%E7%BB%9F%E5%8F%98%E9%87%8F%E5%B9%B6%E4%BD%BF%E7%94%A8os.getenv%E8%8E%B7%E5%8F%96%E5%85%B6%E5%80%BC%E3%80%82)

## Run pipeline

### Stage 1: Data Processing and Preparation
1. Translate Data: Run the concurrent translation script `src/translate.py` (This step requires an API key).
2. Augment Data: Use `src/data_aug.ipynb` to increase the number of samples for minority classes.
3. Clean Data: Use the `notebook/data_aug_clean.ipynb` notebook to clean the translated text.
4. Format for LLaMA-Factory: Use the `notebook/llama_fac.ipynb` notebook to convert the final CSV datasets into the required .jsonl format.

### Stage 2: Train Baseline Model

TO DO
