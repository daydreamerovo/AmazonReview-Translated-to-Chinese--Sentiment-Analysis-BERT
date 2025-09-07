import torch
import os

# file path
RAW_DATA_PATH = "data/amazon_reviews_Digital_Music_100000.csv"
TRANSLATED_DATA_PATH = "data/final_training_data_3class.csv"
MODEL_OUTPUT_DIR = "models/"
FINAL_TRAIN_DATA = "data/TRAINING_DATASET_FINAL_DATAAUG.csv"
FINAL_DATA_RAW = "data/final_training_data_3class_cleaned.csv" 
#MODEL_OUTPUT_DIR_ADJ = "models/"

# API select
TRANSLATOR_SERVICE = "qwen"
QWEN_MODEL_NAME = "qwen-turbo"
# API keys
# load api key from environment variable for security
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
CONCURRENT_THREADS = 15 
# Translation settings
TEXT_COLUMN = "text" # column containing english text
TRANSLATED_COLUMN = "translated_text" # new column name for translated chinese text


# model paramters
PRE_TRAINED_MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'
NUM_CLASSES = 3 # class from 1 to 5 stars

# training parameters
# "cuda" if torch.cuda.is_available() else
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 8
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
MAX_LEN = 128 # max length of a comment