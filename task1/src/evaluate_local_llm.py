# src/evaluate_local_llm.py (Final Stable Version with tqdm and standard print)

import pandas as pd
import requests
import json
from tqdm import tqdm

# --- Configuration ---
API_URL = "http://localhost:7860/v1/chat/completions"
TEST_DATA_PATH = "../LLaMA-Factory/data/sentiment_test.jsonl"
FEW_SHOT_SOURCE_PATH = "../LLaMA-Factory/data/sentiment_validation.jsonl"
NUM_FEW_SHOT = 3

# --- Robust JSONL file reader ---
def read_jsonl_robust(filepath):
    """Reads a .jsonl file line by line, skipping any empty or malformed lines."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping malformed line: {line.strip()}")
    return pd.DataFrame(data)

# --- Prepare Few-Shot Examples ---
def prepare_few_shot_examples(filepath, num_examples):
    """Prepares a list of few-shot example messages for the API payload."""
    examples = []
    df = read_jsonl_robust(filepath)
    if df.empty:
        print("Warning: Few-shot source file is empty or could not be read.")
        return []
    for label in df['output'].unique():
        sample = df[df['output'] == label].iloc[0]
        examples.append({"role": "user", "content": sample['input']})
        examples.append({"role": "assistant", "content": sample['output']})
    return examples[:num_examples*2]

# --- Robust keyword extraction function ---
def extract_sentiment_keyword(text):
    """Extracts the key sentiment word from the model's potentially verbose output."""
    if "偏正向" in text:
        return "偏正向"
    elif "偏负向" in text:
        return "偏负向"
    elif "中性" in text:
        return "中性"
    else:
        return "未知"

# --- Main Evaluation Logic ---
def run_evaluation():
    """Runs the main evaluation loop with real-time feedback."""
    print("--- Preparing Few-Shot Examples from Validation Set ---")
    few_shot_messages = prepare_few_shot_examples(FEW_SHOT_SOURCE_PATH, NUM_FEW_SHOT)
    
    print("--- Loading Test Data ---")
    df_test = read_jsonl_robust(TEST_DATA_PATH)
    
    if df_test.empty:
        print("Error: Test data could not be loaded. Exiting.")
        return
        
    correct_count = 0
    
    print(f"--- Starting {NUM_FEW_SHOT}-Shot Evaluation on {len(df_test)} samples ---")
    
    progress_bar = tqdm(df_test.iterrows(), total=len(df_test))
    for index, row in progress_bar:
        user_prompt = row['input']
        true_label = row['output'].strip()
        
        messages = few_shot_messages + [{"role": "user", "content": user_prompt}]
        payload = { "model": "default", "messages": messages, "temperature": 0.1 }
        
        try:
            response = requests.post(API_URL, json=payload, timeout=60)
            if response.status_code == 200:
                raw_prediction = response.json()['choices'][0]['message']['content'].strip()
                clean_prediction = extract_sentiment_keyword(raw_prediction)
                
                if clean_prediction == true_label:
                    correct_count += 1
                
                # --- KEY CHANGE: Using standard print() for guaranteed visibility ---
                if (progress_bar.n + 1) % 20 == 0:
                    current_accuracy = correct_count / (progress_bar.n + 1)
                    feedback = (
                        f"\n" + "="*50 +
                        f"\n已处理 {progress_bar.n + 1}/{len(df_test)} 条样本" +
                        f"\n  - 模型回答: {raw_prediction}" +
                        f"\n  - 提取关键词: {clean_prediction}" +
                        f"\n  - 正确答案: {true_label}" +
                        f"\n  - 本次是否正确: {'✅' if clean_prediction == true_label else '❌'}" +
                        f"\n  - 当前实时准确率: {current_accuracy:.2%}" +
                        f"\n" + "="*50
                    )
                    # Use standard print() with flush=True to force immediate output
                    print(feedback, flush=True)

        except Exception as e:
            # Also use standard print() for errors
            print(f"\nAn error occurred for sample {progress_bar.n + 1}: {e}", flush=True)

    # --- Final Summary ---
    final_accuracy = correct_count / len(df_test) if len(df_test) > 0 else 0
    print("\n--- Evaluation Complete! ---")
    print(f"Correct Predictions: {correct_count}")
    print(f"Total Samples: {len(df_test)}")
    print(f"Final Few-Shot Accuracy: {final_accuracy:.2%}")

if __name__ == "__main__":
    run_evaluation()
