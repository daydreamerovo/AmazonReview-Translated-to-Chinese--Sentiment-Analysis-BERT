# src/augment_data.py

import pandas as pd
from tqdm import tqdm
import time
import os
import dashscope
from src import config
from concurrent.futures import ThreadPoolExecutor, as_completed

def setup_api_key():
    """Sets up the API key for the dashscope SDK."""
    if not config.QWEN_API_KEY:
        raise ValueError("QWEN_API_KEY environment variable not set.")
    dashscope.api_key = config.QWEN_API_KEY

def back_translate_qwen(text, lang='fr'):
    """
    Performs back-translation for a single piece of text using the Qwen API.
    Returns the new text, or the original text if any step fails.
    """
    if not isinstance(text, str) or not text.strip():
        return text # Return original text for invalid input
    try:
        # Chinese to Foreign Language
        prompt1 = f"Translate the following Chinese text to {lang}. Only return the translated text. Chinese text: '{text}'"
        response1 = dashscope.Generation.call(model=config.QWEN_MODEL_NAME, prompt=prompt1)
        if response1.status_code != 200:
            # print(f"API Error (Step 1): {response1.message}")
            return text
        translated_text = response1.output['text'].strip()

        # Foreign Language back to Chinese
        prompt2 = f"Translate the following {lang} text to Chinese. Only return the translated text. {lang.capitalize()} text: '{translated_text}'"
        response2 = dashscope.Generation.call(model=config.QWEN_MODEL_NAME, prompt=prompt2)
        if response2.status_code != 200:
            # print(f"API Error (Step 2): {response2.message}")
            return text
            
        back_translated_text = response2.output['text'].strip()
        
        # Return the new text only if it's different from the original
        return back_translated_text if back_translated_text != text else text
        
    except Exception:
        return text 

def main():
    """Main function to run the robust, concurrent data augmentation."""
    print("--- Starting Data Augmentation with Qwen API (Concurrent .py Script) ---")
    setup_api_key()

    # Define file paths
    source_file = 'data/final_training_data_3class.csv' # contain both English and Chinese
    output_file = 'data/final_training_data_augmented.csv' # contain both English and Chinese
    labels_to_augment = [0, 1]

    # Load the source data
    df_original = pd.read_csv(source_file)
    
    # Isolate the rows to be used as a base for augmentation
    df_to_augment = df_original[df_original['sentiment'].isin(labels_to_augment)].copy()
    print(f"Found {len(df_to_augment)} original reviews (Negative/Neutral) to augment.")

    texts_to_augment = df_to_augment['final_text'].tolist()
    sentiments_for_augmented = df_to_augment['sentiment'].tolist()
    newly_augmented_rows = []

    # Use ThreadPoolExecutor for concurrent API calls
    with ThreadPoolExecutor(max_workers=config.CONCURRENT_THREADS) as executor:
        # Submit all tasks
        future_to_text = {executor.submit(back_translate_qwen, text): text for text in texts_to_augment}
        
        # Process results as they complete, with a progress bar
        for future in tqdm(as_completed(future_to_text), total=len(texts_to_augment), desc="Augmenting Reviews"):
            new_text = future.result()
            # Find the original sentiment for this text (less efficient but works)
            original_text = future_to_text[future]
            original_sentiment = df_to_augment[df_to_augment['final_text'] == original_text]['sentiment'].iloc[0]

            # Add to our list if the text was successfully changed
            if new_text and new_text != original_text:
                newly_augmented_rows.append({
                    'sentiment': original_sentiment,
                    'final_text': new_text
                })

    print(f"Successfully generated {len(newly_augmented_rows)} new augmented reviews.")

    # Combine, Shuffle, and Save
    if newly_augmented_rows:
        df_augmented_new = pd.DataFrame(newly_augmented_rows)
        df_final = pd.concat([df_original, df_augmented_new], ignore_index=True)
        df_final = df_final.sample(frac=1).reset_index(drop=True)
        df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Final dataset size: {len(df_final)} rows")
        print(f"Final dataset saved to: {output_file}")
    else:
        print("No new data was generated. Saving the original file.")
        df_original.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\n--- Augmentation Complete! ---")
    print("\nFinal class distribution:")
    print(pd.read_csv(output_file)['sentiment'].value_counts())

if __name__ == '__main__':
    main()