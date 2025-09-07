# src/translate.py (Final version with AttributeError fix and configurable concurrency)

import pandas as pd
import dashscope
from tqdm import tqdm
from src import config
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def setup_api_key():
    """Sets up the API key for the dashscope SDK."""
    if not config.QWEN_API_KEY:
        raise ValueError("QWEN_API_KEY environment variable not set.")
    dashscope.api_key = config.QWEN_API_KEY

def translate_text(text: str) -> str:
    """Translates a single piece of text using the Qwen API."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
        
    prompt = f"Please translate this English review into Chinese: '{text}'"
    
    try:
        response = dashscope.Generation.call(
            model=config.QWEN_MODEL_NAME,
            prompt=prompt,
            result_format='text'
        )
        if response.status_code == 200:
            return response.output
        else:
            print(f"API Error: {response.code} - {response.message}")
            return text
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return text

def main():
    """Main function with resume and intermittent save capabilities."""
    setup_api_key()
    
    all_original_df = pd.read_csv(config.RAW_DATA_PATH)
    output_df = pd.DataFrame()

    if os.path.exists(config.TRANSLATED_DATA_PATH):
        print(f"Found existing translated file. Loading to resume...")
        # Make sure to use index_col=0 to correctly load the index
        output_df = pd.read_csv(config.TRANSLATED_DATA_PATH, index_col=0)
        
    translated_indices = set(output_df.index)
    df_to_translate = all_original_df[~all_original_df.index.isin(translated_indices)].copy()
    
    if df_to_translate.empty:
        print("All reviews have already been translated. Exiting.")
        return

    print(f"Total reviews to translate: {len(df_to_translate)}")
    texts_to_translate = df_to_translate[config.TEXT_COLUMN].tolist()
    original_indices = df_to_translate.index.tolist()

    # Use concurrency setting from config file
    print(f"Starting translation with {config.CONCURRENT_THREADS} concurrent threads...")
    results = {}

    with ThreadPoolExecutor(max_workers=config.CONCURRENT_THREADS) as executor:
        future_to_index = {
            executor.submit(translate_text, text): index 
            for index, text in zip(original_indices, texts_to_translate)
        }
        
        for i, future in enumerate(tqdm(as_completed(future_to_index), total=len(texts_to_translate), desc="Translating")):
            original_index = future_to_index[future]
            try:
                results[original_index] = future.result()
            except Exception as e:
                print(f"A task generated an exception: {e}")
                results[original_index] = texts_to_translate[original_indices.index(original_index)]

            if (i + 1) % 1000 == 0:
                print(f"\nSaving progress... Processed {i+1} items in this run.")
                
                # BUG FIX: Create DataFrame from Series, which is more robust.
                temp_series = pd.Series(results, name=config.TRANSLATED_COLUMN)
                temp_df = temp_series.to_frame()

                output_df = pd.concat([output_df, temp_df])
                output_df.to_csv(config.TRANSLATED_DATA_PATH, index=True, encoding='utf-8-sig')
                results.clear()

    if results:
        print("Saving final batch...")
        temp_series = pd.Series(results, name=config.TRANSLATED_COLUMN)
        temp_df = temp_series.to_frame()
        output_df = pd.concat([output_df, temp_df])

    print("Finalizing and saving the complete file...")
    # Join with original data to ensure all columns are present and in order
    final_df = all_original_df.join(output_df[[config.TRANSLATED_COLUMN]])
    # Fill any remaining untranslated rows with the original English text
    final_df[config.TRANSLATED_COLUMN] = final_df[config.TRANSLATED_COLUMN].fillna(final_df[config.TEXT_COLUMN])
    # Select final columns and save without the index
    final_df[['rating', 'title', 'text', config.TRANSLATED_COLUMN]].to_csv(config.TRANSLATED_DATA_PATH, index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print("Process completed successfully!")
    print("="*50)

if __name__ == '__main__':
    main()