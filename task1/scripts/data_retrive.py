# successfully downloaded after downgrade
from datasets import load_dataset
import pandas as pd

# parameter settings
dataset_name = 'McAuley-Lab/Amazon-Reviews-2023'
category = 'Digital_Music' 
# control sample size
num_samples = 100000 

# loading dataset
print(f"loading '{category}'")
config_name = f'raw_review_{category}'
dataset_stream = load_dataset(
    dataset_name, 
    config_name, 
    split='full',
    streaming=True, 
    trust_remote_code=True
)
print("datasets downloading./")

# iterate through datasets
samples = []
for i, example in enumerate(dataset_stream):
    # break after achieve goal size
    if i >= num_samples:
        break
    
    # print process 
    if (i + 1) % 10000 == 0:
        print(f"prcoessed {i + 1}/{num_samples} texts./")
        
    # extract needed info
    samples.append({
        'rating': example['rating'],
        'title': example['title'],
        'text': example['text']
    })

print(f"sampling done: {len(samples)} comments.")

# transfer to csv file and store
df = pd.DataFrame(samples)
output_path = f'amazon_reviews_{category}_{len(df)}.csv'
# encoding using utf-8-sig 
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("-" * 50)
print(df.head())
print("-" * 50)