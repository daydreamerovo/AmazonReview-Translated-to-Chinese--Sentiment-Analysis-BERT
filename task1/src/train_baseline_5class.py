import torch
import torch.nn as nn
from tqdm import tqdm
from src import config
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.dataset import ReviewDataset
from src.model import SentimentClassifier
from torch.optim import AdamW
import os
import pandas as pd

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

            average_loss = total_loss / len(data_loader)
            report = classification_report(preds_all, labels_all, target_names=['1 star', '2 star', '3 star', '4 star', '5 star'], labels=[0, 1, 2, 3, 4], zero_division=0)

        return average_loss, report
        

def run():
    # data prep
    df = pd.read_csv(config.TRANSLATED_DATA_PATH)
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)

    train_dataset = ReviewDataset(df_train['final_text'].to_numpy(), df_train['rating'].to_numpy())
    val_dataset =  ReviewDataset(df_val['final_text'].to_numpy(), df_val['rating'].to_numpy())

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    model = SentimentClassifier(n_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss().to(config.DEVICE)

    for epoch in range(config.EPOCHS):
        print(f'epoch  {epoch + 1}/{config.EPOCHS}')

        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, config.DEVICE)
        print(f'Training loss: {train_loss}')

        val_loss, report = evaluate(model, val_loader, loss_fn, config.DEVICE)
        print(f'Validation loss: {val_loss}')
        print(report)

    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{config.MODEL_OUTPUT_DIR}bert_sentiment_classifier.bin")
    print("\nTraining complete. Model saved to models/bert_sentiment_classifier.bin")




if __name__ == '__main__':
    run()