# src/train.py (Final Corrected Version)

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
from src.loss import FocalLoss

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

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # --- BUG FIX: These lines are now OUTSIDE the for loop ---
    # This ensures they run only after all batches have been evaluated.
    avg_loss = total_loss / len(data_loader)
    report = classification_report(all_labels, all_preds, target_names=['Negative', 'Neutral', 'Positive'], labels=[0, 1, 2], zero_division=0)
    
    return avg_loss, report
      
def run():
    # Log file setup
    log_file_path = f"{config.MODEL_OUTPUT_DIR}training_log_baseline.txt"
    with open(log_file_path, "w") as f:
        f.write(f"--- Starting Training Log for {config.PRE_TRAINED_MODEL_NAME} ---\n\n")

    # Data preparation with stratification
    df = pd.read_csv(config.FINAL_DATA_RAW)
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42, stratify=df['sentiment'])
    class_counts = df_train['sentiment'].value_counts().sort_index()

    train_dataset = ReviewDataset(df_train['final_text'].to_numpy(), df_train['sentiment'].to_numpy())
    val_dataset = ReviewDataset(df_val['final_text'].to_numpy(), df_val['sentiment'].to_numpy())

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    # Model, optimizer, and loss function setup
    model = SentimentClassifier(n_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = FocalLoss().to(config.DEVICE)
    
    best_val_loss = float('inf') 
    epochs_no_improve = 0        
    patience = 2        


    # Training & Evaluation loop
    for epoch in range(config.EPOCHS):

        print(f'--- Epoch {epoch + 1}/{config.EPOCHS} ---')


        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, config.DEVICE)
        val_loss, report = eval_model(model, val_loader, loss_fn, config.DEVICE)

        # --- Logging logic INSIDE the loop ---
        log_output = (
            f"--- Epoch {epoch + 1}/{config.EPOCHS} ---\n"
            f"Training loss: {train_loss}\n"
            f"Validation loss: {val_loss}\n"
            f"--- Validation Classification Report ---\n"
            f"{report}\n"
            "--------------------------------------\n\n"
        )
        print(log_output)
        with open(log_file_path, "a") as f:
            f.write(log_output)

        # EARLY STOPPING
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving best model...")
            best_val_loss = val_loss
            os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
            torch.save(model.state_dict(), f"{config.MODEL_OUTPUT_DIR}best_model_baseline.bin")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Patience: {epochs_no_improve}/{patience}")
         
        if epochs_no_improve >= patience:
            print("Early stopping triggered. Training finished.")
            break

    # This ensures the model is saved only after all epochs are complete.

    print(f"\nTraining complete. Model saved to {config.MODEL_OUTPUT_DIR}")
    print(f"Training log saved to {log_file_path}")

if __name__ == '__main__':

    run()