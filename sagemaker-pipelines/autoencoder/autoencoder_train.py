# Rough version of the script; debugging needed

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from torchvision.models import resnet18

class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = resnet18(pretrained=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, encoding_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 3072),
            nn.ReLU(),
            nn.Linear(3072, 4096),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding_dim', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    
    return parser.parse_known_args()

def load_data(data_dir):
    logger.info(f"Loading data from {data_dir}")
    input_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.csv')]
    if len(input_files) == 0:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    raw_data = [pd.read_csv(file, header=None) for file in input_files]
    data = pd.concat(raw_data)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values
    
    logger.debug(f"Loaded data shape: {X.shape}")
    return X, y

def train_model(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("Loading training data")
    X_train, y_train = load_data(args.train)
    
    logger.info("Loading validation data")
    X_val, y_val = load_data(args.validation)
    
    # scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_normal = X_train_scaled[y_train == 0]

    # convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_normal)
    X_val_tensor = torch.FloatTensor(X_val_scaled)

    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = Autoencoder(
        input_dim=input_dim,
        encoding_dim=args.encoding_dim,
        dropout_rate=args.dropout_rate
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # start training
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for data, _ in train_loader:
            data = data.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs, data)
                val_loss += loss.item() * data.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    with open(os.path.join(args.output_data_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    joblib.dump(scaler, os.path.join(args.model_dir, 'scaler.pkl'))
    
    return model, scaler, X_val_scaled, y_val

def evaluate_model(model, X_val_scaled, y_val, device):
    model.eval()
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    
    with torch.no_grad():
        predictions = model(X_val_tensor).cpu().numpy()
    
    mse = np.mean(np.power(X_val_scaled - predictions, 2), axis=1)
    
    # Use reconstruction error as anomaly score
    # Higher reconstruction error = more likely to be anomaly
    y_pred = mse
    roc_auc = roc_auc_score(y_val, y_pred)
    precision, recall, _ = precision_recall_curve(y_val, y_pred)
    pr_auc = auc(recall, precision)
    
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"PR AUC: {pr_auc:.4f}")
    
    # Return eval metrics
    return {
        'validation_auc': float(roc_auc), 
        'pr_auc': float(pr_auc)
    }

def save_model(model, model_dir):
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    model_info = {
        'class_name': 'Autoencoder',
        'input_dim': model.decoder[-1].out_features,  # Extract input_dim from last layer
        'encoding_dim': model.encoder[-2].out_features,  
        'dropout_rate': model.encoder[2].p  
    }
    
    with open(os.path.join(model_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f)
    
    logger.info(f"Model saved to {model_dir}")

if __name__ == '__main__':
    args, _ = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, scaler, X_val_scaled, y_val = train_model(args)
    metrics = evaluate_model(model, X_val_scaled, y_val, device)
    
    with open(os.path.join(args.output_data_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    save_model(model, args.model_dir)
    logger.info("Training completed successfully")