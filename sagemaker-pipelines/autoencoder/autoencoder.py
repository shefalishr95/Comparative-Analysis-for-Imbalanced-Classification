import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import argparse
from sagemaker.experiments.run import Run
from sagemaker.session import Session

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def get_reconstruction_error(model, data_loader, device):
    """Calculate reconstruction error for each sample"""
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0].to(device)
            outputs = model(inputs)
            error = torch.mean((outputs - inputs) ** 2, dim=1)
            reconstruction_errors.extend(error.cpu().numpy())
    
    return np.array(reconstruction_errors)

def evaluate_model(model, X, y, run, device, batch_size=32):
    """Evaluate autoencoder and log metrics to SageMaker Experiments"""
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size)
    
    # Get reconstruction errors
    reconstruction_errors = get_reconstruction_error(model, loader, device)
    
    # Calculate threshold based on validation set distribution
    threshold = np.percentile(reconstruction_errors, 95)  # Adjust percentile as needed
    
    # Convert to predictions
    y_pred = (reconstruction_errors > threshold).astype(int)
    
    metrics = {
        'auc_roc': roc_auc_score(y, reconstruction_errors),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred)
    }
    
    # Log metrics to SageMaker Experiments
    for metric_name, metric_value in metrics.items():
        run.log_metric(metric_name, metric_value)
    
    return metrics, threshold

def train():
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    
    # Training specific arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    args, _ = parser.parse_known_args()
    
    # Initialize SageMaker experiment
    experiment_name = "autoencoder-training"
    run = Run.create(experiment_name=experiment_name)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess data
    X_train = pd.read_parquet(os.path.join(args.train, "X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(args.train, "y_train.parquet"))
    X_test = pd.read_parquet(os.path.join(args.test, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(args.test, "y_test.parquet"))
    X_val = pd.read_parquet(os.path.join(args.val, "X_val.parquet"))
    y_val = pd.read_parquet(os.path.join(args.val, "y_val.parquet"))
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Log hyperparameters
    run.log_parameters({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "input_dim": input_dim
    })
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch[0].to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Log training loss
        avg_loss = total_loss / len(train_loader)
        run.log_metric(f"epoch_{epoch}_loss", avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")
    
    # Evaluate on all sets
    train_metrics, threshold = evaluate_model(model, X_train_scaled, y_train, run, device, args.batch_size)
    val_metrics, _ = evaluate_model(model, X_val_scaled, y_val, run, device, args.batch_size)
    test_metrics, _ = evaluate_model(model, X_test_scaled, y_test, run, device, args.batch_size)
    
    # Log metrics with prefixes
    for prefix, metrics in [
        ('train_', train_metrics),
        ('val_', val_metrics),
        ('test_', test_metrics)
    ]:
        for metric_name, metric_value in metrics.items():
            run.log_metric(prefix + metric_name, metric_value)
    
    # Save the model, scaler, and threshold
    save_dict = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'threshold': threshold,
        'input_dim': input_dim
    }
    model_path = os.path.join(args.model_dir, 'model.pt')
    torch.save(save_dict, model_path)
    
    # End the experiment run
    run.close()

if __name__ == '__main__':
    train()