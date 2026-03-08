import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset import DeepARDataset
from model import DeepAR
from utils import Gausian_NLL

def sine_wave_table():
    date_col = torch.arange(0, 1000).tolist() * 2
    product_id_col = (["product A"] * 1000) + (["product B"] * 1000)
    noise = torch.randn(2000) * 0.1
    sales_col = (torch.sin(torch.tensor(date_col, dtype=torch.float32) * 0.1) + noise).tolist()

    df = pd.DataFrame({
        "date": date_col,
        "product_id": product_id_col,
        "target": sales_col
    })
    return df

def train_model():
    df = sine_wave_table()
    
    max_encoder_length = 30
    max_prediction_length = 10
    
    dataset = DeepARDataset(
        train_df=df,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_covariates=[],
        time_varying_covariates=["target"],
        group_ids=["product_id"]
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DeepAR(input_size=1, hidden_size=32, n_targets=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for past, future in dataloader:
            optimizer.zero_grad()
            
            combined = torch.cat([past, future], dim=1)
            inputs = combined[:, :-1, :]
            targets = combined[:, 1:, :]
            
            mu, sigma = model(inputs)
            
            loss = Gausian_NLL(mu, sigma, targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    train_model()
