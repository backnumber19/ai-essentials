import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from Model import LSTMModel, generate_nonstationary_data


def train(model, train_loader, test_loader, epochs=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    total_mse = 0
    total_mae = 0
    n_batches = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            mse = nn.MSELoss()(pred, batch_y).item()
            mae = nn.L1Loss()(pred, batch_y).item()
            total_mse += mse
            total_mae += mae
            n_batches += 1
    avg_mse = total_mse / n_batches
    avg_mae = total_mae / n_batches

    return avg_mse, avg_mae


def compare_with_without_revin():
    input_len = 96
    pred_len = 24
    n_features = 7
    batch_size = 32
    epochs = 1000

    train_data = generate_nonstationary_data(
        n_samples=1000,
        seq_len=input_len + pred_len,
        n_features=n_features,
        mean_range=(40, 60),
        std_range=(8, 12),
        seed=42,
    )
    test_data = generate_nonstationary_data(
        n_samples=200,
        seq_len=input_len + pred_len,
        n_features=n_features,
        mean_range=(80, 120),
        std_range=(15, 25),
        seed=123,
    )

    train_x, train_y = train_data[:, :input_len], train_data[:, pred_len:]
    test_x, test_y = test_data[:, :input_len], test_data[:, -pred_len:]

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)

    # Model without RevIN
    model_no_revin = LSTMModel(
        input_size=n_features, pred_len=pred_len, use_revin=False
    )
    mse_no_revin, mae_no_revin = train(
        model_no_revin, train_loader, test_loader, epochs=epochs
    )

    # Model with RevIN
    model_with_revin = LSTMModel(
        input_size=n_features, pred_len=pred_len, use_revin=True
    )
    mse_with_revin, mae_with_revin = train(
        model_with_revin, train_loader, test_loader, epochs=epochs
    )

    improvement_mse = (1 - mse_with_revin / mse_no_revin) * 100
    improvement_mae = (1 - mae_with_revin / mae_no_revin) * 100

    print("\n" + "=" * 64)
    print("Performance Comparison under Distribution Shift")
    print("=" * 64)
    print(f"{'Metric': <20} {'Without RevIN': <20} {'With RevIN': <20}")
    print("-" * 64)
    print(f"{'MSE': <20} {mse_no_revin: <20.4f} {mse_with_revin: <20.4f}")
    print(f"{'MAE': <20} {mae_no_revin: <20.4f} {mae_with_revin: <20.4f}")
    print(f"{'MSE Improvement': <20} {'-': <20} {improvement_mse: >19.1f}%")
    print(f"{'MAE Improvement': <20} {'-': <20} {improvement_mae: >19.1f}%")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    compare_with_without_revin()