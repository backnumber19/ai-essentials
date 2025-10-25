import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

X_class, y_class = make_classification(
    n_samples=1000, n_features=100, n_classes=3, n_informative=50, random_state=42
)

_, y_reg = make_regression(n_samples=1000, n_features=100, noise=0.1, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_class)
X_train, X_test, y_class_train, y_class_test = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42
)
_, _, y_reg_train, y_reg_test = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_class_train_tensor = torch.LongTensor(y_class_train)
y_class_test_tensor = torch.LongTensor(y_class_test)
y_reg_train_tensor = torch.FloatTensor(y_reg_train).unsqueeze(1)
y_reg_test_tensor = torch.FloatTensor(y_reg_test).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_class_train_tensor, y_reg_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# === 1. Ensemble Learning ===
# Train multiple models independently
model1 = RandomForestClassifier().fit(X_train, y_train)
model2 = SVC(probability=True).fit(X_train, y_train)

# Combine predictions
pred1 = model1.predict_proba(X_test)
pred2 = model2.predict_proba(X_test)
ensemble_pred = (pred1 + pred2) / 2


# === 2. Joint Training ===
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(100, 64)  # Shared feature extractor
        self.task1_head = nn.Linear(64, 3)  # Classification task
        self.task2_head = nn.Linear(64, 1)  # Regression task

    def forward(self, x):
        shared_features = torch.relu(self.shared(x))
        return self.task1_head(shared_features), self.task2_head(shared_features)


# Single model learns multiple tasks simultaneously
model = MultiTaskModel()
optimizer = torch.optim.Adam(model.parameters())

for x, y1, y2 in dataloader:
    class_pred, reg_pred = model(x)
    # Combined loss from both tasks
    combined_loss = nn.CrossEntropyLoss()(class_pred, y1) + nn.MSELoss()(reg_pred, y2)
    combined_loss.backward()
    optimizer.step()
