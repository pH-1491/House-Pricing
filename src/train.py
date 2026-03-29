import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import load_and_preprocess
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.model import HouseModel

X, y = load_and_preprocess()

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

model = HouseModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    pred = model(X)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/model.pth")

