import torch
import torch.nn as nn
import torch.optim as optim
from clases import Model
from dataset import rect, circle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

model = Model().to(device)

criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

x_train_rect = torch.tensor([1, 0], dtype=torch.float32).to(device)
y_true_rect = rect.to(device)

x_train_circle = torch.tensor([0, 1], dtype=torch.float32).to(device)
y_true_circle = circle.to(device)

for epoch in range(30001):
    optimizer.zero_grad()

    y_pred_rect = model(x_train_rect)
    y_pred_circle = model(x_train_circle)

    loss_rect = criterion(y_pred_rect, y_true_rect)
    loss_circle = criterion(y_pred_circle, y_true_circle)

    loss_rect.backward()
    loss_circle.backward()

    optimizer.step()

    if epoch % 300 == 0:
        print(f"{epoch/300}% | Loss_rect: {loss_rect.item():.4f} | Loss_circle: {loss_circle.item():.4f}")

torch.save(model.state_dict(), "model_weights.pth")