import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from clases import Vex

transform = transforms.ToTensor()

train_dataset = MNIST(
    root="data",
    train=True,
    download=False,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

load_weights = True

try:
    if load_weights:
        vex = Vex(10)
        vex.load_state_dict(torch.load(f"weights\\vex_weight.pth", weights_only=True))
        vex.train()
        print("Ваги успішно завантажились")
    else:
        vex = Vex(10)
except FileNotFoundError:
    print("Вагів немає навчання починається з нуля")
    vex = Vex(10)

except Exception as e:
    print(f"Не вдалося завантажити ваги: {e}")
    vex = Vex(10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(vex.parameters(), lr=0.00001)

for epoch in range(50):
    total_loss = 0

    for x_train, y_true in train_loader:
        optimizer.zero_grad()
        y_pred = vex(x_train)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)


    print(f"{epoch+1}% | Loss: {average_loss:.8f}")
    torch.save(vex.state_dict(), f"weights/vex_weight.pth")
    
