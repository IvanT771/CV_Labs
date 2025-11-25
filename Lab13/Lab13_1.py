"""
ЛР13. Распознавание рукописных цифр MNIST с помощью PyTorch

Основано на примере:
https://www.kaggle.com/code/geekysaint/solving-mnist-using-pytorch
и базовых примерах из документации PyTorch.
"""

import sys

# === 1. Импорт PyTorch и torchvision с проверкой наличия ===
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
except ImportError as e:
    print("Не установлен PyTorch/torchvision.")
    print("Установите пакеты:\n  pip install torch torchvision")
    sys.exit(1)


# === 2. Гиперпараметры обучения ===
batch_size = 64
test_batch_size = 1000
epochs = 1          # для примера достаточно одного эпоха
learning_rate = 0.01
momentum = 0.9
seed = 1

torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используется устройство:", device)


# === 3. Загрузчики данных MNIST ===

transform = transforms.Compose([
    transforms.ToTensor(),                          # перевод в тензор (0..1)
    transforms.Normalize((0.1307,), (0.3081,)),     # нормализация как в примере MNIST
])

train_dataset = datasets.MNIST(
    "./data_mnist",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    "./data_mnist",
    train=False,
    download=True,
    transform=transform,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
)


# === 4. Модель сверточной нейронной сети (как в базовом примере PyTorch MNIST) ===


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # два сверточных слоя
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # полносвязные слои
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 классов цифр

    def forward(self, x):
        # вход: [batch, 1, 28, 28]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # теперь размер карты признаков: [batch, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()


# === 5. Функции обучения и тестирования (классический цикл) ===


def train(epoch: int) -> None:
    """Один проход по обучающей выборке."""
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)}]  "
                f"Loss: {loss.item():.6f}"
            )


def test() -> None:
    """Оценка точности на тестовой выборке."""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(f"\nTest set: Average loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")


def main() -> None:
    # === 6. Основной цикл обучения ===
    for epoch in range(1, epochs + 1):
        train(epoch)
        test()

    # Сохраняем обученную модель на диск
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Модель сохранена в файл mnist_cnn.pth")


if __name__ == "__main__":
    main()

