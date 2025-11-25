"""
ЛР14. Классификация изображений CIFAR‑10 с помощью PyTorch

Основано на официальном туториале:
https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import sys

# === 1. Импорт PyTorch и torchvision с проверкой наличия ===
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
except ImportError as e:
    print("Не установлен PyTorch/torchvision.")
    print("Установите пакеты:\n  pip install torch torchvision")
    sys.exit(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используется устройство:", device)


# === 2. Подготовка данных CIFAR‑10 ===

# Нормализация как в официальном примере:
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data_cifar10",
    train=True,
    download=True,
    transform=transform,
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
)

testset = torchvision.datasets.CIFAR10(
    root="./data_cifar10",
    train=False,
    download=True,
    transform=transform,
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# === 3. Определение сверточной нейросети (как в туториале CIFAR‑10) ===


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(device)


# === 4. Функция потерь и оптимизатор ===

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# === 5. Обучение сети ===


def train(num_epochs: int = 2) -> None:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # каждые ~2000 мини‑батчей
                print(
                    f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}"
                )
                running_loss = 0.0

    print("Обучение завершено.")


# === 6. Оценка точности на тестовой выборке ===


def evaluate() -> None:
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Точность сети на 10000 тестовых изображениях: {100 * correct / total:.2f}%")


def main() -> None:
    train(num_epochs=2)
    evaluate()
    torch.save(net.state_dict(), "cifar10_cnn.pth")
    print("Модель сохранена в файл cifar10_cnn.pth")


if __name__ == "__main__":
    main()

