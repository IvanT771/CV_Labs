"""
Lab14_2: show CIFAR-10 predictions from a saved model without training.
"""

import os
import sys

import numpy as np
from matplotlib import pyplot as plt

# === 1. Import PyTorch and torchvision with a basic check ===
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    print("PyTorch/torchvision not installed.")
    print("Install with:\n  python -m pip install torch torchvision")
    sys.exit(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# === 2. CIFAR-10 test data (same normalization as training) ===
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

testset = torchvision.datasets.CIFAR10(
    root="./data_cifar10",
    train=False,
    download=True,
    transform=test_transform,
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


# === 3. CNN model definition (same as Lab14_1) ===
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


def imshow(img: torch.Tensor) -> None:
    img = img * 0.5 + 0.5
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def show_predictions(net: nn.Module, num_images: int = 4) -> None:
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    num_images = min(num_images, images.size(0))
    images = images[:num_images]
    labels = labels[:num_images]

    net.eval()
    with torch.no_grad():
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs, 1)

    plt.figure("CIFAR-10 Predictions")
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        imshow(images[i])
        gt = classes[labels[i].item()]
        pred = classes[predicted[i].item()]
        plt.title(f"GT: {gt}\nPred: {pred}")
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()


def main() -> None:
    model_path = "cifar10_cnn.pth"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Train and save it first with Lab14_1.py.")
        return

    net = Net().to(device)
    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state)
    show_predictions(net, num_images=4)


if __name__ == "__main__":
    main()
