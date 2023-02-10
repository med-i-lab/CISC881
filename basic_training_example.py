import argparse
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.datasets import MNIST


# pytorch neural networks are defined by subclassing nn.Module and defining a forward function
class ConvNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7 * 7 * 64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def parse_args():
    """Read command line arguments for this script."""
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )

    args = parser.parse_args()
    return args


def main(args):
    # define the device (gpu or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the model
    model = ConvNet().to(device)

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # define the loss function
    criterion = nn.CrossEntropyLoss()

    # define the data transforms
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    # load the training and test data
    train_data = MNIST(root="data", train=True, transform=transform, download=True)
    test_data = MNIST(root="data", train=False, transform=transform, download=True)

    # define the data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=args.batch_size, shuffle=False
    )

    # train the model
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, args.epochs, i + 1, len(train_loader), loss.item()
                    )
                )

    # test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            "Test Accuracy of the model on the 10000 test images: {} %".format(
                100 * correct / total
            )
        )

    # save the model checkpoint
    torch.save(model.state_dict(), "model.ckpt")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
