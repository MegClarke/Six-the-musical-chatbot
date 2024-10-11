"""FashionMNIST example using ResNet34 from torchvision.models."""
import os
import time
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

device = torch.device("cuda")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4,), (0.22,)),
    ]
)

train_set = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_set = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
dataloaders = {"train": train_loader, "test": test_loader}
dataset_sizes = {"train": len(train_loader.dataset), "test": len(test_loader.dataset)}

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, patience=5):
    """Train the model."""
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
        torch.save(model.state_dict(), best_model_params_path)

        best_acc = 0.0
        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            for phase in ["train", "test"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # If in validation phase, step the scheduler with the validation loss
                if phase == "test":
                    scheduler.step(epoch_loss)
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping after {epoch + 1} epochs due to no improvement in validation loss."
                )
                model.load_state_dict(torch.load(best_model_params_path))
                break

            print()

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best test Acc: {best_acc:.4f}")

        # Load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def visualize_model(model, num_images=6):
    """Visualize the model."""
    was_training = model.training
    model.eval()
    images_so_far = 0
    # fig = plt.figure()

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"predicted: {class_names[preds[j]]}")
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = numpy.array([0.485, 0.456, 0.406])
    std = numpy.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = numpy.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


model_ft = models.resnet34(weights="IMAGENET1K_V1")
num_ftrs = model_ft.fc.in_features

model_ft.conv1 = nn.Conv2d(
    1,
    model_ft.conv1.out_channels,
    kernel_size=model_ft.conv1.kernel_size,
    stride=model_ft.conv1.stride,
    padding=model_ft.conv1.padding,
    bias=False,
)

model_ft.fc = nn.Linear(num_ftrs, 10)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(
    model_ft.parameters(), lr=0.001, weight_decay=1e-4
)  # adam, rmsprop
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer_ft, mode="min", factor=0.1, patience=3, verbose=True
)

model_ft = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25
)
