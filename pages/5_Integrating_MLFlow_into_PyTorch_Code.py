# pages/5_Integrating_MLFlow_into_PyTorch_Code.py

import streamlit as st

st.title("Integrating MLFlow into Your PyTorch Code")

st.write("""
In this section, we'll integrate **MLFlow Tracking** into your provided PyTorch MNIST MLP code. This will enable you to log hyperparameters, metrics, and artifacts, and visualize them using the MLFlow UI.

### Step 1: Import MLFlow

At the beginning of your `mnist_mlp.py` script, import MLFlow:

```python
import mlflow
import mlflow.pytorch
```

### Step 2: Set the Experiment (Optional)

Optionally, you can set a specific experiment name for better organization:

```python
mlflow.set_experiment("MNIST_MLP_Experiment")
```

### Step 3: Start an MLFlow Run
Wrap your training and validation code within an MLFlow run context:

```python
with mlflow.start_run():
    # Training and validation code goes here
```

### Step 4: Log Hyperparameters
Log the hyperparameters at the start of the run:

```python
# Hyperparameters
epochs = 20
valid_every_n_step = 100
train_valid_frac = 0.8
batch_size = 1_024
lr = 1e-3

# Log hyperparameters
mlflow.log_param("epochs", epochs)
mlflow.log_param("valid_every_n_step", valid_every_n_step)
mlflow.log_param("train_valid_frac", train_valid_frac)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("learning_rate", lr)
```

### Step 5: Log Model Architecture

Optionally, log details about your model architecture:
```python
mlflow.log_param("model_architecture", "3-layer MLP with ReLU activations")
```

### Step 6: Log Metrics During Training
Inside your training loop, log training loss and accuracy at each step:

```python
train_history, valid_history, lr_history = [], [], []
train_batches = iter(cycle(train_loader))
pbar = tqdm(range(epochs * len(train_loader)), desc="Train")

for step in pbar:
    train_loss, train_acc = train(model, optim, scheduler, train_batches)
    train_history.append(train_loss)
    lr_history.append(scheduler.get_last_lr()[0])

    # Log training metrics
    mlflow.log_metric("train_loss", train_loss, step=step)
    mlflow.log_metric("train_accuracy", train_acc, step=step)
    mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=step)

    if step % valid_every_n_step == 0:
        valid_loss, valid_acc = valid(model, valid_loader, valid_set)
        valid_history.append(valid_loss)

        # Log validation metrics
        mlflow.log_metric("valid_loss", valid_loss, step=step)
        mlflow.log_metric("valid_accuracy", valid_acc, step=step)

    # Existing pbar.set_postfix() code...
```
### Step 7: Log Test Metrics
After evaluating the model on the test set, log the results:
```python
# Test the model
test_loss, test_acc = valid(model, test_loader, test_set)
print(f"Test Loss: {test_loss:.2e}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Log test metrics
mlflow.log_metric("test_loss", test_loss)
mlflow.log_metric("test_accuracy", test_acc)
```

### Step 8: Log the Final Model
After training, log the model as an artifact:
```python
# Save the Model Weights
torch.save(model.state_dict(), "mnist.pt")
mlflow.log_artifact("mnist.pt")

# Optionally, log the model using MLFlow's PyTorch integration
mlflow.pytorch.log_model(model, "model")
```

### Complete Code with MLFlow Integration
Below is the complete code with MLFlow integrated:
```python
from __future__ import annotations

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from typing import Iterator, Sized

import torch
import torch.nn as nn
import torch.nn.functional as F

import mlflow
import mlflow.pytorch

@torch.jit.script
def criterion(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.nll_loss(torch.log_softmax(logits, dim=1), labels)

def train(model: nn.Module, optim: Optimizer, scheduler: LRScheduler, loader: Iterator[tuple[torch.Tensor, torch.Tensor]]) -> tuple[float, float]:
    model.train()
    x, l = next(loader)
    optim.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, l)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    optim.step()
    scheduler.step()
    acc = (torch.argmax(logits, dim=1) == l).sum() / x.size(0)
    return loss.item(), acc.item()

@torch.inference_mode()
def valid(model: nn.Module, loader: DataLoader, dataset: Sized) -> tuple[float, float]:
    model.eval()
    loss = torch.tensor(0, dtype=torch.float32)
    acc = torch.tensor(0, dtype=torch.float32)
    for x, l in loader:
        logits = model(x)
        loss += criterion(logits, l)
        acc += (torch.argmax(logits, dim=1) == l).sum().item()
    return loss.item() / len(loader), acc.item() / len(dataset)

if __name__ == "__main__":
    from itertools import cycle
    from torch.optim.adamw import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    from torch.utils.data import Subset
    from torchvision.datasets.mnist import MNIST
    from tqdm import tqdm

    import matplotlib.pyplot as plt
    import torch.onnx as tonnx
    import torchvision.transforms as T

    # Import MLFlow
    import mlflow
    import mlflow.pytorch

    # Set the experiment (optional)
    mlflow.set_experiment("MNIST_MLP_Experiment")

    with mlflow.start_run():
        # ========== Hyperparameters
        epochs = 20
        valid_every_n_step = 100
        train_valid_frac = 0.8
        batch_size = 1024
        lr = 1e-3

        # Log hyperparameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("valid_every_n_step", valid_every_n_step)
        mlflow.log_param("train_valid_frac", train_valid_frac)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)

        # ========== Dataset Splits
        transform = T.ToTensor()

        dataset = MNIST("/tmp/datasets", train=True, download=True, transform=transform)
        valid_indices = range(int(train_valid_frac * len(dataset)), len(dataset))
        train_indices = range(0, int(train_valid_frac * len(dataset)))

        valid_set = Subset(dataset, indices=valid_indices)
        train_set = Subset(dataset, indices=train_indices)
        test_set = MNIST("/tmp/datasets", train=False, download=True, transform=transform)

        # ========== Dataloaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        # ========== Setup Model, Optimizer, and LRScheduler
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 28 * 28, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )
        optim = AdamW(model.parameters(), lr, fused=True)
        scheduler = OneCycleLR(optim, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

        # ========== Training Loop w/ Validation
        valid_loss, valid_acc = 0, 0
        train_history, valid_history, lr_history = [], [], []
        train_batches = iter(cycle(train_loader))
        pbar = tqdm(range(epochs * len(train_loader)), desc="Train")

        for step in pbar:
            train_loss, train_acc = train(model, optim, scheduler, train_batches)
            train_history.append(train_loss)
            lr_history.append(scheduler.get_last_lr()[0])

            # Log training metrics
            mlflow.log_metric("train_loss", train_loss, step=step)
            mlflow.log_metric("train_accuracy", train_acc, step=step)
            mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=step)

            if step % valid_every_n_step == 0:
                valid_loss, valid_acc = valid(model, valid_loader, valid_set)
                valid_history.append(valid_loss)

                # Log validation metrics
                mlflow.log_metric("valid_loss", valid_loss, step=step)
                mlflow.log_metric("valid_accuracy", valid_acc, step=step)

            pbar.set_postfix(
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                train_loss=f"{train_loss:.2e}",
                train_acc=f"{train_acc * 100:.2f}%",
                valid_loss=f"{valid_loss:.2e}",
                valid_acc=f"{valid_acc * 100:.2f}%",
            )

        # ========== Monitoring Plot
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_history, label="train")
        plt.plot(range(0, len(train_history), valid_every_n_step), valid_history, label="valid")
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.subplot(1, 2, 2)
        plt.plot(lr_history, label="Learning Rate")
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("LR")
        plt.tight_layout()
        plt.savefig("training_plots.png")
        mlflow.log_artifact("training_plots.png")

        # ========== Test
        test_loss, test_acc = valid(model, test_loader, test_set)
        print(f"Test Loss: {test_loss:.2e}")
        print(f"Test Accuracy: {test_acc * 100:.2f}%")

        # Log test metrics
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        # ========== Save the Model Weights
        torch.save(model.state_dict(), "mnist.pt")
        mlflow.log_artifact("mnist.pt")

        # Save and log the ONNX model
        tonnx.export(model, torch.empty((1, 1, 28, 28)), "mnist.onnx", input_names=["input"], output_names=["output"])
        mlflow.log_artifact("mnist.onnx")

        # Optionally, log the model using MLFlow's PyTorch integration
        mlflow.pytorch.log_model(model, "model")
```
### Summary
In this step, we've integrated MLFlow into our PyTorch MNIST MLP code by:

- Importing MLFlow libraries.
- Setting the experiment name (optional for better organization).
- Starting an MLFlow run to encapsulate the experiment.
- Logging hyperparameters at the beginning of the run.
- Logging metrics (loss, accuracy, learning rate) during training and validation.
- Logging test metrics after evaluation on the test set.
- Logging artifacts such as the trained model weights, ONNX model, and training plots.

### Next Steps
With MLFlow integrated into your code, you can now:

- Run your training script as usual.
- Launch the MLFlow UI using `mlflow ui` to visualize the logged data.
- Compare different runs by adjusting hyperparameters and observing the results.
""")