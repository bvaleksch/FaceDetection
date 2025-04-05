import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn 
from dataset import create_datasets
from model import *

"""
The train_epoch and evaluate_epoch functions are poorly implemented due to code duplication.
It is necessary to create a single common function to avoid copy-pasting code.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(epoch):
    model.train()
    progress = Progress(
        TextColumn("Epoch {task.fields[epoch]}: "),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("Error {task.fields[error]:.5f}: "),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[green]Training"))
    task = progress.add_task("", epoch=epoch, error=0, total=len(dl_train))

    progress.start()
    err = 0
    for n, (x, y) in enumerate(dl_train, 1):
        x, y = x.to(device), y.to(device)
        predict = model(x)
        local_error = torch.mean(loss(predict, y))

        optimizer.zero_grad()
        local_error.backward()
        optimizer.step()

        err += local_error
        progress.update(task, error=err.item()/n, advance=1, update=True)
    err = err.item() / len(dl_train)

    progress.refresh()
    progress.stop()
    progress.console.clear_live()

    return err

def evaluate_epoch(epoch):
    model.eval()
    progress = Progress(
        TextColumn("Epoch {task.fields[epoch]}: "),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("Error {task.fields[error]:.5f}: "),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[green]Evaluating"))
    task = progress.add_task("", epoch=epoch, error=0, total=len(dl_test))

    progress.start()
    err = 0
    for n, (x, y) in enumerate(dl_test, 1):
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            predict = model(x)
            local_error = torch.mean(loss(predict, y))
            err += local_error
        progress.update(task, error=err.item()/n, advance=1, update=True)
    err = err.item() / len(dl_test)

    progress.refresh()
    progress.stop()
    progress.console.clear_live()

    return err

dataset_folder = "/Datasets/CelebA/"
model_name = "first_model.pth"
model_save_path = f"./models/{model_name}"
lr = 1e-3
epochs = 50
batch_size = 64
image_size = (1, 128, 128)
model = MyModel(image_size).to(device)
ds_train, ds_test = create_datasets(dataset_folder, image_size=image_size[1:], seed=42)
dl_train, dl_test = DataLoader(ds_train, batch_size, shuffle=True), DataLoader(ds_test, batch_size, shuffle=False)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
loss = torchvision.ops.distance_box_iou_loss

train_error = []
test_error = []
for epoch in range(epochs):
    train_error.append(train_epoch(epoch + 1))
    test_error.append(evaluate_epoch(epoch + 1))

torch.save(model.state_dict(), model_save_path)

plt.plot(range(1, epochs + 1), train_error, label='Training Error', marker='o', color='blue')
plt.plot(range(1, epochs + 1), test_error, label='Test Error', marker='o', color='orange')
plt.title('Training and Test Error Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.grid(True)
plt.legend()
plt.show()



