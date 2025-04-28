import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn 
from dataset import create_datasets
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def calculate_error(epoch, mode, ds, fn):
    progress = Progress(
        TextColumn("Epoch {task.fields[epoch]}: "),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("Error {task.fields[error]:.5f}: "),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[green]" + mode))
    task = progress.add_task("", epoch=epoch, error=0, total=len(ds))
    progress.start()
    
    err = 0
    for n, (x, y) in enumerate(ds, 1):
        x, y = x.to(device), y.to(device)
        local_error = fn(x, y)
        err += local_error
        progress.update(task, error=err.item()/n, advance=1, update=True)

    err = err.item() / len(ds)

    progress.refresh()
    progress.stop()
    progress.console.clear_live()

    return err

def train_epoch(epoch):
    def fn(x, y):
        predict = model(x)
        local_error = torch.mean(loss(predict, y))
        optimizer.zero_grad()
        local_error.backward()
        optimizer.step()

        return local_error

    model.train()

    return calculate_error(epoch, "Training", dl_train, fn)
    
def evaluate_epoch(epoch):
    def fn(x, y):
        with torch.no_grad():
            predict = model(x)
            local_error = torch.mean(loss(predict, y))
        return local_error

    model.eval()

    return calculate_error(epoch, "Evaluating", dl_test, fn)

def loss(output, target):
    # Создаем новые тензоры для хранения измененных значений
    output_boxes = output.clone()
    target_boxes = target.clone()

    # Преобразуем координаты (x1, y1, w, h) в (x1, y1, x2, y2)
    output_boxes[:, 2] += output_boxes[:, 0]  # x2 = x1 + w
    output_boxes[:, 3] += output_boxes[:, 1]  # y2 = y1 + h
    target_boxes[:, 2] += target_boxes[:, 0]  # x2 = x1 + w
    target_boxes[:, 3] += target_boxes[:, 1]  # y2 = y1 + h

    # Вычисляем потери
    return torchvision.ops.distance_box_iou_loss(output_boxes, target_boxes)


dataset_folder = "/Datasets/CelebA/"
model_name = "first_model.pt"
model_save_path = f"./models/{model_name}"
lr = 1e-4
epochs = 50
batch_size = 64
image_size = (1, 256, 256)
absolute_coordinates = False
model = FirstModel(image_size).to(device)
#model.load_state_dict(torch.load("./models/first_model.pt", map_location=device))
ds_train, ds_test = create_datasets(dataset_folder, image_size=image_size[1:], absolute_coordinates=absolute_coordinates, seed=42)
#ds_train.load()
#ds_test.load()
dl_train, dl_test = DataLoader(ds_train, batch_size, shuffle=True), DataLoader(ds_test, batch_size, shuffle=False)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

print("Device:", device)
print("Absolute coordinates:", absolute_coordinates)
print(f"Total number of parameters in the model: {get_n_params(model):,}")
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

