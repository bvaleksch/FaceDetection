from PIL import Image, ImageDraw
from dataset import create_datasets
import numpy as np
import torch as torch


dataset_folder = "/Datasets/CelebA/"
ds_train, ds_test = create_datasets(dataset_folder, image_size=(256, 256), absolute_coordinates=True, seed=42)

def show_random_photo(ind):
    img, bbox = ds_train[ind]
    img *= 255
    img = img.numpy().astype(np.uint8).reshape((256, 256))
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    x, y, w, h = bbox 
    x1, y1, x2, y2 = x, y, x + w, y + h
    for a, b, c, d in [(x1, y1, x1, y2), (x1, y2, x2, y2), (x2, y2, x2, y1), (x2, y1, x1, y1)]:
    	draw.line((int(a), int(b), int(c), int(d)), fill=255)
    img.show()

show_random_photo(5)