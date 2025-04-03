import os
import torch
import cv2
import numpy as np
from tkinter import Tk, Button, filedialog, Canvas
from PIL import Image, ImageTk
from model import MyModel 

def resize_with_padding(image, target_size=(128, 128)):
    h, w = image.shape[:2]

    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (target_size[1] - new_h) // 2
    bottom = target_size[1] - new_h - top
    left = (target_size[0] - new_w) // 2
    right = target_size[0] - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded, (left/target_size[0], top/target_size[1], new_w/target_size[0], new_h/target_size[1])

image_size = (1, 128, 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel(image_size).to(device)
model.load_state_dict(torch.load("./models/first_model.pth"))
model.eval()

def draw_bounding_boxes(image, boxes, correct):
    for box in boxes:
        print(box)
        x1, y1, x2, y2 = box
        x1, x2 = x1 - correct[0], x2 - correct[0]
        x1, x2 = x1 / correct[2], x2 / correct[2]
        y1, y2 = y1 + correct[1], y2 + correct[1]
        y1, y2 = y1 / correct[3], y2 / correct[3]
        cv2.rectangle(image, (int(x1 * image.shape[1]), int(y1 * image.shape[0])),
                      (int(x2 * image.shape[1]), int(y2 * image.shape[0])), (0, 255, 0), 4)
    return image

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img_cv = cv2.imread(file_path)
        img_org = img_cv.copy()
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        img_resized, correct = resize_with_padding(img_cv, image_size[1:])
        img_resized = img_resized.reshape(image_size)
        img_tensor = (torch.tensor(img_resized, dtype=torch.float32) - 31.5) / 31.5
        img_tensor = img_tensor.unsqueeze(0).to(device)  

        with torch.no_grad():
            output = model(img_tensor)
            boxes = output.cpu().numpy()
            boxes = boxes.reshape(-1, 4)

        img_with_boxes = draw_bounding_boxes(img_org, boxes, correct)
        cv2.imwrite("test.png", img_with_boxes)

        img_with_boxes_pil = Image.fromarray(img_with_boxes)
        img_with_boxes_tk = ImageTk.PhotoImage(img_with_boxes_pil)

        canvas.image = img_with_boxes_tk 
        canvas.create_image(0, 0, anchor='nw', image=img_with_boxes_tk)

root = Tk()
root.title("Face Detection Model Tester")

canvas = Canvas(root, width=600, height=600)
canvas.pack()
load_button = Button(root, text="Load Image", command=load_image)
load_button.pack()

root.mainloop()
