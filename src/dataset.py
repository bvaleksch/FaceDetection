import os
import random
import torch
import cv2
import numpy as np
import pandas as pd
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def resize_with_padding_and_bbox(image, bbox, target_size=(128, 128)):
    h, w = image.shape[:2]

    x1, y1, x2, y2 = bbox  

    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)

    new_x1 = int(x1 * scale)
    new_y1 = int(y1 * scale)
    new_x2 = int(x2 * scale)
    new_y2 = int(y2 * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (target_size[1] - new_h) // 2
    bottom = target_size[1] - new_h - top
    left = (target_size[0] - new_w) // 2
    right = target_size[0] - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    final_x1 = new_x1 + left
    final_y1 = new_y1 + top
    final_x2 = new_x2 + left
    final_y2 = new_y2 + top

    return padded, (final_x1/target_size[0], final_y1/target_size[1], final_x2/target_size[0], final_y2/target_size[1])


class CompressedDataset(Dataset):
    def __init__(self, df, image_folder, image_size=(128, 128), lazy_loading=True):
        """
        Initializes the CompressedDataset with a pre-processed DataFrame.

        Parameters:
        - image_folder: Folder that contains images
        - lazy_loading: If True, images will be loaded only when accessed, 
                        which can save memory but may increase loading time.
        - df (pd.DataFrame): A DataFrame containing the annotations with the following columns:
            - filename: The ID of the image
            - x1: The x-coordinate of the top-left corner of the bounding box
            - y1: The y-coordinate of the top-left corner of the bounding box
            - w: The width of the bounding box
            - h: The height of the bounding box
            - x2: The x-coordinate of the bottom-right corner of the bounding box
            - y2: The y-coordinate of the bottom-right corner of the bounding box
        """
        self.df = df
        self.lazy_loading = lazy_loading
        self.image_folder = image_folder
        self.image_size = image_size
        self._isload = False
        self._cnt = 0
        self.data_x = [None for _ in range(len(self))]
        self.data_y = [None for _ in range(len(self))]

    def _load_image_by_index(self, ind):
        if self.data_x[ind] is not None:
            return

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        row = self.df.iloc[ind]
        img_path = os.path.join(self.image_folder, row["filename"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        img, bbox = resize_with_padding_and_bbox(img, (x1, y1, x2, y2), self.image_size)
        x1, y1, x2, y2 = bbox
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        self.data_x[ind] = encimg
        self.data_y[ind] = (min((x1, x2)), min((y1, y2)), max((x1, x2)), max((y1, y2)))

    def load(self, verbose=True):
        if verbose:
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn())
            task = progress.add_task("[green]Loading...", total=len(self))
            progress.start()

        for ind in range(len(self.df)):
            self._load_image_by_index(ind)
            if verbose:
                progress.update(task, advance=1, update=True)

        if verbose:
            progress.refresh()
            progress.stop()
            progress.console.clear_live()

        self._cnt = len(self)
        self._isload = True

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index, ignore_isload=False):
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of bounds for dataset of size {len(self)}.")
        
        if self._isload or ignore_isload:
            img, y = self.data_x[index], self.data_y[index]
            img = cv2.cvtColor(cv2.imdecode(img, 1), cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32).reshape((1, self.image_size[0], self.image_size[1]))
            img -= 31.5
            img /= 31.5
            return torch.from_numpy(img), torch.Tensor(y)
        else:
            if self.lazy_loading:
                if self.data_x[index] is None:
                    self._load_image_by_index(index)
                    self._cnt += 1
                    if self._cnt == len(self):
                        self._isload = True

                return self.__getitem__(index, True)
            else:
                raise RuntimeError("Dataset is not loaded. Please call the method load")
        

def create_datasets(folder, test_size=0.25, image_size=(128, 128), seed=None, lazy_loading=True):
    """
    Returns train and test datasets.

    The specified folder should contain the following structure:
        1) A sub-folder named "img_celeba" containing images located at:
           "./img_celeba/img_celeba/*". //I understand that this is bad, but I don't want to change the folder structure on my computer
        2) A CSV file named "list_bbox_celeba.csv" that contains bounding box information for the images.

    Parameters:
    - folder (str): The path to the main directory containing the dataset.
    """
    seed = random.randint(1, 2**32-1) if seed is None else seed
    input_folder = os.path.join(folder, "./img_celeba/img_celeba/")
    annotations_file = os.path.join(folder, "./list_bbox_celeba.csv")

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    if not os.path.isfile(annotations_file):
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

    df = pd.read_csv(annotations_file)
    df.rename(columns={"image_id": "filename", "x_1": "x1", "y_1": "y1", "width": "w", "height": "h"}, inplace=True)
    df["x2"] = df["x1"] + df["w"]
    df["y2"] = df["y1"] + df["h"]

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed)
    return (CompressedDataset(df_train, input_folder, image_size, lazy_loading),
            CompressedDataset(df_test, input_folder, image_size, lazy_loading))
    
