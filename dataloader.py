import cv2
import pathlib
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, anno_name, dataset_type, img_size):
        self.root = pathlib.Path(root)
        self.anno_name = anno_name
        self.dataset_type = dataset_type
        self.img_size = img_size

        self.data = self._read_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self._read_img(self.data[idx]['ImageID'])
        landmark = self.data[idx]['landmarks']

        if image.shape[-1] != 1 and image.shape[-1] != 3:
            image = torch.from_numpy(image.astype(np.float32) / 255).unsqueeze(-1).permute(2, 0, 1)
        else:
            image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

        landmark = np.array(list(landmark.split(' '))).astype(np.float32) / self.img_size[0]
        landmark = torch.from_numpy(landmark)

        return image, landmark

    def get_exam(self, idx):
        image = self._read_img(self.data[idx]['ImageID'])
        landmark = self.data[idx]['landmarks']

        image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)
        landmark = np.array(list(landmark.split(' '))).astype(np.float32)
        landmark = torch.from_numpy(landmark)

        return image, landmark

    def _read_data(self):
        csv_name = f"{self.root}/{self.anno_name}.csv"
        csvs = pd.read_csv(csv_name)

        data = []
        for img_name, landmark in zip(csvs['ImageID'], csvs['landmarks']):
            data.append({
                'ImageID': img_name,
                'landmarks': landmark
            })
        return data

    def _read_img(self, img_name):
        image_path = self.root / self.dataset_type / f"{img_name}"

        img = cv2.imread(str(image_path))
        if img.shape[-1] != self.img_size[-1]:
            if img.shape[-1] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))

        return img


def main():
    root = './dataset/300W-LP'
    anno_name = 'train'

    cd = CustomDataset(root, anno_name, 'images', img_size=(640, 480, 3))

    img, landmark = cd.get_exam(10000)

    img = (img.permute(1, 2, 0).numpy().astype(np.float32))

    for i in range(0, landmark.shape[0], 2):
        img = cv2.circle(img, (int(landmark[i].item()*640), int(landmark[i+1].item()*480)), 2, (0, 0, 255), -1)

    cv2.imshow('t', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
