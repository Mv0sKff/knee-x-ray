import cv2
from torch.utils.data import Dataset


def read_img(file_path):
    xray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return xray


class knee_xray_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        print(self.dataset[idx])
        img = read_img(self.dataset["Path"].iloc[idx])
        name = self.dataset["Name"].iloc[idx]
        target = self.dataset["KL"].iloc[idx]

        result = {"xray": img, "name": name, "target": target}

        return result
