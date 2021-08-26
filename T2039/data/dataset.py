from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, img_paths, transform, labels=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        if self.transform:
            image = self.transform(image)
        if self.labels:
            label = self.labels[idx]
            return image, label
        return image