from torch.utils import data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class ImageDataset(data.Dataset):
    def __init__(self, ids, labels, data_directory):
        self.ids = ids
        self.labels = labels
        self.data_directory = data_directory
        self.image_size = 200

        # Transforms the images into a representation which can be used for training and testing
        self.transform = transforms.Compose([transforms.CenterCrop(self.image_size), transforms.ToTensor()])

    # Returns the number of samples
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image = Image.open(self.data_directory + "\\" + self.ids[index])
        image.load()
        image = self.transform(image)
        image_data = np.asarray(image, dtype="float32")
        label = self.labels[self.ids[index]]
        return [image_data, label]
