import os
from torch.utils import data
from CustomDataStructures.ImageDataset import ImageDataset


class DataLoader:

    def __init__(self):
        self.data_directory = "\\data"
        self.params = {'batch_size': 64, 'shuffle': True, 'num_workers': 6}
        self.partition = {"train": [], "validation": []}
        self.label_lookup = {}
        self.labels = {}
        self.classes = []
        self.train_set_percentage = .60

        self.training_generator = None
        self.validation_generator = None

        # Populate self.training_generator and validation_generator
        self.load_data()

    # Loads all of the data into the corresponding training and validation generators
    def load_data(self):
        self.create_partitions_and_labels()

        training_set = ImageDataset(self.partition['train'], self.labels, os.getcwd() + self.data_directory)
        self.training_generator = data.DataLoader(training_set, **self.params)

        validation_set = ImageDataset(self.partition['validation'], self.labels, os.getcwd() + self.data_directory)
        self.validation_generator = data.DataLoader(validation_set, **self.params)

    # Splits the data up into training and validation sets
    def create_partitions_and_labels(self):
        self.classes = [d for d in os.listdir(os.getcwd() + self.data_directory)]
        for i, class_name in enumerate(self.classes):
            class_full_path = os.getcwd() + self.data_directory + "\\" + class_name
            num_images = len(os.listdir(class_full_path))
            train_set_size = num_images * self.train_set_percentage
            self.label_lookup[i] = class_name
            for root, dirs, files in os.walk(class_full_path):
                for j, image_id in enumerate(files):
                    if j < train_set_size:
                        self.partition['train'].append(class_name + "\\" + image_id)
                    else:
                        self.partition['validation'].append(class_name + "\\" + image_id)
                    self.labels[class_name + "\\" + image_id] = i
