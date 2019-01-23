"""
This tutorial assumes you have the following dataset:

https://www.kaggle.com/alxmamaev/flowers-recognition

Which should be stored in a file structure such as:

        /data/daisy
        /data/tulip
        /data/dandelion
        .....

The purpose of this example is to implement all of the concepts covered in the generic PyTorch 60-minute blitz, but
to add additional functionality that will be used in the neural network for our project, including:

1.) Loading a custom image data utilizing parallel CPU computation for extremely efficient loading times
2.) Checking for the presence of a GPU utilizing it to train the network
3.) Saving the weights at the end of training
4.) Loading a network from a file -> don't need to retrain each time you want to classify images

This program also splits up training / validation data based on the fact that each class has roughly the same
number of images in each class
"""

from CustomDataStructures.DataLoader import DataLoader
from CustomDataStructures.ImageClassifier import ImageClassifier
from CustomDataStructures.Optimizer import Optimizer


def main():
    image_classifier = ImageClassifier()
    data_loader = DataLoader()
    data_loader.load_data()
    optimizer = Optimizer(image_classifier, data_loader)
    optimizer.train()




if __name__ == "__main__":
    main()
