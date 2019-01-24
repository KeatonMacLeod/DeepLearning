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

1.) Loading a custom image data utilizing parallel CPU computation for extremely efficient loading times (X)
2.) Checking for the presence of a GPU utilizing it to train the network (X)
3.) Saving the weights of the most accurate model on the validation set (X)
4.) Loading a network from a file -> don't need to retrain each time you want to classify images -> NOT YET IMPLEMENTED
"""

from CustomDataStructures.DataLoader import DataLoader
from Classifier.ImageClassifier import ImageClassifier
from CustomDataStructures.Optimizer import Optimizer


def main():
    image_classifier = ImageClassifier()
    data_loader = DataLoader()
    optimizer = Optimizer(image_classifier, data_loader)


if __name__ == "__main__":
    main()
