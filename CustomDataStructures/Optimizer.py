import torch
import torch.nn as nn
import torch.optim as optim


class Optimizer:
    def __init__(self, image_classifier, data_loader):
        self.image_classifier = image_classifier
        self.data_loader = data_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.image_classifier.parameters(), lr=0.001, momentum=0.9)
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.epochs = 2
        self.gpu_available = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.gpu_available else "cpu")

    def train(self):
        for epoch in range(self.epochs):
            running_loss = 0.0

            # Training
            for local_batch, local_labels in self.data_loader.training_generator:
                if self.gpu_available:
                    local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.image_classifier(local_batch)
                loss = self.criterion(outputs, self.data_loader.classes)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

            # # Validation
            # with torch.set_grad_enabled(False):
            #     for local_batch, local_labels in data_loader.validation_generator:
            #         if self.gpu_available:
            #             local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)
