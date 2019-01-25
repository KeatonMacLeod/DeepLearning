import torch
import torch.nn as nn
import torch.optim as optim
import os
from Visualization.DynamicLossVisualizer import DynamicLossVisualizer


class Optimizer:
    def __init__(self, image_classifier, data_loader):
        self.image_classifier = image_classifier
        self.data_loader = data_loader
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.optimizer = optim.SGD(self.image_classifier.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.epochs = 1
        self.iterations = len(self.data_loader.training_generator)
        self.best_accuracy = 0
        self.loss_print_iterator = 10  # How often the losses print
        self.gpu_available = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.gpu_available else "cpu")
        # self.loss_visualizer = LossVisualizer()
        self.dynamic_loss_visualizer = DynamicLossVisualizer(self.epochs, self.iterations)
        self.optimize()

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        train_loss = 0
        correct = 0
        total = 0

        # Training
        for batch_id, batch in enumerate(self.data_loader.training_generator):
            local_batch, local_labels = batch
            if self.gpu_available:
                local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.image_classifier(local_batch)
            loss = self.criterion(outputs, local_labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += local_labels.size(0)
            correct += predicted.eq(local_labels).sum().item()

            if batch_id % self.loss_print_iterator == 9:
                print('[%d, %5d] TRAINING LOSS: %.3f' %
                      (epoch + 1, batch_id + 1, train_loss / self.loss_print_iterator))
                self.dynamic_loss_visualizer.update_visualization_loss(train_loss / self.loss_print_iterator, training=True)
                train_loss = 0.0

    def validate(self, epoch):
        validation_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            # Training
            for batch_id, batch in enumerate(self.data_loader.validation_generator):
                local_batch, local_labels = batch
                if self.gpu_available:
                    local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)

                outputs = self.image_classifier(local_batch)
                loss = self.criterion(outputs, local_labels)

                validation_loss += loss.item()
                _, predicted = outputs.max(1)
                total += local_labels.size(0)
                correct += predicted.eq(local_labels).sum().item()

                if batch_id % self.loss_print_iterator == 9:
                    print('[%d, %5d] VALIDATION LOSS: %.3f' %
                          (epoch + 1, batch_id + 1, validation_loss / self.loss_print_iterator))
                    # self.dynamic_loss_visualizer.update_visualization_loss(test_loss / self.loss_print_iterator, training=False)
                    validation_loss = 0.0

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > self.best_accuracy:
            print('- Saving Most Accurate Model -')
            state = {
                'net': self.image_classifier.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            self.best_accuracy = acc
        print()

    def test(self):
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            # Training
            for batch_id, batch in enumerate(self.data_loader.validation_generator):
                local_batch, local_labels = batch
                if self.gpu_available:
                    local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)

                outputs = self.image_classifier(local_batch)
                loss = self.criterion(outputs, local_labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += local_labels.size(0)
                correct += predicted.eq(local_labels).sum().item()

                if batch_id % self.loss_print_iterator == 9:
                    print('[%5d] TEST LOSS: %.3f' %
                          (batch_id + 1, test_loss / self.loss_print_iterator))
                    test_loss = 0.0

        print("- MODEL TRAINING, VALIDATION AND TESTING COMPLETED -")

    def optimize(self):
        # The model will be trained with the training data in the training stage on a training data set
        # according to the training_set_percentage, and validated during the validation stage according
        # to the validation_set_percentage
        for epoch in range(self.epochs):
            self.train(epoch)
            self.validate(epoch)

        # Calculates the final overall accuracy of the trained model using the testing set
        self.test()
