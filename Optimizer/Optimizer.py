import torch
import torch.nn as nn
import torch.optim as optim
import os
from Visualization.DynamicLossVisualizer import DynamicLossVisualizer


class Optimizer:
    def __init__(self, image_classifier, data_loader):
        self.image_classifier = image_classifier
        self.data_loader = data_loader
        self.checkpoint_directory = './checkpoint/'
        self.last_run_directory = './last-run/'
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
        self.dynamic_loss_visualizer = DynamicLossVisualizer(self.epochs)
        self.optimize()

    def train(self, epoch):
        print("-" * 100)
        print('Epoch: %d' % epoch)
        print("-" * 100)
        epoch_train_loss = 0
        train_loss = 0
        correct = 0
        total = 0

        # Training
        for batch_id, batch in enumerate(self.data_loader.training_generator):
            local_batch, local_labels = batch
            if self.gpu_available:
                self.image_classifier.cuda()
                local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward + backward + optimize
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
                epoch_train_loss = train_loss / self.loss_print_iterator
                train_loss = 0.0

        return epoch_train_loss

    def validate(self, epoch):
        epoch_validation_loss = 0
        validation_loss = 0
        correct = 0
        total = 0

        # We don't need to store gradients during validation since we're not training at this point, just testing the
        # current classification ability of our model, so we use torch.no_grad() to save memory
        with torch.no_grad():
            for batch_id, batch in enumerate(self.data_loader.validation_generator):
                local_batch, local_labels = batch
                if self.gpu_available:
                    local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)

                # Test validation loss on the local batch
                outputs = self.image_classifier(local_batch)
                loss = self.criterion(outputs, local_labels)

                validation_loss += loss.item()
                _, predicted = outputs.max(1)
                total += local_labels.size(0)
                correct += predicted.eq(local_labels).sum().item()

                if batch_id % self.loss_print_iterator == 9:
                    print('[%d, %5d] VALIDATION LOSS: %.3f' %
                          (epoch + 1, batch_id + 1, validation_loss / self.loss_print_iterator))
                    epoch_validation_loss = validation_loss / self.loss_print_iterator
                    validation_loss = 0.0

        return correct, total, epoch_validation_loss

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
                    print("FINAL MODEL TEST LOSS: {}".format(test_loss / self.loss_print_iterator))
                    test_loss = 0.0
        if not os.path.isdir('last-run'):
            os.mkdir('last-run')
        self.dynamic_loss_visualizer.figure.savefig(self.last_run_directory)
        print("- MODEL TRAINING, VALIDATION AND TESTING COMPLETED -")

    def save_most_accurate_model(self, epoch, correct_predictions, total_predictions):
        # Save checkpoint.
        accuracy = 100. * correct_predictions / total_predictions
        print('Validation Accuracy: {}'.format(accuracy))
        if accuracy > self.best_accuracy:
            print('Current Validation Accuracy: {} better than Current Best Accuracy: {}'.format(accuracy, self.best_accuracy))
            print('- Saving Model -')
            state = {
                'net': self.image_classifier.state_dict(),
                'acc': accuracy,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, self.checkpoint_directory + 'ckpt.t7')
            self.dynamic_loss_visualizer.figure.savefig(self.checkpoint_directory)
            self.best_accuracy = accuracy
        print()

    def optimize(self):
        # The model will be trained with the training data in the training stage on a training data set
        # according to the training_set_percentage, and validated during the validation stage according
        # to the validation_set_percentage
        for epoch in range(self.epochs):
            epoch_train_loss = self.train(epoch)
            correct_predictions, total_predictions, epoch_validation_loss = self.validate(epoch)
            self.save_most_accurate_model(epoch, correct_predictions, total_predictions)
            self.dynamic_loss_visualizer.update_visualization_loss(epoch_train_loss, epoch_validation_loss)

        # Calculates the final overall accuracy of the trained model using the testing set
        self.test()
