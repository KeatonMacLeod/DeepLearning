import torch
import torch.nn as nn
import torch.optim as optim
import os


class Optimizer:
    def __init__(self, image_classifier, data_loader):
        self.image_classifier = image_classifier
        self.data_loader = data_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.image_classifier.parameters(), lr=0.001, momentum=0.9)
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.epochs = 100
        self.best_accuracy = 0
        self.gpu_available = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.gpu_available else "cpu")
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

            if batch_id % 10 == 9:
                print('[%d, %5d] TRAINING LOSS: %.3f' %
                      (epoch + 1, batch_id + 1, train_loss / 10))
                train_loss = 0.0

    def validate(self, epoch):
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

                if batch_id % 10 == 9:
                    print('[%d, %5d] TEST LOSS: %.3f' %
                          (epoch + 1, batch_id + 1, test_loss / 10))
                    test_loss = 0.0

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > self.best_accuracy:
            print('Saving..')
            state = {
                'net': self.image_classifier.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            self.best_accuracy = acc

    def optimize(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            self.validate(epoch)
