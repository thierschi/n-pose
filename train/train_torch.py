from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, dl_train, dl_test, loss_fn, optimizer, device="cpu"):
        self.model = model
        self.dl_train = dl_train
        self.dl_test = dl_test
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.writer = None

        self.model.to(self.device)

    def use_device(self, device):
        self.device = device

    def train_one_epoch(self, epoch_idx):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, (x, y) in enumerate(self.dl_train):
            # Every data instance is an input + label pair
            x, y = x.to(self.device), y.to(self.device)

            # Zero your gradients for every batch!

            # Make predictions for this batch
            outputs = self.model(x)
            self.optimizer.zero_grad()

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, y)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 500 == 499:
                last_loss = running_loss / 1000  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_idx * len(self.dl_train) + i + 1
                self.writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def train(self, no_epochs):
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0

        EPOCHS = no_epochs

        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.dl_test):
                    vinputs, vlabels = vdata
                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            self.writer.add_scalars('Training vs. Validation Loss',
                                    {'Training': avg_loss, 'Validation': avg_vloss},
                                    epoch_number + 1)
            self.writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1
