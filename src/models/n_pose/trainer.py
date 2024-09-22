import os

import torch


class Trainer:
    """
    Trainer class that trains a model and evaluates it.
    """

    def __init__(self, model, train_loader, test_loader=None, metrics=None, callback=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.metrics = metrics if metrics is not None else []
        self.callback = callback

        self.device = torch.device('cpu')
        self.model.to(self.device)

    def use_device(self, device):
        """
        Set the device to use for training.
        :param device: Torch device
        """
        self.device = device
        self.model.to(self.device)
        print(f"Using device {device}")

    def use_best_device(self):
        """
        Set the device to use for training based on availability.
        """
        if torch.cuda.is_available():
            self.use_device(torch.device('cuda'))
        elif torch.backends.mps.is_available():
            self.use_device(torch.device('mps'))
        else:
            self.use_device(torch.device('cpu'))

    def train(self, num_epochs, criterion, optimizer, scheduler=None):
        """
        Train the model for a given number of epochs.
        :param num_epochs: Epochs to train for
        :param criterion: Loss function
        :param optimizer:
        :param scheduler:
        """
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            # Reset metrics
            for metric in self.metrics:
                metric.reset()

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()

                # Update metrics
                for metric in self.metrics:
                    metric.update(outputs, targets)

            avg_train_loss = running_loss / len(self.train_loader)
            train_metrics = [metric.compute() for metric in self.metrics]

            avg_test_loss = 0
            val_metrics = []
            if self.test_loader:
                avg_test_loss, val_metrics = self.evaluate(criterion)

            if scheduler:
                scheduler.step(avg_test_loss)

            # Invoke callback with training values
            if self.callback:
                self.callback([epoch, avg_train_loss, avg_test_loss] + train_metrics + val_metrics)

    def evaluate(self, criterion):
        """
        Evaluate the model on the test set.
        :param criterion: Loss function
        """
        self.model.eval()
        test_loss = 0.0
        val_metrics = [metric.__class__() for metric in self.metrics]  # Create new instances for validation metrics

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                # Update validation metrics
                for metric in val_metrics:
                    metric.update(outputs, targets)

        avg_test_loss = test_loss / len(self.test_loader)
        val_metrics = [metric.compute() for metric in val_metrics]
        return avg_test_loss, val_metrics

    def evaluate_with_loader(self, data_loader, criterion):
        """
        Evaluate the model on a different data loader.
        :param data_loader: New data loader
        :param criterion: Loss function
        """
        self.model.eval()
        test_loss = 0.0
        val_metrics = [metric.__class__() for metric in self.metrics]  # Create new instances for validation metrics

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                # Update validation metrics
                for metric in val_metrics:
                    metric.update(outputs, targets)

        avg_test_loss = test_loss / len(data_loader)
        val_metrics = [metric.compute() for metric in val_metrics]
        return avg_test_loss, val_metrics

    def save(self, file_path):
        """
        Save the model's weights to a file.
        :param file_path: Path to save the weights to
        :return:
        """
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        base, ext = os.path.splitext(file_path)
        counter = 1
        save_path = file_path

        # If the file already exists, add a counter to the filename
        while os.path.exists(save_path):
            save_path = f"{base}_{counter}{ext}"
            counter += 1

        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to {save_path}')
