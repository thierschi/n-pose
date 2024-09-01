import os

import torch


class Trainer:
    def __init__(self, model, train_loader, test_loader=None, metrics=None, callback=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.metrics = metrics if metrics is not None else []
        self.callback = callback

        self.device = torch.device('cpu')
        self.model.to(self.device)

    def use_device(self, device):
        self.device = device
        self.model.to(self.device)

    def use_best_device(self):
        if torch.cuda.is_available():
            self.use_device(torch.device('cuda'))
        elif torch.backends.mps.is_available():
            self.use_device(torch.device('mps'))
        else:
            self.use_device(torch.device('cpu'))

    def train(self, num_epochs, criterion, optimizer, scheduler=None):
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
                self.callback([avg_train_loss, avg_test_loss] + train_metrics + val_metrics)

    def evaluate(self, criterion):
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

    def save(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        base, ext = os.path.splitext(file_path)
        counter = 1
        save_path = file_path

        while os.path.exists(save_path):
            save_path = f"{base}_{counter}{ext}"
            counter += 1

        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to {save_path}')
