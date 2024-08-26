import os

import progressbar as pb
import torch


class Trainer:
    def __init__(self, model, train_loader, test_loader=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

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
        with pb.ProgressBar(max_value=num_epochs, widgets=[
            "Epoch: ", pb.Counter(), "/", str(num_epochs), " ",
            pb.Percentage(), ' ',
            pb.Timer(), ' ',
            pb.ETA(), ' ',
            pb.DynamicMessage('train_loss'),
            " ",
            pb.DynamicMessage('test_loss')
        ]) as bar:
            for epoch in range(num_epochs):
                self.model.train()
                running_loss = 0.0

                for inputs, targets in self.train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    running_loss += loss.item()

                avg_train_loss = running_loss / len(self.train_loader)
                # print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')

                avg_test_loss = 0
                if self.test_loader:
                    avg_test_loss = self.evaluate(criterion)
                    # print(f'Epoch {epoch + 1}/{num_epochs}, Test Loss: {avg_test_loss:.4f}')
                    if scheduler:
                        scheduler.step(avg_test_loss)

                bar.update(epoch + 1, train_loss=avg_train_loss, test_loss=avg_test_loss)

    def evaluate(self, criterion):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        return test_loss / len(self.test_loader)

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
