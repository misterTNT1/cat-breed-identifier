import torch


class Model(torch.nn.Module):
    def __init__(self, input_dimension, hidden_dimension, num_classes):
        super(Model, self).__init__()
        # convolution
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        # classifier

        self.connected_layer1 = torch.nn.Linear(input_dimension, hidden_dimension)
        self.connected_layer2 = torch.nn.Linear(hidden_dimension, hidden_dimension // 2)
        self.connected_layer3 = torch.nn.Linear(hidden_dimension // 2, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = self.connected_layer1(x)
        x = self.relu(x)
        x = self.connected_layer2(x)
        x = self.relu(x)
        x = self.connected_layer3(x)
        return x
