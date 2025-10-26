import torch

#used for the resnet model
class ModelHead(torch.nn.Module):
    def __init__(self, input_dimension, hidden_dimension, num_classes):
        super(ModelHead, self).__init__()
        self.fc1 = torch.nn.Linear(input_dimension, hidden_dimension)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dimension, hidden_dimension // 2)
        self.fc3 = torch.nn.Linear(hidden_dimension // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x