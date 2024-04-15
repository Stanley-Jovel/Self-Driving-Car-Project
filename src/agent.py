import torch.nn as nn
from enum import Enum

class BehaviorCloningModel(nn.Module):
    def __init__(self, num_history, num_features, output_size):
        super(BehaviorCloningModel, self).__init__()
        self.flattened_size = 64 * (num_features // 4)
        self.policy = nn.Sequential(
            nn.Conv1d(in_channels=num_history, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(32),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(64)
        )
        self.classifier = nn.Sequential(
            # Calculate the size after convolution and pooling
            nn.Linear(self.flattened_size, self.flattened_size),
            nn.ReLU(),
            nn.Linear(self.flattened_size, self.flattened_size),
            nn.ReLU(),
            nn.Linear(self.flattened_size, 128),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.policy(x)
        x = x.view(-1, self.flattened_size)  # Flatten the tensor for the fully connected layer
        x = self.classifier(x)
        return x

class Constants(Enum):
    INPUT_SIZE = 25  # Number of features in observation
    HIDDEN_SIZE = 128  # Number of units in hidden layer
    NUM_HISTORY = 10  # Number of history steps to use
    OUTPUT_SIZE = 2  # Number of actions
    DROPOUT = 0.25  # Dropout rate
    lr = 1e-3  # Learning rate
    EPOCHS = 20  # Number of epochs to train

    NUM_LAYERS = 4  # Number of LSTM layers