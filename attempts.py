attempts

# history=5, hidden=256 loss=0.0113
class MultiHistoryNetwork(nn.Module):
  def __init__(self, num_features, hidden_size, output_size, num_history):
    super(MultiHistoryNetwork, self).__init__()
    self.lstm = nn.LSTM(num_features, hidden_size, num_layers=1, batch_first=True)
    self.output_features = hidden_size * num_history
    self.linear = nn.Sequential(
        nn.Linear(self.output_features, self.output_features),
        nn.ReLU(),
        nn.Linear(self.output_features, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, output_size),
        nn.Tanh()
    )

# history=5, hidden=256 loss=0.0112
class BehaviorCloningModel(nn.Module):
    def __init__(self, num_history, num_features, output_size):
        super(BehaviorCloningModel, self).__init__()
        self.flattened_size = 64 * ((num_features // 2))
        self.policy = nn.Sequential(

            nn.Conv1d(in_channels=num_history, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
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
            