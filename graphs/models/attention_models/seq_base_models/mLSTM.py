import torch.nn as nn
"""
LSTM to insert the different output merging methods
"""
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, merge_func = lambda x: x, num_layers = 2, bidirectional= False):
        super().__init__()

        # LSTM approach
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.merge_func = merge_func

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        return self.merge_func(lstm_out)
