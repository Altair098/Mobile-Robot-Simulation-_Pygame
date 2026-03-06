import torch
import torch.nn as nn


class RobotDynamicsLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=3):
        """
        LSTM model for predicting next robot state.
        Input: [x, y, theta, v, omega]
        Output: [x_next, y_next, theta_next]
        """
        super(RobotDynamicsLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.hidden = None  # (h_0, c_0) will be set during runtime

    def forward(self, x):
        """
        x: Tensor of shape (batch_size=1, seq_len=1, input_size)
        returns: predicted next state and updated hidden state
        """
        out, self.hidden = self.lstm(x, self.hidden)
        pred = self.fc(out[:, -1, :])  # take last output
        return pred

    def reset_hidden(self):
        """Reset hidden and cell state to zeros"""
        self.hidden = (torch.zeros(1, 1, self.hidden_size),
                       torch.zeros(1, 1, self.hidden_size))

    def detach_hidden(self):
        """Detaches hidden state from current computation graph to avoid backprop through history"""
        if self.hidden is not None:
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
