import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        # 定义 RNN 层
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # 定义输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出
        return out, hidden