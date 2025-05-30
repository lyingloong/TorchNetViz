import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(RevIN, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.mean = None
        self.std = None

    def forward(self, x, mode="norm"):
        if mode == "norm":
            self.mean = x.mean(dim=(1, 2), keepdim=True)
            self.std = x.std(dim=(1, 2), keepdim=True) + self.eps
            return (x - self.mean) / self.std
        elif mode == "denorm":
            return x * self.std + self.mean
        else:
            raise ValueError("Mode must be 'norm' or 'denorm'")


class DoubleCNN(nn.Module):
    """
    DoubleCNN model for time series forecasting.
    This model uses two CNN layers to capture time dependencies and variable dependencies respectively.
    """

    def __init__(self, configs):
        super(DoubleCNN, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.revin = RevIN(configs.enc_in)

        # Time dimension CNN
        self.time_cnn = nn.Conv1d(
            in_channels=self.enc_in,
            out_channels=self.enc_in,
            kernel_size=configs.time_kernel_size,
            padding=configs.time_kernel_size//2,
            stride=1
        )
        self.time_cnn_GELU = nn.GELU()

        # Variable dimension CNN
        self.variable_cnn = nn.Conv1d(
            in_channels=self.seq_len,
            out_channels=self.seq_len,
            kernel_size=configs.variable_kernel_size,
            padding=configs.variable_kernel_size//2,
            stride=1
        )
        self.variable_cnn_GELU = nn.GELU()

        self.mlp = nn.Sequential(
            nn.Linear(self.seq_len * self.enc_in * 2, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, self.pred_len * self.enc_in)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x_enc: batch * seq_len * enc_in
        """
        # Norm
        x_enc = self.revin(x_enc.permute(0, 2, 1))

        # Time dimension convolution
        time_cnn_out = self.time_cnn_GELU(self.time_cnn(x_enc))

        # Variable dimension convolution
        variable_cnn_out = self.variable_cnn_GELU(self.variable_cnn(x_enc.permute(0, 2, 1)))

        combined_out = torch.cat((variable_cnn_out.view(variable_cnn_out.size(0), -1), time_cnn_out.view(time_cnn_out.size(0), -1)), dim=-1)
        out = self.mlp(combined_out)
        out = out.view(out.size(0), self.enc_in, self.pred_len)

        # Denorm
        out = self.revin(out, mode='denorm').permute(0, 2, 1)

        return out


class DoubleCNNLayer(nn.Module):
    """
    DoubleCNN layer for time series forecasting.
    This layer uses two CNN layers to capture time dependencies and variable dependencies respectively.
    """

    def __init__(self, enc_in, seq_len, out_len,
                 time_kernel_size, variable_kernel_size):
        super(DoubleCNNLayer, self).__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.out_len = out_len

        # Time dimension CNN
        self.time_cnn = nn.Conv1d(in_channels=self.enc_in,
                                  out_channels=self.enc_in,
                                  kernel_size=time_kernel_size,
                                  padding=time_kernel_size//2,
                                  stride=1)
        self.time_cnn_GELU = nn.GELU()

        # Variable dimension CNN
        self.variable_cnn = nn.Conv1d(in_channels=self.seq_len,
                                      out_channels=self.seq_len,
                                      kernel_size=variable_kernel_size,
                                      padding=variable_kernel_size//2,
                                      stride=1)
        self.variable_cnn_GELU = nn.GELU()

        self.mlp = nn.Sequential(nn.Linear(self.seq_len * self.enc_in * 2, 512),
                                 nn.GELU(),
                                 nn.Linear(512, 256),
                                 nn.GELU(),
                                 nn.Linear(256, self.out_len * self.enc_in))

    def forward(self, x_enc):
        """
        x_enc: batch * seq_len * enc_in
        """
        x_enc = x_enc.permute(0, 2, 1)

        # Time dimension convolution
        time_cnn_out = self.time_cnn_GELU(self.time_cnn(x_enc))

        # Variable dimension convolution
        variable_cnn_out = self.variable_cnn_GELU(self.variable_cnn(x_enc.permute(0, 2, 1)))

        combined_out = torch.cat((variable_cnn_out.view(variable_cnn_out.size(0), -1), time_cnn_out.view(time_cnn_out.size(0), -1)), dim=-1)
        out = self.mlp(combined_out)
        out = out.view(out.size(0), self.out_len, self.enc_in)

        return out