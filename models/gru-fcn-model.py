import torch
import torch.nn as nn


class GRUFCNModel(nn.Module):
    """
    TODO
    """
    def __init__(self):
        super().__init__()
        self.rnn = nn.Sequential(
            nn.GRU(18, #
            8, #hidden_size
            batch_first=True)
        )
        nn.init.xavier_uniform(self.rnn[0].weight_hh_l0)
        nn.init.xavier_uniform(self.rnn[0].weight_ih_l0)

        self.dropout = nn.Dropout(p=0.8)
        self.conv = nn.Sequential(
            nn.Conv1d(18, 128, 8),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, 5),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 128, 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        nn.init.kaiming_uniform_(self.conv[0].weight)
        nn.init.kaiming_uniform_(self.conv[3].weight)
        nn.init.kaiming_uniform_(self.conv[6].weight)


        self.fc = nn.Linear(8 + 128*147, 1)

    def forward(self, x):
        # x : [batch_size, seq_len, features]
        _, rnn_hidden_state = self.rnn(x)
        rnn_hidden_state = self.dropout(rnn_hidden_state)

        x = x.transpose(1, 2)
        conv_output = self.conv(x)

        x = torch.cat((rnn_hidden_state.squeeze(0), torch.flatten(conv_output, 1)), dim=1)
        output = self.fc(x)

        return output
