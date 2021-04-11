import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=128,
        p_dropout=1/2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size,
                          hidden_size,
                          batch_first=True)
        self.dropout = nn.Dropout(p=p_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, input_seq):
        _, hidden_state = self.rnn(input_seq)
        output = self.dropout(hidden_state)
        predictions = self.classifier(output.squeeze(0))
        return predictions


class SimpleGRU(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=128,
        p_dropout=1/2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size,
                          hidden_size,
                          batch_first=True)
        self.dropout = nn.Dropout(p_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, input_seq):
        _, hidden_state = self.rnn(input_seq)
        predictions = self.classifier(hidden_state.squeeze(0))
        return predictions


class SimpleLSTM(nn.Module):
    """
    Just simple LSTM model
    """
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=128,
        p_dropout=1/2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           batch_first=True)
        self.dropout = nn.Dropout(p_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, input_seq):
        _, (hidden_state, _) = self.rnn(input_seq)
        hidden_state = self.dropout(hidden_state)
        predictions = self.classifier(hidden_state.squeeze(0))
        return predictions

class AttentionModel(nn.Module):
    """
    DOT attention with LSTM
    """
    def __init__(self, input_size, output_size, hidden_size, p_dropout):
        super(AttentionModel, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.p_dropout = p_dropout

        self.rnn = nn.LSTM(input_size, hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Linear(64, output_size)
        )
    
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

    def forward(self, input):
        input = input.permute(1, 0, 2)
        output, (final_hidden_state, final_cell_state) = self.rnn(input)
        output = output.permute(1, 0, 2)
        context = self.attention_net(output, final_hidden_state)
        general_vector = torch.cat((context, final_hidden_state.squeeze(0)), dim=1)
        logits = self.classifier(general_vector)

        return logits


class GRUFCNModel(nn.Module):
    """
    TODO
    """
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.rnn = nn.Sequential(nn.GRU(self.input_size, self.hidden_size, batch_first =True))
        nn.init.xavier_uniform_(self.rnn[0].weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn[0].weight_ih_l0)

        self.dropout = nn.Dropout(p=0.8)
        self.conv = nn.Sequential(
            nn.Conv1d(self.input_size, 128, 8),
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


        self.fc = nn.Linear(self.hidden_size + 128*167, self.output_size)

    def forward(self, x):
        # x : [batch_size, seq_len, features]
        _, rnn_hidden_state = self.rnn(x)
        rnn_hidden_state = self.dropout(rnn_hidden_state)
        x = x.transpose(1, 2)
        conv_output = self.conv(x)
        x = torch.cat((rnn_hidden_state.squeeze(0), torch.flatten(conv_output, 1)), dim=1)
        output = self.fc(x)

        return output