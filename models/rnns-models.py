import torch
import torch.nn as nn


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