import torch
import torch.nn as nn
import torch.nn.functional as F

import math


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
            nn.Linear(hidden_size, output_size)
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
            nn.Linear(hidden_size, output_size)
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
            nn.Linear(hidden_size, output_size)
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
            nn.Linear(2 * self.hidden_size, output_size)
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
    def __init__(self, input_size, output_size, hidden_size, p_dropout):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.p_dropout = p_dropout

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first =True)
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)

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
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size + 128, 1024),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        # x : [batch_size, seq_len, features]
        _, (rnn_hidden_state, _) = self.rnn(x)
        rnn_hidden_state = self.dropout(rnn_hidden_state)
        x = x.transpose(1, 2)
        # conv_output = [batch_size, 128, ..]
        conv_output = self.conv(x)
        conv_output = F.adaptive_max_pool1d(conv_output, 1)
        x = torch.cat((rnn_hidden_state.squeeze(0), torch.flatten(conv_output, 1)), dim=1)
        output = self.classifier(x)

        return output



class CnnAttentionModel(nn.Module):
    """
    DOT attention with LSTM
    """
    def __init__(self, input_size, output_size, hidden_size, p_dropout):
        super(CnnAttentionModel, self).__init__()

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

        self.conv1d = nn.Conv1d(hidden_size, hidden_size, 16)
    
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        processed_output = self.conv1d(lstm_output.transpose(1, 2))
        attn_weights = torch.bmm(processed_output.transpose(1, 2), hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(processed_output, soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

    def forward(self, input):
        input = input.permute(1, 0, 2)
        output, (final_hidden_state, final_cell_state) = self.rnn(input)
        output = output.permute(1, 0, 2)
        context = self.attention_net(output, final_hidden_state)
        general_vector = torch.cat((context, final_hidden_state.squeeze(0)), dim=1)
        logits = self.classifier(general_vector)

        return logits


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)
        
    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]
        # Here we assume q_dim == k_dim (dot product attention)
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

        values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return linear_combination


class RnnSelfAttention(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, p_dropout):
        super(RnnSelfAttention, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=False)
        self.attention = Attention(hidden_size, hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, output_size)
        self.conv1d = nn.Conv1d(hidden_size, hidden_size, 16)
        self.self_attention = nn.MultiheadAttention(hidden_size, 1)
        
        
    def forward(self, input):
        input = input.permute(1, 0, 2)
        outputs, hidden = self.rnn(input)
        if isinstance(hidden, tuple): # LSTM
            hidden = hidden[1] # take the cell state
        if self.rnn.bidirectional: # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        # ================================

        attn_output, _ = self.self_attention(hidden.unsqueeze(0), outputs.transpose(0, 1), outputs.transpose(0, 1))


        # ================================
        '''
        #outputs = self.conv1d(outputs.permute(1, 0, 2).transpose(1, 2)).permute(1, 0, 2)
        linear_combination = self.attention(hidden, outputs, outputs) 
        logits = self.classifier(linear_combination)
        '''

        logits = self.classifier(attn_output).squeeze(0)
        return logits