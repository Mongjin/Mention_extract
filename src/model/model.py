from __future__ import print_function
import torch.nn as nn
from transformers.activations import get_activation
from transformers.modeling_electra import ElectraPreTrainedModel, ElectraModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn.functional as F
from torch.autograd import *


class ElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config, lstm_hidden, open_emb_size, close_emb_size, open_size, close_size, num_layer, bilstm_flag):
        super().__init__(config)

        assert open_emb_size == lstm_hidden * 2, "Please set score-embedding-size to twice the lstm-hidden-size"

        # 분류할 라벨의 개수
        self.num_labels = config.num_labels
        # ELECTRA 모델
        self.electra = ElectraModel(config)

        self.n_hidden = lstm_hidden

        self.open_emb = nn.Embedding(open_size, open_emb_size, scale_grad_by_freq=True)
        self.close_emb = nn.Embedding(close_size, close_emb_size, scale_grad_by_freq=True)

        self.num_layers = num_layer
        self.bidirectional = 2 if bilstm_flag else 1

        self.open_label_lstm_first = nn.LSTM(config.hidden_size, lstm_hidden, bidirectional=True, batch_first=True)
        self.open_label_lstm_last = nn.LSTM(lstm_hidden * 4, lstm_hidden, num_layers=self.num_layers,
                                       batch_first=True, bidirectional=bilstm_flag)

        self.close_label_lstm_first = nn.LSTM(config.hidden_size, lstm_hidden, bidirectional=True, batch_first=True)
        self.close_label_lstm_last = nn.LSTM(lstm_hidden * 4, lstm_hidden, num_layers=self.num_layers,
                                            batch_first=True, bidirectional=bilstm_flag)

        self.open_q_liner = nn.Linear(lstm_hidden * 2, lstm_hidden * 2)
        self.open_k_liner = nn.Linear(lstm_hidden * 2, lstm_hidden * 2)
        self.open_v_liner = nn.Linear(lstm_hidden * 2, lstm_hidden * 2)

        self.close_q_liner = nn.Linear(lstm_hidden * 2, lstm_hidden * 2)
        self.close_k_liner = nn.Linear(lstm_hidden * 2, lstm_hidden * 2)
        self.close_v_liner = nn.Linear(lstm_hidden * 2, lstm_hidden * 2)

        self.softmax = nn.Softmax(dim=-1)

        self.open_label_attn = multihead_attention(lstm_hidden * 2, num_heads=1, dropout_rate=config.hidden_dropout_prob)
        self.open_label_attn_last = multihead_attention(lstm_hidden * 2, num_heads=1, dropout_rate=0)

        self.close_label_attn = multihead_attention(lstm_hidden * 2, num_heads=1, dropout_rate=config.hidden_dropout_prob)
        self.close_label_attn_last = multihead_attention(lstm_hidden * 2, num_heads=1, dropout_rate=0)

        self.lstm_output2open = nn.Linear(lstm_hidden * 2, open_size)
        self.lstm_output2close = nn.Linear(lstm_hidden * 2, close_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, open_labels=None,
                open_label_seq_tensor=None, close_labels=None, close_label_seq_tensor=None, word_seq_lengths=None):
        discriminator_hidden_states = self.electra(input_ids, attention_mask, token_type_ids)

        # (batch_size, max_length, hidden_size)
        discriminator_hidden_states = discriminator_hidden_states[0]

        self.batch_size = discriminator_hidden_states.shape[0]
        open_embs = self.open_emb(open_label_seq_tensor)
        close_embs = self.close_emb(close_label_seq_tensor)

        hidden = None
        scaler = self.n_hidden ** 0.5

        """
        Open tag predict layer
        """
        open_lstm_outputs, hidden = self.open_label_lstm_first(discriminator_hidden_states, hidden)
        open_lstm_outputs = self.dropout(open_lstm_outputs)

        open_q = self.open_q_liner(open_lstm_outputs)
        open_k = self.open_k_liner(open_embs)
        open_v = self.open_v_liner(open_embs)

        open_attention_score = open_q.matmul(open_k.permute(0, 2, 1)) / scaler
        open_attention_align = self.softmax(open_attention_score)
        open_attention_output = open_attention_align.matmul(open_v)
        open_attention_output = self.dropout(open_attention_output)

        open_lstm_outputs = torch.cat([open_lstm_outputs, open_attention_output], dim=-1)

        open_lstm_outputs, hidden = self.open_label_lstm_last(open_lstm_outputs, hidden)
        open_lstm_outputs = self.dropout(open_lstm_outputs)

        open_q = self.open_q_liner(open_lstm_outputs)
        open_k = self.open_k_liner(open_embs)

        open_attention_score = open_q.matmul(open_k.permute(0, 2, 1)) / scaler
        open_attention_score = self.dropout(open_attention_score)

        """
        Close tag predict layer
        """
        hidden = None
        close_lstm_outputs, hidden = self.close_label_lstm_first(discriminator_hidden_states, hidden)
        close_lstm_outputs = self.dropout(close_lstm_outputs)

        close_q = self.close_q_liner(close_lstm_outputs)
        close_k = self.close_k_liner(close_embs)
        close_v = self.close_v_liner(close_embs)

        close_attention_score = close_q.matmul(close_k.permute(0, 2, 1)) / scaler
        close_attention_align = self.softmax(close_attention_score)
        close_attention_output = close_attention_align.matmul(close_v)
        close_attention_output = self.dropout(close_attention_output)

        close_lstm_outputs = torch.cat([close_lstm_outputs, close_attention_output], dim=-1)

        close_lstm_outputs, hidden = self.close_label_lstm_last(close_lstm_outputs, hidden)
        close_lstm_outputs = self.dropout(close_lstm_outputs)

        close_q = self.close_q_liner(close_lstm_outputs)
        close_k = self.close_k_liner(close_embs)

        close_attention_score = close_q.matmul(close_k.permute(0, 2, 1)) / scaler
        close_attention_score = self.dropout(close_attention_score)

        return open_attention_score.permute(0, 2, 1), close_attention_score.permute(0, 2, 1)