import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, input_dim, output_dim, n_class, n_layer, rnn_type, device):
        super(RNNModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # embedding layer
        self.emb = nn.Embedding(vocab_size, input_dim)


        # rnn layer
        assert rnn_type in ['gru', 'lstm', 'rnn']
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, output_dim, num_layers=n_layer, batch_first=True)
        elif rnn_type== 'lstm': 
            self.rnn = nn.LSTM(input_dim, output_dim, num_layers=n_layer, batch_first=True)
        else:
            self.rnn = nn.RNN(input_dim, output_dim, num_layers=n_layer, batch_first=True)


        # prediction layer
        self.predict = nn.Linear(output_dim, n_class)

    def forward(self, x, l_list):
        """
            input:
                x: input sequences
                l_list: indexes of real last elements in the sequences
            output:
                predict_result, dimension: batch x n_class
        """
        x = self.emb(x)
        n, l, d = x.shape

        if self.rnn_type=='lstm':
            output, (h_n, c_n) = self.rnn(x)
        else:
            output, h_n = self.rnn(x)

        idx = l_list.reshape(n,1,1).expand(n, 1, self.output_dim) 
        predict_result = self.predict(torch.gather(output,1,idx).squeeze(1))
        return predict_result
    

