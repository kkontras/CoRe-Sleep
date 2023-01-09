import torch
import torch.functional as F
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, hidden=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        # Convert word indexes to embeddings
        # embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        # packed = torch.nn.utils.rnn.pack_padded_sequence(input_seq)
        # Forward pass through GRU
        outputs, hidden = self.gru(input_seq, hidden)
        # Unpack padding
        # outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        self.softmax = nn.Softmax(dim=1)

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return self.softmax(attn_energies).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, input_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        # self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        # embedded = self.embedding(input_step)
        # embedded = self.embedding_dropout(input_step)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(input_step, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = self.softmax(output)
        # Return output and final hidden state
        return output, hidden

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, fc_size, num_classes, decoder_n_layers):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._SOS_token = torch.rand(5).cuda()
        self._decoder_n_layers = decoder_n_layers
        fc_inner = 32
        self.fc_out = nn.Sequential(
                        nn.BatchNorm1d(fc_size),
                        nn.Dropout(0.45),
                        nn.Linear(fc_size, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

    __constants__ = ['_device', '_SOS_token', '_decoder_n_layers']

    def forward(self, input_seq : torch.Tensor):
        input_seq = input_seq.permute(1,0,2)
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self._decoder_n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = (torch.ones(1, 1, 1, dtype=torch.long).cuda() * self._SOS_token).repeat(1, input_seq.shape[1], 1)
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], dtype=torch.long).cuda()
        # all_scores = torch.zeros([0], device=self._device)
        # Iteratively decode one word token at a time
        max_length = encoder_outputs.shape[0]
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            pseudo_target = self.fc_out(decoder_output)
            # Obtain most likely word token and its softmax score
            # decoder_scores, decoder_input = torch.max(pseudo_target, dim=1)

            # print(decoder_scores)
            # print(decoder_input)
            # print(pseudo_target)

            decoder_input = pseudo_target
            # Record token and score
            all_tokens = torch.cat((all_tokens, pseudo_target.unsqueeze(dim=0)), dim=0)
            # all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens.permute(1,0,2)

class seq2seq_GRU(nn.Module):
    def __init__(self, dmodel, num_classes):
        super().__init__()
        encoder = EncoderRNN(dmodel)
        decoder = LuongAttnDecoderRNN('dot', input_size = 5, hidden_size=dmodel, output_size=dmodel)
        self.model = GreedySearchDecoder(encoder, decoder, dmodel, num_classes, 1)

    def forward(self,input_seq):
        return self.model(input_seq)