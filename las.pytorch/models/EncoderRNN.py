import math
import torch
import torch.nn as nn

import logging
import numpy as np
import torch.nn.functional as F

from .utils import make_pad_mask


logging.basicConfig(level=logging.WARNING) # level=logging.DEBUG


"""
Copied from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/rnn/encoders.py#L174
Copyright 2017 Johns Hopkins University (Shinji Watanabe)
Apache License
"""
class VGG2L(torch.nn.Module):
    """VGG-like module
    :param int in_channel: number of input channels
    """

    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        # CNN layer (VGG motivated)
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.in_channel = in_channel

    def forward(self, xs_pad, ilens, **kwargs):
        """VGG2L forward
        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of padded hidden state sequences (B, Tmax // 4, 128 * D // 4)
        :rtype: torch.Tensor
        """
        logging.debug(self.__class__.__name__ + " input lengths: " + str(ilens))

        # x: utt x frame x dim
        # xs_pad = F.pad_sequence(xs_pad)

        # x: utt x 1 (input channel num) x frame x dim
        xs_pad = xs_pad.view(
            xs_pad.size(0),
            xs_pad.size(1),
            self.in_channel,
            xs_pad.size(2) // self.in_channel,
        ).transpose(1, 2)

        # NOTE: max_pool1d ?
        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)

        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        if torch.is_tensor(ilens):
            ilens = ilens.cpu().numpy()
        else:
            ilens = np.array(ilens, dtype=np.float32)
        ilens = np.array(np.ceil(ilens / 2), dtype=np.int64)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64
        ).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3)
        )
        return xs_pad, ilens, None  # no state in this layer


"""
Copied from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py#L14
Copyright 2017 Johns Hopkins University (Shinji Watanabe)
Apache License
"""
class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    :param torch.nn.Module pos_enc: custom position encoding layer
    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU()
        )
#         self.out = torch.nn.Sequential(
#             torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim) # ,
#             # pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
#         )

    def forward(self, x, x_mask):
        """Subsample x.
        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        # x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Subsample x.
        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.
        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths
    

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1,
                 input_dropout_p=0, dropout_p=0,
                 bidirectional=False, rnn_cell='gru', variable_lengths=False):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.variable_lengths = variable_lengths

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))
        
        """
        Copied from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
        Copyright (c) 2017 Sean Naren
        MIT License
        """
        outputs_channel = 32
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, outputs_channel, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(outputs_channel),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(outputs_channel, outputs_channel, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(outputs_channel),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        rnn_input_dims = int(math.floor(input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_dims = int(math.floor(rnn_input_dims + 2 * 10 - 21) / 2 + 1)
        rnn_input_dims *= outputs_channel
        
        

#         outputs_channel = 256
#         rnn_input_dims = ((input_size - 1) // 2 - 1) // 2
#         rnn_input_dims *= outputs_channel
#         print(f"[Conv2dSubsampling]: rnn_input_dims: {rnn_input_dims}")
        
        # outputs_channel = 128
        # rnn_input_dims = (input_size) // 4
        # rnn_input_dims = ((input_size - 1) // 2 - 1) // 2
        # rnn_input_dims *= outputs_channel
        # rnn_input_dims = 5248
        print(f"[VGG2L]: rnn_input_dims: {rnn_input_dims}")
        
        # self.conv = Conv2dSubsampling(idim=input_size, odim=outputs_channel, dropout_rate=input_dropout_p)
        # self.conv = VGG2L(in_channel=1)

        self.rnn =  self.rnn_cell(rnn_input_dims, self.hidden_size, self.n_layers, batch_first=True, dropout=self.dropout_p, bidirectional=self.bidirectional)
        
        # self.last = nn.Linear(2*hidden_size if self.bidirectional else hidden_size, hidden_size)

    def forward(self, input_var, input_lengths=None):
        """
        param:input_var: Encoder inputs, Spectrogram, Shape=(B,1,D,T)
        param:input_lengths: inputs sequence length without zero-pad
        """
        
        output_lengths = self.get_seq_lens(input_lengths)

        x = input_var # (B,1,D,T)
        
        x, _ = self.conv(x, output_lengths) # (B, C, D, T)
        x_size = x.size()
        x = x.view(x_size[0], x_size[1] * x_size[2], x_size[3]) # (B, C * D, T)
        x = x.permute(0, 2, 1).contiguous() # (B,T,D)

        total_length = x_size[3]
        
        #x = x.squeeze(1).permute(0, 2, 1).contiguous() # (B, T, D)
        #x, _ = self.conv(x, x_mask=None) # |x| = (B, T, D) for Conv2dSubsampling
        # x, output_lengths, _ = self.conv(xs_pad=x, ilens=input_lengths) # |x| = (B, T, D) for VGG2L
        
        # total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              output_lengths,
                                              batch_first=True,
                                              enforce_sorted=False)
        x, h_state = self.rnn(x)
        x, x_lengths = nn.utils.rnn.pad_packed_sequence(x,
                                                batch_first=True,
                                                total_length=total_length)
        
        # x = torch.tanh(self.last(x))
        # x = x.masked_fill(make_pad_mask(x_lengths, x, 1), 0.0)
        
        return x, output_lengths, h_state
    

    def get_seq_lens(self, input_length):
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d :
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)

        return seq_len.int()
