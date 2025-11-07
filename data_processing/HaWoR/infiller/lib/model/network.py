import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from cmib.model.positional_encoding import PositionalEmbedding

class SinPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(SinPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout=0.1,
                 pre_lnorm=True, bias=False):
        """
        Multi-headed attention with relative positional encoding and
        memory mechanism.

        Args:
            n_head (int): Number of heads.
            d_model (int): Input dimension.
            d_head (int): Head dimension.
            dropout (float, optional): Dropout value. Defaults to 0.1.
            pre_lnorm (bool, optional):
                Apply layer norm before rest of calculation. Defaults to True.
                In original Transformer paper (pre_lnorm=False):
                    LayerNorm(x + Sublayer(x))
                In tensor2tensor implementation (pre_lnorm=True):
                    x + Sublayer(LayerNorm(x))
            bias (bool, optional):
                Add bias to q, k, v and output projections. Defaults to False.

        """
        super(MultiHeadedAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm
        self.bias = bias
        self.atten_scale = 1 / math.sqrt(self.d_model)

        self.q_linear = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.k_linear = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.v_linear = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.out_linear = nn.Linear(n_head * d_head, d_model, bias=bias)

        self.droput_layer = nn.Dropout(dropout)
        self.atten_dropout_layer = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, hidden, memory=None, mask=None,
                extra_atten_score=None):
        """
        Args:
            hidden (Tensor): Input embedding or hidden state of previous layer.
                Shape: (batch, seq, dim)
            pos_emb (Tensor): Relative positional embedding lookup table.
                Shape: (batch, (seq+mem_len)*2-1, d_head)
                pos_emb[:, seq+mem_len]

            memory (Tensor): Memory tensor of previous layer.
                Shape: (batch, mem_len, dim)
            mask (BoolTensor, optional): Attention mask.
                Set item value to True if you DO NOT want keep certain
                attention score, otherwise False. Defaults to None.
                Shape: (seq, seq+mem_len).
        """
        combined = hidden
        # if memory is None:
        #     combined = hidden
        #     mem_len = 0
        # else:
        #     combined = torch.cat([memory, hidden], dim=1)
        #     mem_len = memory.shape[1]

        if self.pre_lnorm:
            hidden = self.layer_norm(hidden)
            combined = self.layer_norm(combined)

        # shape: (batch, q/k/v_len, dim)
        q = self.q_linear(hidden)
        k = self.k_linear(combined)
        v = self.v_linear(combined)

        # reshape to (batch, q/k/v_len, n_head, d_head)
        q = q.reshape(q.shape[0], q.shape[1], self.n_head, self.d_head)
        k = k.reshape(k.shape[0], k.shape[1], self.n_head, self.d_head)
        v = v.reshape(v.shape[0], v.shape[1], self.n_head, self.d_head)

        # transpose to (batch, n_head, q/k/v_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # add n_head dimension for relative positional embedding lookup table
        # (batch, n_head, k/v_len*2-1, d_head)
        # pos_emb = pos_emb[:, None]

        # (batch, n_head, q_len, k_len)
        atten_score = torch.matmul(q, k.transpose(-1, -2))

        # qpos = torch.matmul(q, pos_emb.transpose(-1, -2))
        # DEBUG
        # ones = torch.zeros(q.shape)
        # ones[:, :, :, 0] = 1.0
        # qpos = torch.matmul(ones, pos_emb.transpose(-1, -2))
        # atten_score = atten_score + self.skew(qpos, mem_len)
        atten_score = atten_score * self.atten_scale

        # if extra_atten_score is not None:
        #     atten_score = atten_score + extra_atten_score

        if mask is not None:
            # print(atten_score.shape)
            # print(mask.shape)
            # apply attention mask
            atten_score = atten_score.masked_fill(mask, float("-inf"))
        atten_score = atten_score.softmax(dim=-1)
        atten_score = self.atten_dropout_layer(atten_score)

        # (batch, n_head, q_len, d_head)
        atten_vec = torch.matmul(atten_score, v)
        # (batch, q_len, n_head*d_head)
        atten_vec = atten_vec.transpose(1, 2).flatten(start_dim=-2)

        # linear projection
        output = self.droput_layer(self.out_linear(atten_vec))

        if self.pre_lnorm:
            return hidden + output
        else:
            return self.layer_norm(hidden + output)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1, pre_lnorm=True):
        """
        Positionwise feed-forward network.

        Args:
            d_model(int): Dimension of the input and output.
            d_inner (int): Dimension of the middle layer(bottleneck).
            dropout (float, optional): Dropout value. Defaults to 0.1.
            pre_lnorm (bool, optional):
                Apply layer norm before rest of calculation. Defaults to True.
                In original Transformer paper (pre_lnorm=False):
                    LayerNorm(x + Sublayer(x))
                In tensor2tensor implementation (pre_lnorm=True):
                    x + Sublayer(LayerNorm(x))
        """
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm

        self.layer_norm = nn.LayerNorm(d_model)
        self.network = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        if self.pre_lnorm:
            return x + self.network(self.layer_norm(x))
        else:
            return self.layer_norm(x + self.network(x))
class TransformerModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        out_dim=91,
        masked_attention_stage=False,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.pos_embedding = SinPositionalEncoding(d_model=d_model, dropout=0.1, max_len=seq_len)
        if masked_attention_stage:
            self.input_layer = nn.Linear(input_dim+1, d_model)
            # visible to invisible attention
            self.att_layers = nn.ModuleList()
            self.pff_layers = nn.ModuleList()
            self.pre_lnorm = True
            self.layer_norm = nn.LayerNorm(d_model)
            for i in range(self.nlayers):
                self.att_layers.append(
                    MultiHeadedAttention(
                        self.nhead, self.d_model,
                        self.d_model // self.nhead, dropout=dropout,
                        pre_lnorm=True,
                        bias=False
                    )
                )

                self.pff_layers.append(
                    FeedForward(
                        self.d_model, d_hid,
                        dropout=dropout,
                        pre_lnorm=True
                    )
                )
        else:
            self.att_layers = None
            self.input_layer = nn.Linear(input_dim, d_model)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, activation="gelu"
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(d_model, out_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor, data_mask=None, atten_mask=None) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, embedding_dim]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        if not data_mask is None:
            src = torch.cat([src, data_mask.expand(*src.shape[:-1], data_mask.shape[-1])], dim=-1)
        src = self.input_layer(src)
        output = self.pos_embedding(src)
        # output = src
        if self.att_layers:
            assert not atten_mask is None
            output = output.permute(1, 0, 2)
            for i in range(self.nlayers):
                output = self.att_layers[i](output, mask=atten_mask)
                output = self.pff_layers[i](output)
            if self.pre_lnorm:
                output = self.layer_norm(output)
            output = output.permute(1, 0, 2)
        output = self.transformer_encoder(output)
        output = self.decoder(output)
        return output
