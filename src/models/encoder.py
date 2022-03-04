import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.neural import MultiHeadedAttention, PositionwiseFeedForward
from torch_geometric.nn import GCNConv, GATConv
INFINITY_NUMBER = 1e12
SMALL_NUMBER = 1e-12
class BasicAttention(nn.Module):
    """
    A basic MLP attention layer.
    """

    def __init__(self, dim):
        super(BasicAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_c = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=1)

    #attention_layer(hy, ctx, mask=ctx_mask)

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        target = self.linear_in(input)  # batch x dim
        source = self.linear_c(context.contiguous().view(-1, dim)).view(batch_size, source_len, dim)
        attn = target.unsqueeze(1).expand_as(context) + source
        attn = self.tanh(attn)  # batch x sourceL x dim
        attn = self.linear_v(attn.view(-1, dim)).view(batch_size, source_len)

        if mask is not None:
            attn.masked_fill_(mask, -INFINITY_NUMBER)

        attn = self.sm(attn)
        if attn_only:
            return attn

        weighted_context = torch.bmm(attn.unsqueeze(1), context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), dim=1)
        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, weighted_context, attn

class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~ mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores


class GCN(torch.nn.Module):
    def __init__(self,
                 num_features,
                 hidden_size,
                 dropout=0.5,
                 activation="relu"):
        super(GCN, self).__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size

        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, 'relu')

    def forward(self, xs, edge_indexs, src=None):

        out = []
        for i in range(xs.shape[0]):

            x = xs[i]
            edge_index = edge_indexs[i]
            x = self.activation(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout)
            x = self.conv2(x, edge_index)
            x = x.unsqueeze(dim=0)
            out.append(x)

        out_x = torch.cat(out, dim=0)


        return out_x


class GAT(torch.nn.Module):
    def __init__(self,
                 num_features,
                 hidden_size,
                 dropout=0.5,
                 activation="relu",
                 num_heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(
            num_features, hidden_size, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(
            hidden_size * num_heads, hidden_size, dropout=dropout)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, 'relu')

    def forward(self, xs, edge_indexs, src=None):

        out = []
        for i in range(xs.shape[0]):

            x = xs[i]
            edge_index = edge_indexs[i]
            try:
                x = self.activation(self.conv1(x, edge_index))
                x = F.dropout(x, p=self.dropout)
                x = self.conv2(x, edge_index)

                x = x.unsqueeze(dim=0)
                out.append(x)
            except:
                import pdb
                pdb.set_trace()
        out_x = torch.cat(out, dim=0)

        return out_x
