#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import csv
import time
from typing import Optional, Tuple, List, Dict, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from torch.nn.modules.module import Module
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.modules.container import ModuleList

torch.manual_seed(0)
np.random.seed(0)


class Data(object):
    def __init__(self, players: torch.Tensor, rusher: torch.Tensor, meta: torch.Tensor, y: Optional[torch.Tensor],
                 yardLine: np.ndarray, year: np.ndarray, player_cols: List[str], rusher_cols: List[str],
                 meta_cols: List[str]):
        self.players = players
        self.rusher = rusher
        self.meta = meta
        self.y = y
        self.yardLine = yardLine
        self.year = year
        self.player_cols = player_cols
        self.rusher_cols = rusher_cols
        self.meta_cols = meta_cols

        assert self.players.size(0) == self.rusher.size(0)
        if yardLine is not None:
            assert len(yardLine) == self.players.size(0)

    def len(self):
        return self.players.size(0)

    def y_soft(self, sigma: float = 1.0):
        from scipy.ndimage.filters import gaussian_filter1d
        return torch.from_numpy(gaussian_filter1d(self.y.numpy(), sigma=sigma))

    def slice(self, begin: int, end: int) -> 'Data':
        p = self.players[begin:end] if self.players is not None else None
        r = self.rusher[begin:end] if self.rusher is not None else None
        m = self.meta[begin:end] if self.meta is not None else None
        y = self.y[begin:end] if self.y is not None else None
        yd = self.yardLine[begin:end].copy() if self.yardLine is not None else None
        yr = self.year[begin:end].copy() if self.year is not None else None

        return Data(p, r, m, y, yd, yr, self.player_cols, self.rusher_cols, self.meta_cols)

    def _sample_by_mask(self, mask):
        mask_tensor = torch.from_numpy(mask)
        p = self.players[mask_tensor] if self.players is not None else None
        r = self.rusher[mask_tensor] if self.rusher is not None else None
        m = self.meta[mask_tensor] if self.meta is not None else None
        y = self.y[mask_tensor] if self.y is not None else None
        yd = self.yardLine[mask].copy() if self.yardLine is not None else None
        yr = self.year[mask].copy() if self.year is not None else None

        return Data(p, r, m, y, yd, yr, self.player_cols, self.rusher_cols, self.meta_cols)

    def downsample_2017(self, dropout_rate: float):
        assert 0 <= dropout_rate <= 1.0
        dropout = np.random.choice([True, False], size=len(self.year), p=[1 - dropout_rate, dropout_rate])
        mask = (self.year != 2017) | dropout
        return self._sample_by_mask(mask)

    def shuffled(self):
        indices = np.random.permutation(self.players.shape[0])
        p = np.take(self.players, indices, axis=0)
        r = np.take(self.rusher, indices, axis=0) if self.rusher is not None else None
        m = np.take(self.meta, indices, axis=0) if self.meta is not None else None
        y = np.take(self.y, indices, axis=0) if self.y is not None else None
        yd = np.take(self.yardLine, indices, axis=0).copy() if self.yardLine is not None else None
        yr = np.take(self.year, indices, axis=0).copy() if self.year is not None else None
        return Data(p, r, m, y, yd, yr, self.player_cols, self.rusher_cols, self.meta_cols)

    def random_split(self, p: float):
        mask = np.random.choice([True, False], p=[p, 1 - p], size=self.meta.size(0))
        d1 = self._sample_by_mask(mask)
        d2 = self._sample_by_mask(~mask)
        return d1, d2


def concat_dataset(l: Data, r: Data):
    p = torch.cat((l.players, r.players))
    rs = torch.cat((l.rusher, r.rusher))
    m = torch.cat((l.meta, r.meta))
    y = torch.cat((l.y, r.y)) if l.y is not None else None
    yd = np.concatenate([l.yardLine, r.yardLine]) if l.yardLine is not None else None
    yr = np.concatenate([l.year, r.year]) if l.year is not None else None
    print(p.size())
    print(yd.shape)
    return Data(p, rs, m, y, yd, yr, l.player_cols, l.rusher_cols, l.meta_cols)


#######################################################################################################################
## Transformer Building Blocks (backported from PyTorch v1.3)
#######################################################################################################################


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiheadAttention(Module):
    # __annotations__ = {
    #    'bias_k': torch._jit_internal.Optional[torch.Tensor],
    #    'bias_v': torch._jit_internal.Optional[torch.Tensor],
    # }
    __constants__ = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


class TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, dropout_attn=0.1, pre_LN=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout_attn)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.relu
        self.pre_LN = pre_LN

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if self.pre_LN:
            src2 = self.norm1(src)
            src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
        else:
            src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def multi_head_attention_forward(query,  # type: torch.Tensor
                                 key,  # type: torch.Tensor
                                 value,  # type: torch.Tensor
                                 embed_dim_to_check,  # type: int
                                 num_heads,  # type: int
                                 in_proj_weight,  # type: torch.Tensor
                                 in_proj_bias,  # type: torch.Tensor
                                 bias_k,  # type: Optional[torch.Tensor]
                                 bias_v,  # type: Optional[torch.Tensor]
                                 add_zero_attn,  # type: bool
                                 dropout_p,  # type: float
                                 out_proj_weight,  # type: torch.Tensor
                                 out_proj_bias,  # type: torch.Tensor
                                 training=True,  # type: bool
                                 key_padding_mask=None,  # type: Optional[torch.Tensor]
                                 need_weights=True,  # type: bool
                                 attn_mask=None,  # type: Optional[torch.Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,  # type: Optional[torch.Tensor]
                                 k_proj_weight=None,  # type: Optional[torch.Tensor]
                                 v_proj_weight=None,  # type: Optional[torch.Tensor]
                                 static_k=None,  # type: Optional[torch.Tensor]
                                 static_v=None  # type: Optional[torch.Tensor]
                                 ):
    # type: (...) -> Tuple[torch.Tensor, Optional[torch.Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.zeros((attn_mask.size(0), 1),
                                                   dtype=attn_mask.dtype,
                                                   device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


#######################################################################################################################
# The Model
#######################################################################################################################


class TransformerModel(nn.Module):
    def __init__(self, ninp: int, nemb: int = 1, nhead: int = 1, nhid: int = 32, nlayers: int = 4, nfinal: int = 1024,
                 dropout_encoder: float = 0.1, dropout_embed: float = 0.0, dropout_classifier: float = 0.0,
                 n_class: int = 199, ninp_rusher: int = 16, pre_LN: bool = False, rusher_emb: int = 32,
                 n_emb_layers: int = 2,
                 ninp_meta: int = 8, meta_emb: int = 32, gauss_noise: float = 0.0,
                 gauss_xy_noise: float = 0.0,
                 n_fin_layers: int = 3, dropout_attn: float = 0):
        """

        :param ninp: 入力の次元数（選手一人あたりの特徴量次元
        :param nemb: Embedding層の次元
        :param nhead: multi-head attentionのheadの数
        :param nhid: transformerの中のFeedForward(FFN)の隠れ層の次元
        :param nlayers: transformer-encoderの層の数
        :param nfinal: readout後のLinear層の次元
        :param dropout_encoder: Self-Attention, Encoder内のdropout
        :param dropout_embed: Embedding層のdropout
        :param dropout_classifier: readout後のdropout
        :param n_class:
        :param ninp_rusher: Rusherの特徴量次元
        :param pre_LN: Layer-Normalizationの配置方法をpre-LNにするか
        :param rusher_emb: RusherのEmbedding次元
        """
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(nemb, nhead, nhid, dropout_encoder, pre_LN=pre_LN, dropout_attn=dropout_attn)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.n_emb_layers = n_emb_layers
        self.conv1 = nn.Conv1d(in_channels=ninp, out_channels=nemb, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=nemb, out_channels=nemb, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=nemb, out_channels=nemb, kernel_size=1)

        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.relu3 = nn.PReLU()
        self.relu4 = nn.PReLU()
        self.relu5 = nn.ReLU()
        if dropout_embed > 0:
            self.dropout1 = nn.Dropout(dropout_embed)
        else:
            self.dropout1 = None

        self.avgpool = nn.AvgPool1d(kernel_size=22)  # nn.MaxPool1d(kernel_size=22)

        assert n_fin_layers == 3

        self.linear = nn.Sequential(
            nn.Dropout(dropout_classifier),
            nn.Linear(nemb + rusher_emb + meta_emb, nfinal),
            nn.ReLU(),
            nn.Dropout(dropout_classifier),
            nn.Linear(nfinal, nfinal),
            nn.ReLU(),
            nn.Dropout(dropout_classifier),
            nn.Linear(nfinal, n_class)
        )

        self.activation = nn.Softmax(dim=1)
        self.linear2 = nn.Linear(ninp_rusher, rusher_emb)
        self.gauss_noise = gauss_noise
        self.gauss_xy_noise = gauss_xy_noise
        if meta_emb > 0:
            self.linear3 = nn.Linear(ninp_meta, meta_emb)
        else:
            self.linear3 = None

    def forward(self, x: Data, with_clip: bool = True):
        # src: [Batch x Players(22) x Player Vector]
        src = x.players
        src_rusher = x.rusher

        # src: [Batch x Player Vector x Players(22)]
        src = src.permute([0, 2, 1])

        if self.training:
            # gaussian augmentation on training data
            if self.gauss_noise > 0.0:
                noise = torch.randn_like(src) * self.gauss_noise
                src = src + noise

            if self.gauss_xy_noise > 0.0:
                # dx = torch.randn(src.size(0)) * self.gauss_xy_noise
                dy = torch.randn(src.size(0)) * self.gauss_xy_noise

                # Batch x 1 x 22
                # src[:, 0, :] += dx.reshape(src.size(0), 1).expand(src.size(0), src.size(2))
                src[:, 1, :] += dy.reshape(src.size(0), 1).expand(src.size(0), src.size(2))

                # src_rusher[:, 0] += dx
                src_rusher[:, 1] += dy

        # src: [Batch x Player Vector(embedded, 4*inp) x Players(22)]
        src = self.relu1(self.conv1(src))
        src = self.dropout1(src)
        src = self.relu2(self.conv2(src))
        src = self.dropout1(src)
        src = self.relu3(self.conv3(src))

        src = src.permute([2, 0, 1])

        # output: [Players(22) x Batch x Transformed Player Vector]
        output = self.transformer_encoder(src)

        # output: [Batch x Transformed Player Vector x Players(22)]
        output = output.permute([1, 2, 0])

        # output: [Batch x Transformed Player Vector]
        output = torch.squeeze(self.avgpool(output), dim=2)

        if self.linear3 is not None:
            output = torch.cat((output, self.relu4(self.linear2(src_rusher)), self.relu5(self.linear3(x.meta))), dim=1)
        else:
            output = torch.cat((output, self.relu4(self.linear2(src_rusher))), dim=1)

        # output: [Batch x n_class]
        output = self.linear(output)
        output = self.activation(output)

        if not self.training and x.yardLine is not None:
            output = torch.cumsum(output, dim=1).numpy()

            output = np.clip(output, 0.0, 1.0)

            # mask
            if with_clip:
                left = 99 - x.yardLine
                right = 199 - x.yardLine
                for k in range(len(output)):
                    output[k, :left[k] + 1] = 0.0
                    output[k, right[k]:] = 1.0

        return output


#######################################################################################################################
# Prep and Feature Engineering
#######################################################################################################################

# Thanks to: https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python
def prep(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df['ToLeft'] = df.PlayDirection == "left"
    df['IsBallCarrier'] = df.NflId == df.NflIdRusher

    df.loc[df.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
    df.loc[df.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

    df.loc[df.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
    df.loc[df.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

    df.loc[df.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
    df.loc[df.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

    df.loc[df.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
    df.loc[df.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"

    # standardization
    df['TeamOnOffense'] = "home"
    df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    df['IsOnOffense'] = df.Team == df.TeamOnOffense  # Is player on offense?
    df['YardLine_std'] = 100 - df.YardLine
    df.loc[df.FieldPosition.fillna('') == df.PossessionTeam, 'YardLine_std'] = df.loc[
        df.FieldPosition.fillna('') == df.PossessionTeam, 'YardLine']
    df['X_std'] = df.X
    df.loc[df.ToLeft, 'X_std'] = 120 - df.loc[df.ToLeft, 'X']
    df['Y_std'] = df.Y
    df.loc[df.ToLeft, 'Y_std'] = 160 / 3 - df.loc[df.ToLeft, 'Y']
    df['Orientation_std'] = df.Orientation
    df.loc[df.ToLeft, 'Orientation_std'] = np.mod(180 + df.loc[df.ToLeft, 'Orientation_std'], 360)
    df['Dir_std'] = df.Dir
    df.loc[df.ToLeft, 'Dir_std'] = np.mod(180 + df.loc[df.ToLeft, 'Dir_std'], 360)
    df['IsOffence'] = df['Team'] == df['TeamOnOffense']

    # translate Home/Visitor to Offence/Defense
    df['OffenceScoreBeforePlay'] = df['HomeScoreBeforePlay']
    df.loc[df.TeamOnOffense == "away", 'OffenceScoreBeforePlay'] = df.loc[
        df.TeamOnOffense == "away", 'VisitorScoreBeforePlay']
    df['DefenseScoreBeforePlay'] = df['VisitorScoreBeforePlay']
    df.loc[df.TeamOnOffense == "away", 'DefenseScoreBeforePlay'] = df.loc[
        df.TeamOnOffense == "away", 'HomeScoreBeforePlay']

    df['OffenceTeamAbbr'] = df['HomeTeamAbbr']
    df.loc[df.TeamOnOffense == "away", 'OffenceTeamAbbr'] = df.loc[df.TeamOnOffense == "away", 'VisitorTeamAbbr']
    df['DefenseTeamAbbr'] = df['VisitorTeamAbbr']
    df.loc[df.TeamOnOffense == "away", 'DefenseTeamAbbr'] = df.loc[df.TeamOnOffense == "away", 'HomeTeamAbbr']

    df['Year'] = pd.to_datetime(df.TimeSnap).dt.year
    df.loc[df['Year'] == 2017, 'Orientation_std'] = np.mod(90 + df.loc[df['Year'] == 2017, 'Orientation_std'], 360)

    player_features = ['X_std', 'Y_std', 'S', 'A', 'Dis', 'Orientation_std', 'Dir_std', 'NflId', 'JerseyNumber',
                       'PlayerHeight',
                       'PlayerWeight', 'PlayerBirthDate', 'PlayerCollegeName', 'Position', 'IsBallCarrier', 'IsOffence']

    play_features = ['YardLine_std', 'Quarter', 'GameClock', 'PossessionTeam', 'Down', 'Distance', 'FieldPosition',
                     'OffenceScoreBeforePlay', 'DefenseScoreBeforePlay', 'OffenseFormation', 'OffensePersonnel',
                     'DefendersInTheBox', 'DefensePersonnel', 'TimeHandoff', 'TimeSnap',
                     'OffenceTeamAbbr', 'DefenseTeamAbbr', 'Week', 'Stadium', 'Location',
                     'Turf', 'GameWeather', 'Temperature', 'Humidity', 'WindSpeed', 'WindDirection',
                     'TeamOnOffense', 'HomeTeamAbbr', 'Year']

    if 'Yards' in df:
        play_features.append('Yards')

    players = df[['PlayId', 'GameId'] + player_features].copy()

    play = df[['GameId', 'PlayId'] + play_features].copy().drop_duplicates(subset=['PlayId'])

    return play, players


def prep_players_nn(play, players, scaler=None, scaler_meta=None):
    p = players.drop(['NflId', 'JerseyNumber', 'PlayerHeight', 'PlayerBirthDate', 'PlayerCollegeName', 'GameId'],
                     axis=1).copy()

    if scaler is None:
        is_training = True
    else:
        assert scaler_meta is not None
        is_training = False

    if 'YardLine_std' not in p:
        p = pd.merge(p, play[['YardLine_std', 'PlayId']], on='PlayId', how='left')

    p['IsBallCarrier'] = p['IsBallCarrier'].astype(int)
    p['IsOffence'] = p['IsOffence'].astype(int)
    p['X_std'] -= p['YardLine_std']

    p['Dir_cos'] = np.cos(np.deg2rad(90 - p['Dir_std']))
    p['Dir_sin'] = np.sin(np.deg2rad(90 - p['Dir_std']))

    p = p.fillna(0)

    rb = p[p['IsBallCarrier'] == 1][['X_std', 'Y_std', 'PlayId', 'S', 'Dir_sin', 'Dir_cos', 'Dir_std']]
    rb.columns = ['X_', 'Y_', 'PlayId', 'S_', 'Dir_sin_', 'Dir_cos_', 'Dir_std_']

    p = pd.merge(p, rb, how='left')
    p['DX'] = p['X_std'] - p['X_']
    p['DY'] = p['Y_std'] - p['Y_']

    # Relative angle from on Rusher's vector
    angles = 90.0 - np.rad2deg(np.arctan2(p['DY'], p['DX']))
    p['AngleFromRB'] = angles - p['Dir_std_']
    p['AngleFromRB'] = np.mod(p['AngleFromRB'] + 360, 360)
    p.loc[p['AngleFromRB'] > 180, 'AngleFromRB'] -= 360

    # dTheta of AngleFromRB
    dt = 0.001
    p['DX2'] = (p['X_std'] + dt * p['Dir_cos'] * p['S']) - (p['X_'] + dt * p['Dir_cos_'] * p['S_'])
    p['DY2'] = (p['Y_std'] + dt * p['Dir_sin'] * p['S']) - (p['Y_'] + dt * p['Dir_sin_'] * p['S_'])

    angles = 90.0 - np.rad2deg(np.arctan2(p['DY2'], p['DX2']))
    p['AngleFromRB2'] = angles - p['Dir_std_']
    p['AngleFromRB2'] = np.mod(p['AngleFromRB2'] + 360, 360)
    p.loc[p['AngleFromRB2'] > 180, 'AngleFromRB2'] -= 360
    p['AngleFromRB2'] = p['AngleFromRB2'] - p['AngleFromRB']
    p.loc[p['AngleFromRB2'] > 180, 'AngleFromRB2'] -= 360
    p.loc[p['AngleFromRB2'] < -180, 'AngleFromRB2'] += 360

    p.loc[p['IsBallCarrier'] == 1, 'AngleFromRB'] = 0
    p.loc[p['IsBallCarrier'] == 1, 'AngleFromRB2'] = 0

    p.drop(['DX2', 'DY2'], axis=1, inplace=True)

    p['AngleTan'] = np.arctan2(p['DY'], p['DX'])

    p = p.replace([np.inf, -np.inf], np.nan)
    p = p.fillna(0)
    p.drop(['X_', 'Y_', 'Dir_sin_', 'Dir_cos_', 'S_', 'Dir_std_', 'Orientation_std', 'Dir_std'], axis=1, inplace=True)

    concat = p.drop(['Position', 'PlayId', 'YardLine_std'], axis=1)

    if is_training:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(concat.values)
    else:
        scaled = scaler.transform(concat.values)
    df = pd.DataFrame(scaled, columns=concat.columns)

    df_rusher = df[p['IsBallCarrier'] == 1].reset_index(drop=True)
    drop_cols = ['IsBallCarrier', 'IsOffence', 'OffenseDist0', 'PlayerWeight', 'DX', 'DY', 'AngleFromRB', 'AngleToRB',
                 'DistDelta', 'AngleTan', 'AngleFromRB2', 'delaunay_adj']
    drop_cols = [c for c in drop_cols if c in df_rusher.columns]
    df_rusher.drop(drop_cols, axis=1, inplace=True)
    df_rusher = df_rusher.set_index(players['PlayId'].iloc[np.arange(0, len(players), 22)])

    # meta features
    meta = play[['YardLine_std', 'Distance']].copy()
    meta = meta.replace([np.inf, -np.inf], np.nan)
    meta = meta.fillna(0)

    if scaler_meta is not None:
        scaled_meta = scaler_meta.transform(meta.values)
    else:
        scaler_meta = StandardScaler()
        scaled_meta = scaler_meta.fit_transform(meta.values)
    scaled_meta = pd.DataFrame(scaled_meta, columns=meta.columns)

    return df.set_index(players['PlayId']), df_rusher, scaled_meta, scaler, scaler_meta


#######################################################################################################################
## SnapShot & Ensemble
#######################################################################################################################


class EnsembleModel(object):
    def __init__(self):
        self.models = []

    def add_model(self, model):
        self.models.append(model)

    def train(self):
        for m in self.models:
            m.train()

    def eval(self):
        for m in self.models:
            m.eval()

    def __call__(self, *input, **kwargs):
        assert len(self.models) >= 1

        base = self.models[0](*input, **kwargs)

        for m in self.models[1:]:
            base += m(*input, **kwargs)

        return base / len(self.models)


class SnapShot(object):
    def __init__(self, model: nn.Module, epoch: int, loss: float):
        self.state = copy.deepcopy(model.state_dict())
        self.epoch = epoch
        self.loss = loss
        torch.save(self.state, f'snapshot_{epoch}_{loss}')


class SnapShots(object):
    def __init__(self, interval: int = 3, torelance: float = 1.01, verbose: bool = True):
        self.best_model = None  # type: Optional[SnapShot]
        self.snap_shots = []  # type: List[SnapShot]
        self.best_val_loss = 1.0
        self.interval = interval
        self.torelance = torelance
        self.verbose = verbose

    def add(self, model: nn.Module, epoch: int, loss: float):
        if loss < self.best_val_loss:
            if self.verbose:
                print(f'best model is updated. epoch{epoch}, loss={loss:.7f}')
            self.best_model = SnapShot(model, epoch, loss)
            self.best_val_loss = loss
        if epoch % self.interval == 0:
            if self.verbose:
                print(f'Add snapshot. epoch{epoch}, loss={loss:.7f}')
            self.snap_shots.append(SnapShot(model, epoch, loss))

    def load_best_single_model(self, model: nn.Module):
        if self.best_model is not None:
            model.load_state_dict(self.best_model.state)
        return model

    def load_ensemble_model(self, cls: Type, params: Dict, max_models: int = 5):
        model = EnsembleModel()
        best = cls(**params)
        self.load_best_single_model(best)
        model.add_model(best)

        candidates = []  # Tuple[int, float]

        for i, s in enumerate(self.snap_shots):
            if s.loss > self.torelance * self.best_model.loss:
                continue
            if s.epoch == self.best_model.epoch:
                continue

            candidates.append((i, s.loss))

        # Collect top-n models from snapshot
        candidates = sorted(candidates, key = lambda x: x[1])
        if len(candidates) > max_models - 1:
            candidates = candidates[:max_models - 1]
        for i, _ in candidates:
            s = self.snap_shots[i]
            sub = cls(**params)
            sub.load_state_dict(s.state)
            model.add_model(sub)
            if self.verbose:
                print(f'add {s.epoch}-th epoch to ensemble (loss:{s.loss:.7f})')

        return model


#######################################################################################################################
# Training & Pseudo-Labeling
#######################################################################################################################


def train_model(model: nn.Module, scheduler, batch_size: int,
                train_data: Data, valid_data: Data, writer, epochs: int,
                downsample_2017: float,
                calc_train_loss: bool = True,
                params: Dict = None):
    np.random.seed(0)

    # Keep number of data in 1 epoch same between 1st/2nd stage
    std_batch_length = 12000 // batch_size + 1
    n_total_batch = 0
    epoch = 1
    epoch_start_time = time.time()
    ensemble = None

    model.eval()

    snapshots = SnapShots(torelance=1.007, interval=3)
    snapshots.add(model, 0, evaluate(model, valid_data))

    model.train()  # Turn on the train mode

    while epoch < epochs:
        data = train_data.shuffled()
        if downsample_2017 > 0.0:
            data = data.downsample_2017(downsample_2017)

        for i in range(0, data.len() - 1, batch_size):
            iend = min(i + batch_size, data.len())
            batch_data = data.slice(i, iend)
            optimizer.zero_grad()
            output = model(batch_data)

            loss = crps(output, batch_data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            n_total_batch += 1

            if n_total_batch > std_batch_length:
                model.eval()

                if calc_train_loss and epoch % 10 == 0:
                    train_loss = evaluate(model, data)
                else:
                    train_loss = -1

                if epoch % 4 == 0 or epoch > 30:
                    val_loss = evaluate(model, valid_data)
                    snapshots.add(model, epoch, val_loss)
                else:
                    val_loss = -1

                print('| end of epoch {:3d} | lr: {:2.5f} | time: {:5.2f}s | train loss {:5.6f} | '
                      'valid loss {:5.9f}'.format(epoch, scheduler.get_lr()[0], (time.time() - epoch_start_time),
                                                  train_loss, val_loss))
                scheduler.step()
                if writer is not None:
                    writer.writerow([epoch, scheduler.get_lr()[0], train_loss, val_loss])
                    f.flush()

                n_total_batch = 0
                epoch += 1
                epoch_start_time = time.time()
                model.train()  # Turn on the train mode

                if epoch in [40]:
                    batch_size *= 2
                    std_batch_length = 12000 // batch_size + 1

    # reload best model
    model = snapshots.load_best_single_model(model)
    eval_single = evaluate(model, valid_data)

    try:
        ensemble = snapshots.load_ensemble_model(TransformerModel, params)
        eval_ensemble = evaluate(ensemble, valid_data)
        print(f'final loss: {eval_single:.8f} (single) / {eval_ensemble:.8f} ({len(ensemble.models)} models)')
    except:
        pass

    return ensemble if ensemble is not None else model


def evaluate(eval_model: nn.Module, data: Data, with_clip: bool = True):
    eval_model.eval()  # Turn on the evaluation mode
    assert data.y is not None

    with torch.no_grad():
        y_actual = data.y.numpy()
        y_predicted = eval_model(data, with_clip)
        loss = np.sum((y_predicted - y_actual) ** 2) / (199 * len(y_predicted))

    return loss


#######################################################################################################################
# Load & Validation Split
#######################################################################################################################


def return_delta(x):
    temp = np.zeros(199)
    temp[x + 99:] = 1
    return temp


def NFL_validation_split(df: pd.DataFrame, game_set=None):
    games = df[['GameId', 'PossessionTeam']].drop_duplicates()

    # Sort so the latest games are first and label the games with cumulative counter
    games = games.sort_values(['PossessionTeam', 'GameId'], ascending=[True, False])
    games['row_number'] = games.groupby(['PossessionTeam']).cumcount() + 1

    # Use last 5 games for each team as validation. There will be overlap since two teams will have the same
    # GameId
    game_set = game_set or {1, 2, 3, 4, 5}

    # Set of unique game ids
    game_ids = set(games[games['row_number'].isin(game_set)]['GameId'].unique().tolist())

    return game_ids


def NFL_group_split(df: pd.DataFrame, nfolds: int, nidx: int):
    kf = GroupKFold(nfolds)

    train_idx, valid_idx = list(kf.split(df, groups=df['GameId']))[nidx]

    return set(df['GameId'].iloc[valid_idx].unique())


def load_data_nn_test(test_df: pd.DataFrame, scaler: StandardScaler, scaler_meta: StandardScaler) -> Data:
    play, players = prep(test_df)
    players_nn, rusher_nn, meta_nn, _, _ = prep_players_nn(play, players, scaler, scaler_meta)
    X = players_nn.values.astype(np.float32).reshape((-1, 22, len(players_nn.columns)))
    Xr = rusher_nn.values.astype(np.float32)
    Xm = meta_nn.values.astype(np.float32)
    year = play['Year'].values.copy()

    return Data(torch.from_numpy(X), torch.from_numpy(Xr), torch.from_numpy(Xm),
                None, play['YardLine_std'].values.copy(), year,
                list(players_nn.columns), list(rusher_nn.columns), list(meta_nn.columns))


def load_data_nn(n_plays_sample=None,
                 nfolds=None, nidx=None,
                 skiprows=None,
                 game_set=None) -> Tuple[Data, Data, StandardScaler, StandardScaler]:
    train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', nrows=n_plays_sample, skiprows=skiprows)
    

    play, players = prep(train)
    players_nn, rusher_nn, meta_nn, scaler, scaler_meta = prep_players_nn(play, players, None, None)

    assert len(players_nn) == len(rusher_nn) * 22
    assert len(rusher_nn) == len(meta_nn)
    assert len(rusher_nn) == len(play)

    play_ids = np.array(players_nn.index[np.arange(0, len(players_nn), 22)])
    game_ids = np.array(players['GameId'].values[np.arange(0, len(players_nn), 22)])

    assert len(play_ids) == len(game_ids)

    if nfolds is not None:
        assert nidx is not None
        game_ids_valid = NFL_group_split(train, nfolds, nidx)
    else:
        game_ids_valid = NFL_validation_split(train, game_set)

    X = players_nn.values.astype(np.float32).reshape((-1, 22, len(players_nn.columns)))
    Xr = rusher_nn.values.astype(np.float32)
    Xm = meta_nn.values.astype(np.float32)

    y = np.vstack(play['Yards'].apply(return_delta).values).astype(np.float32)

    yardLine = play['YardLine_std'].values
    year = play['Year'].values

    valid_mask = np.isin(game_ids, np.array(list(game_ids_valid)))

    X_valid = torch.from_numpy(X[valid_mask])
    Xr_valid = torch.from_numpy(Xr[valid_mask])
    Xm_valid = torch.from_numpy(Xm[valid_mask])
    y_valid = torch.from_numpy(y[valid_mask])
    yd_valid = yardLine[valid_mask].copy()
    yr_valid = year[valid_mask].copy()

    X_train = torch.from_numpy(X[~valid_mask])
    Xr_train = torch.from_numpy(Xr[~valid_mask])
    Xm_train = torch.from_numpy(Xm[~valid_mask])
    y_train = torch.from_numpy(y[~valid_mask])
    yd_train = yardLine[~valid_mask].copy()
    yr_train = year[~valid_mask].copy()

    print(f'X_train: {X_train.shape}, {Xr_train.shape}, {Xm_train.shape}')
    print(f'X_valid: {X_valid.shape}, {Xr_valid.shape}, {Xm_valid.shape}')
    print(f'y_train: {y_train.shape}, y_valid: {y_valid.shape}')
    print(f'players: {list(players_nn.columns)}')
    print(f'rusher: {list(rusher_nn.columns)}')
    print(f'meta: {list(meta_nn.columns)}')

    dtrain = Data(X_train, Xr_train, Xm_train, y_train, yd_train, yr_train,
                  list(players_nn.columns), list(rusher_nn.columns), list(meta_nn.columns))
    dvalid = Data(X_valid, Xr_valid, Xm_valid, y_valid, yd_valid, yr_valid,
                  list(players_nn.columns), list(rusher_nn.columns), list(meta_nn.columns))

    assert len(X) == len(y)
    return dtrain, dvalid, scaler, scaler_meta


def crps(y_pred, y_true):
    loss = torch.mean((torch.cumsum(y_pred, dim=1) - y_true) ** 2)
    return loss


#######################################################################################################################
## Start Training
#######################################################################################################################

n_holdout = 0
n_train = None

train_data, valid_data, scaler, scaler_meta = load_data_nn(None)
n_inp = train_data.players.shape[2]  # Dimension of feature vector per player
nemb = 128  # Dimension of embedding vector per player
nhead = 1  # multi-head attention
nhid = 96  # number of hidden units in attention
nlayers = 4  # number of transformers stacked
nfinal = 512  # number of hidden units in final layers
lr = 0.0001
gamma = 0.976
dropout_encoder = 0.0
n_emb_layers = 3
dropout_classifier = 0.3
dropout_embed = 0.15
dropout_attn = 0.35
batch_size = 16
epochs = 100
meta_emb = 8
downsample_2017 = 0.4
gauss_noise = 0.15
gauss_xy_noise = 0.1
n_fin_layers = 3

mode = 'ensemble'
log_filename = 'kernel_v92_encoder_0_attn_02'

params = {
    'ninp': n_inp,
    'nemb': nemb,
    'nhead': nhead,
    'nhid': nhid,
    'nlayers': nlayers,
    'nfinal': nfinal,
    'ninp_rusher': train_data.rusher.shape[1],
    'pre_LN': True,
    'dropout_encoder': dropout_encoder,
    'dropout_embed': dropout_embed,
    'dropout_classifier': dropout_classifier,
    'n_emb_layers': n_emb_layers,
    'ninp_meta': train_data.meta.shape[1],
    'meta_emb': meta_emb,
    'gauss_noise': gauss_noise,
    'gauss_xy_noise': gauss_xy_noise,
    'n_fin_layers': n_fin_layers,
    'dropout_attn': dropout_attn
}

no_decay = ['bias', '.norm']

if mode == 'grid':
    for dropout_embed in [0.05, 0.1, 0.15]:
        for dropout_encoder in [0.2, 0.3, 0.4]:
            log_f = f'{log_filename}_embed{dropout_embed}_encoder{dropout_encoder}.csv'
            params['dropout_encoder'] = dropout_encoder
            params['dropout_embed'] = dropout_embed
            model = TransformerModel(**params)

            f = open(log_f, 'w+', newline='')
            writer = csv.writer(f)
            writer.writerow([dropout_embed, dropout_classifier])
            writer.writerow(['epoch', 'lr', 'train_loss', 'valid_loss'])

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=gamma)

            model = train_model(model, scheduler, batch_size, train_data, valid_data, writer, epochs, downsample_2017, params=params)
elif mode == 'ensemble':
    models = EnsembleModel()
    game_sets = [
        {1, 3, 5, 7, 9},
        {2, 4, 6, 8, 10}
    ]

    for i in range(len(game_sets)):
        model = TransformerModel(**params)
        train_data, valid_data, scaler, scaler_meta = load_data_nn(None, nfolds=8, nidx=i) #load_data_nn(None, game_set=game_sets[i])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)    
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=gamma)

        model = train_model(model, scheduler, batch_size, train_data, valid_data, None, epochs, downsample_2017,
                            calc_train_loss=False, params=params)
        models.add_model(model)
    model = models
else:
    assert mode == 'single'
    model = TransformerModel(**params)
    f = open(log_filename + '.csv', 'w+', newline='')
    writer = csv.writer(f)
    writer.writerow(train_data.player_cols)
    writer.writerow(train_data.rusher_cols)
    writer.writerow(['epoch', 'lr', 'train_loss', 'valid_loss'])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=gamma)

    model = train_model(model, scheduler, batch_size, train_data, valid_data, writer, epochs, downsample_2017, params=params)



# In[2]:


def tta(test_df, sigma_dir=1.0, sigma_y=1.0):
    n_aug = 10
    test_df_aug = pd.concat([test_df]*n_aug)
    
    test_df_aug['Dir'] += np.random.normal(0, sigma_dir, size=len(test_df_aug))

    # yは共通で上げ下げ
    test_df_aug['Y'] += np.repeat(np.random.normal(0, sigma_y, size=n_aug), 22)
    test_df_aug['PlayId'] += np.repeat(np.arange(0, n_aug), 22)
    
    return test_df_aug


# In[3]:


from kaggle.competitions import nflrush
env = nflrush.make_env()


# In[4]:


model.eval()  # Turn on the evaluation mode
n_prev = 0

original_data = concat_dataset(train_data, valid_data)

for (test_df, sample_prediction_df) in env.iter_test():
    try:
        test_df = tta(test_df)
        X_test = load_data_nn_test(test_df, scaler, scaler_meta)

    except Exception as e:
        print(f'### ERROR ### {e} / {test_df}')
        # submit as-is if something happened
        env.predict(sample_prediction_df)
        continue

    with torch.no_grad():
        predicted = model(X_test).mean(axis=0)
        sample_prediction_df.iloc[0,:] = np.squeeze(predicted)
        env.predict(sample_prediction_df)


# In[5]:


env.write_submission_file()   

