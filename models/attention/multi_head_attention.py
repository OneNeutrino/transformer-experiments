"""
Multi-Head Attention Implementation.

This module implements the Multi-Head Attention mechanism as described in
'Attention Is All You Need' (Vaswani et al., 2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=True):
        """
        Args:
            query: (batch_size, tgt_len, embed_dim)
            key: (batch_size, src_len, embed_dim)
            value: (batch_size, src_len, embed_dim)
            key_padding_mask: (batch_size, src_len)
            attn_mask: (tgt_len, src_len) or (batch_size, num_heads, tgt_len, src_len)
            need_weights: bool, return attention weights
        """
        batch_size, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        
        scaling = self.scaling
        
        # Project and reshape
        q = self.q_proj(query) * scaling
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        q = q.contiguous().view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.contiguous().view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.contiguous().view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply masks
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
            
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
        
        # Normalize attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn = attn.transpose(1, 2).contiguous().view(batch_size, tgt_len, embed_dim)
        attn = self.out_proj(attn)
        
        if need_weights:
            return attn, attn_weights
        else:
            return attn, None