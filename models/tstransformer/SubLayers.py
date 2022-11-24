""" Implements the sublayers of the encoder decoder layers """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformer.Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    
    An Implementation of the MultiHeadAttention mechanism from the 
    'Attention is all you need paper'.
    
    Attributes
    ----------
    n_head (int): The number of attention heads for the module
    d_k (int): The dimensionality of the key-vector
    d_v (int): The dimensionality of the value-vector
    w_qs (nn.Linear): A Linear Layer that takes the query weights and gives them to the scaled dot-product attention layer
    w_ks (nn.Linear): A Linear Layer that takes the key weights and gives them to the scaled dot-product attention layer
    w_vs (nn.Linear): A Linear Layer that takes the value weights and gives them to the scaled dot-product attention layer
    fc (nn.Linear): Linear output layer
    attention (ScaledDotProductAttention): Implementation of the scaled dot product attention
    dropout (nn.Dropout): a dropout Layer
    layer_norm (nn.LayerNorm): NoramlisationLayer
    
    Methods
    -------
    forward(q, k, v, mask=None): Perfroms one forward step through the Module
    
    """
    
    def __init__ (self, n_head, d_model, d_k, d_v, dropout=0.1):
        """
        Parameters
        ----------
        n_head (int): The number of attention heads for the module
        d_model (int): The dimensionality of the input
        d_k (int): The dimensionality of the keys matrix
        d_v (int): The dimensionality of the value matrix
        dropout (float): The dropout rate
        """
        super().__init__()
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, q, k, v, mask=None):
        """Perfroms one forward step through the Module
        :param: q (torch.Tensor): Query Matrix
        :param: k (torch.Tensor): Key Matrix
        :param: v (torch.Tensor): Value Matrix
        :param: mask (BoolTensor): Boolean Tensor describing which elements should be masked
        :return: q (): Querie Matrix
        :return: attn (): Attention Matrix
        """
        
        d_k = self.d_k
        d_v = self.d_v
        n_head = self.n_head
        sz_b = q.size(0)
        len_q = q.size(1)
        len_k = k.size(1)
        len_v = v.size(1)
        

        
        residual = q
        
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        
        
        # Transpose for attention dot product: b x n x lq x dv
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        
        
        if mask is not None:
            mask = mask.unsqueeze(1) # For head axis broadcasting
        
        
        q, attn = self.attention(q, k, v, mask=mask)
        
        
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        

        q = self.layer_norm(q)

        return q, attn

    
class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module
    
    Position-Wise Feed-Forward Layer is a type of feedforward layer consisting of two 
    dense layers that applies to the last dimension, which means the same dense layers 
    are used for each position item in the sequence, so called position-wise.

    Attributes
    ----------
    w_1 (nn.Linear): First linear layers
    w_2 (nn.Linear): Second linear layers
    layer_norm (nn.LayerNorm): Normalisation layer
    dropout (nn.Dropout): Dropout layer
    
    Methods
    -------
    forward(self, x): Performs one step forward through the Module
    """

    def __init__(self, d_in, d_hid, dropout=0.1):
        """
        d_in (int): Input dimensionality
        d_out (int): Output dimensionality
        dropout (float): Dropout rate
        """
        super().__init__()
        
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """  Performs one step forward through the Module
        :param: x (torch.Tensor): Input Vector
        :return: x (torch.Tensor): Ouput Vecotr after running throught layers of this module
        """

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

    
class ScaledDotProductAttention(nn.Module):
    """Scaled Dot Product Attention
    
    Perfroms an Attention step on the values given the querys and keys.
    
    Attributes
    ----------
    temperature (float): The sqrt(d_k) is scaling the Query Matrix
    dropout (nn.Dropout): Droput Layer
    
    Methods
    -------
    forward(self, q, k, v, mask=None): Perfroms one attention step on the query , key and value matrix.
    
    """

    def __init__(self, temperature, attn_dropout=0.1):
        """
        temeprature (float): Square root of the key dimensionality
        attn_dropout (float): Dropout rate
        """
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """ Perfroms one forward step through the Module
        :param: q (torch.Tensor): Query Matrix
        :param: k (torch.Tensor): Key Matrix
        :param: v (torch.Tensor): Value Matrix
        :param: mask (BoolTensor): Boolean Tensor describing which elements should be masked
        :return: output (torch.Tensor): The Attention Matrix applied(multiplied) with the value matrix. I.e. The Scaled dot-product Attention Matrix 
        :return: attn (torch.Tensor): The attention matrix  
        """
        
        
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        
        
        if mask is not None:
            #TODO: mask is not working properly
            #attn = attn.masked_fill(mask == 0, -1e9)
            pass

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn