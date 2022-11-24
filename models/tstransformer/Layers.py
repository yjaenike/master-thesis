""" The encoder and decoder layer of the transformer network """

import torch.nn as nn
import torch
from tstransformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """The encoder layer 
    
    This is a simple encoder layer based on the "Attention is all you need paper" - Ashish Vaswani
    
    Atributes
    ---------
    slf_attn (MultiHeadAttention): Multihead Attention Module 
    pos_fnn (PositionwiseFeedForward): Positionwise Feed Forward Module
    
    Methods
    -------
    forward(self, enc_input, slf_attn_mask=None): Performs one forward step through the layer
    """
    
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        """
        d_model (int): Model dimensionality
        d_inner (int): Dimensionality of the hidden layer in the Positionwise Feed Forward Module
        n_head (int): Number of head in the Multi Head Attention Module
        d_k (int): key dimensionality
        d_v (int): value dimensionality
        dropout (float): dropout rate
        """
        super(EncoderLayer, self).__init__()
        #self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        
        self.slf_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, kdim=d_k, vdim=d_v, batch_first=True)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        
    def forward(self, enc_input, slf_attn_mask=None):
        """Performs one forward step through the layer
        
        :param: enc_input (torch.Tensor): Input to the encoder layer
        :param: slf_attn_mask (torch.BoolTensor): A boolean tensor indicating which values should be masked
        
        :return: enc_output(): Encoder Output 
        :return: enc_slf_attn():The Attention map from the MultiheadAttentionModule
        """
        
        #enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        
        enc_output, enc_slf_attn = self.slf_attn(enc_input,enc_input,enc_input, attn_mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
    
class DecoderLayer(nn.Module):
    """The decoder layer 
    
    This is a simple decoder layer based on the "Attention is all you need paper" - Ashish Vaswani
    
    Atributes
    ---------
    slf_attn (MultiHeadAttention): Masked Multihead Attention Module 
    enc_attn (MultiHeadAttention): Multi head Atenntion Module that has shared values with the encoder
    pos_fnn (PositionwiseFeedForward): Positionwise Feed Forward Module
    
    Methods
    -------
    forward(self, dec_input, enc_output,slf_attn_mask=None, dec_enc_attn_mask=None): Performs one forward step through the layer
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        """
        d_model (int): Model dimensionality
        d_inner (int): Dimensionality of the hidden layer in the Positionwise Feed Forward Module
        n_head (int): Number of head in the Multi Head Attention Module
        d_k (int): key dimensionality
        d_v (int): value dimensionality
        dropout (float): dropout rate
        """
        super(DecoderLayer, self).__init__()
        #self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        #self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        
        self.slf_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout, kdim=d_k, vdim=d_v, batch_first=True)
        self.enc_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout, kdim=d_k, vdim=d_v, batch_first=True)
        
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output,slf_attn_mask=None, dec_enc_attn_mask=None):
        """Performs one forward step through the layer
        
        :param: dec_input (torch.Tensor): Input to the encoder layer
        :param: enc_output (torch.Tensor): 
        :param: slf_attn_mask (torch.BoolTensor): A boolean tensor indicating which values should be masked
        :param: dec_enc_attn_mask (torch.BoolTensor): A boolean tensor indicating which values should be masked
        
        :return: dec_output(): Decoder Output 
        :return: dec_slf_attn(): The Attention map from the masked MultiheadAttentionModule
        :return: dec_enc_attn(): The Attention map from the encoder-decoder MultiheadAttentionModule 
        """
        
        #dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        #dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        
        
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, attn_mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        
        
        dec_output = self.pos_ffn(dec_output)
        
        
        return dec_output, dec_slf_attn, dec_enc_attn

