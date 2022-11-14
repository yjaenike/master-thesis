""" Defines the transformer network """
import torch
import torch.nn as nn
import numpy as np
from tstransformer.Layers import EncoderLayer, DecoderLayer


def get_pad_mask(seq, pad_idx):
    """ Rezurns a padded sequence mask"""
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class PositionalEncoding(nn.Module):
    """Positional Encoding of the input sequence
    
    Atributes
    ---------
    register_buffer (register_buffer): https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
    pos_table (torch.Tensor): The sinusoidal encoding table
    
    Methods
    -------
    _get_sinusoid_encoding_table(self, n_position, d_hid): Creates a sinusoidal encoding table 
    forward(self, x): Adds the positional encoding to the input vector x
    
    """

    def __init__(self, d_hid, n_position=200):
        """
        d_hid (int): Inner dimensionality 
        n_position (int): Outer dimenisonality
        """
        super(PositionalEncoding, self).__init__()

        # Not a parameter, but part of the module’s state
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """ Sinusoid position encoding table """
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """ Adds the positional encoding to the input vector x """
        return x + self.pos_table[:, :x.size(1)].clone().detach()
    
    
class Encoder(nn.Module):
    """The encoder part of the transformer model
    
    Atributes
    ---------
    src_word_emb (nn.Embedding): Embedding Layer 
    position_enc (PositionalEncoding): Positional Encoding Layer
    dropout (float): dropout rate
    layer_stack (nn.ModuleList): List of encoding layers
    layer_norm (nn.LayerNorm): Normalisation Layer
    scale_emb (Boolean): If true, the enc_output is scaled with the sqrt of the model dimensioanlity
    d_model (int): dimensionality of the model
    
    Methods
    -------
    forward(self, src_seq, src_mask, return_attns=False):
    """

    def __init__(self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
        """
        n_src_vocab (int): size of the dictionary of embeddings
        d_word_vec (int): the size of each embedding vector
        n_layers (int): the number of encoder layers 
        n_head (int): the number of attention heads
        d_k (int): dimensionality of the key vector
        d_v (int): dimensionality of the value vector
        d_model (int): dimensionality of the model
        d_inner (int): dimensionality of the hidden layer in the Positionwise Feed Forward Module
        pad_idx (int): ff specified, the entries at padding_idx do not contribute to the gradient of the embedding
        dropout (float): dropout rate
        n_position (int): Outer dimenisonality of the positional encoding
        scale_emb (Boolean): If true, the enc_output is scaled with the sqrt of the model dimensioanlity
        """
        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        """Performs one forward step through the encoder 
        
        :param: src_seq (Tensor): The Input sequence that is fed into the Input Embedding
        :param: src_mask (BoolTensor): A boolean tensor indicating which values should be masked
        :param: return_attns (Boolean): If True, returns the attention mat.
        
        :return: enc_output (torch.Tensor): The output Tensor of the Module
        :return: enc_slf_attn_list (List): List of self attention matrices
        """

        enc_slf_attn_list = []

        # -- Forward
        # Create input embedding
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        
        # Add position encoding
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)
        
        # Run through encoder layers
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        
        
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,
    
    
class Decoder(nn.Module):
     """The decoder part of the transformer model
    
    Atributes
    ---------
    src_word_emb (nn.Embedding): Embedding Layer 
    position_enc (PositionalEncoding): Positional Encoding Layer
    dropout (float): dropout rate
    layer_stack (nn.ModuleList): List of encoding layers
    layer_norm (nn.LayerNorm): Normalisation Layer
    scale_emb (Boolean): If true, the enc_output is scaled with the sqrt of the model dimensioanlity
    d_model (int): dimensionality of the model
    
    Methods
    -------
    forward(self, src_seq, src_mask, return_attns=False):
    """

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):
        """
        n_src_vocab (int): size of the dictionary of embeddings
        d_word_vec (int): the size of each embedding vector
        n_layers (int): the number of encoder layers 
        n_head (int): the number of attention heads
        d_k (int): dimensionality of the key vector
        d_v (int): dimensionality of the value vector
        d_model (int): dimensionality of the model
        d_inner (int): dimensionality of the hidden layer in the Positionwise Feed Forward Module
        pad_idx (int): ff specified, the entries at padding_idx do not contribute to the gradient of the embedding
        dropout (float): dropout rate
        n_position (int): Outer dimenisonality of the positional encoding
        scale_emb (Boolean): If true, the enc_output is scaled with the sqrt of the model dimensioanlity
        """
        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
         """Performs one forward step through the decoder 
        
        :param: src_seq (Tensor): The Input sequence that is fed into the Input Embedding
        :param: src_mask (BoolTensor): A boolean tensor indicating which values should be masked
        :param: return_attns (Boolean): If True, returns the attention mat.
        
        :return: enc_output (torch.Tensor): The output Tensor of the Module
        :return: enc_slf_attn_list (List): List of self attention matrices
        """
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,