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
    src_sequence_emb (nn.Embedding): Embedding Layer 
    position_enc (PositionalEncoding): Positional Encoding Layer
    dropout (float): dropout rate
    layer_stack (nn.ModuleList): List of encoding layers
    layer_norm (nn.LayerNorm): Normalisation Layer
    scale_emb (Boolean): If true, the enc_output is scaled with the sqrt of the model dimensioanlity
    d_model (int): dimensionality of the model
    
    Methods
    -------
    forward(self, src_seq, src_mask, return_attns=False): Performs one forward step through the encoder 
    """

    def __init__(self, n_src_sequence, d_sequence_vec, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
        """
        n_src_sequence (int): size of the dictionary of the embedding
        d_sequence_vec (int): the size of each embedding vector
        n_layers (int): the number of encoder layers 
        n_head (int): the number of attention heads
        d_k (int): dimensionality of the key vector
        d_v (int): dimensionality of the value vector
        d_model (int): dimensionality of the model
        d_inner (int): dimensionality of the hidden layer in the Positionwise Feed Forward Module
        pad_idx (int): if specified, the entries at padding_idx do not contribute to the gradient of the embedding
        dropout (float): dropout rate
        n_position (int): Outer dimenisonality of the positional encoding
        scale_emb (Boolean): If true, the enc_output is scaled with the sqrt of the model dimensioanlity
        """
        super().__init__()

        self.src_sequence_emb = nn.Embedding(n_src_sequence, d_sequence_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_sequence_vec, n_position=n_position)
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
        enc_output = self.src_sequence_emb(src_seq)
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
    trg_sequence_emb (nn.Embedding): Input Embedding Layer of the target sequence
    position_enc (PositionalEncoding): Positional Encoding Layer
    dropout (float): dropout rate
    layer_stack (nn.ModuleList): List of encoding layers
    layer_norm (nn.LayerNorm): Normalisation Layer
    scale_emb (Boolean): If true, the dec_output is scaled with the sqrt of the model dimensioanlity
    d_model (int): dimensionality of the model
    
    Methods
    -------
    forward(trg_seq, trg_mask, enc_output, src_mask, return_attns=False): Performs one forward step through the decoder 
    """
    
    def __init__(self, n_trg_sequence, d_sequence_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
        """
        n_trg_sequence (int): size of the dictionary of the embedding
        d_sequence_vec (int): the size of each embedding vector
        n_layers (int): the number of encoder layers 
        n_head (int): the number of attention heads
        d_k (int): dimensionality of the key vector
        d_v (int): dimensionality of the value vector
        d_model (int): dimensionality of the model
        d_inner (int): dimensionality of the hidden layer in the Positionwise Feed Forward Module
        pad_idx (int): if specified, the entries at padding_idx do not contribute to the gradient of the embedding
        dropout (float): dropout rate
        n_position (int): Outer dimenisonality of the positional encoding
        scale_emb (Boolean): If true, the enc_output is scaled with the sqrt of the model dimensioanlity
        """
        super().__init__()

        self.trg_sequence_emb = nn.Embedding(n_trg_sequence, d_sequence_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_sequence_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        """Performs one forward step through the decoder 
        
        :param: trg_seq (Tensor): The Input sequence that is fed into the Input Embedding
        :param: trg_mask (BoolTensor): A boolean tensor indicating which values should be masked
        :param: enc_output (torch.Tensor): 
        :param: src_mask (BoolTensor): A boolean tensor indicating which values should be masked
        :param: return_attns (Boolean): If True, returns the attention mat.
        
        :return: dec_output (torch.Tensor): The output Tensor of the Module
        :return: dec_slf_attn_list (List): List of decoder self attention matrices
        :return: dec_enc_attn_list(List): List of decoder encoder attention matrices
        """
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_sequence_emb(trg_seq)
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
    

class Transformer(nn.Module):
    """ The Transformer encoder-decoder Sequence to Sequence Model
    
    Atributes
    ---------
    src_pad_idx (int): if specified, the entries at padding_idx do not contribute to the gradient of the embedding
    trg_pad_idx (int): if specified, the entries at padding_idx do not contribute to the gradient of the embedding
    scale_prj (boolean): True if the scale_emb_or_prj is 'prj' -> scales the seq_logit at the end of the decoder
    d_model (int): dimensionality of the model
    encoder (Encoder): The Encoder part of the Transformer
    decoder (Decoder): The Decoder part of the Transformer
    trg_sequence_prj (nn.Linear): The linear layer after the decoder
    
    Others
    ------
    trg_sequence_prj.weight : The weights of he linear layer after the decoder (can be shared with the weights of the input Embedding layer)
    encoder.src_sequence_emb.weight The weights of he encoder embedding layer (can be shared with the weights of the decoder input Embedding layer)
    
    Methods
    -------
    forward(self, src_seq, trg_seq): Perfoms one forward pass through the transformer
    
    """

    def __init__(
            self, n_src_sequence, n_trg_sequence, src_pad_idx, trg_pad_idx,
            d_sequence_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):
        """
        n_src_sequence (int): size of the dictionary of the embedding
        n_trg_sequence (int): size of the dictionary of the embedding
        src_pad_idx (int): if specified, the entries at padding_idx do not contribute to the gradient of the source embedding
        trg_pad_idx (int): if specified, the entries at padding_idx do not contribute to the gradient of the target embedding
        d_sequence_vec (int): dimensionality of the sequence vector
        d_model (int): dimensionality of the model
        d_inner (int): dimensionality of the hidden layer in the Positionwise Feed Forward Module
        n_layers (int): the number of encoder layers 
        n_head (int): the number attention heads
        d_k (int): dimensionality of the key vector
        d_v (int): dimensionality of the value vector
        dropout (float): dropout rate
        n_position (int): Outer dimenisonality of the positional encoding
        trg_emb_prj_weight_sharing (Boolean): If True, the decoder imput embedding weights and the weights of the linear layer afetr the decoder are shared
        emb_src_trg_weight_sharing (Boolean): If True, the decoder input embedding and the encoder input embedding weights are shared
        scale_emb_or_prj (String): Either ['emb', 'prj', 'none'] Choose whether the embedidng, output prijection or none is scaled
        """
        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_sequence=n_src_sequence, n_position=n_position,
            d_sequence_vec=d_sequence_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_sequence=n_trg_sequence, n_position=n_position,
            d_sequence_vec=d_sequence_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_sequence_prj = nn.Linear(d_model, n_trg_sequence, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_sequence_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target sequence embedding & last dense layer
            self.trg_sequence_prj.weight = self.decoder.trg_sequence_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_sequence_emb.weight = self.decoder.trg_sequence_emb.weight


    def forward(self, src_seq, trg_seq):
        """Perfroms a forward pass through the transformer
        
        :param: src_seq (torch.Tensor): input sequence
        :param: trg_seq (torch.Tensor): target sequence
        
        :return: pred_seq (torch.Tensor) the predicted sequence
        """

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_sequence_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5
        
        pred_seq = seq_logit.view(-1, seq_logit.size(2))
        return pred_seq
