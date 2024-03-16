import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
from Transformer.modules import MultiHeadAttention

class TransformerDecoderLayer(nn.Module):

    def decoder_svdchange(self, model_dim, ffn_dim): # All SVD operations can be done here
        temp_weight = torch.randn(model_dim, ffn_dim) # Random initialization parameters
        U, S, V = torch.svd(temp_weight, some=False)  # decompose
        self.register_buffer('U', U)  # fixed

        # self.E = nn.Parameter(torch.diag(E)) update this, but dimension error, so I do like down:
        self.diag_param = nn.Parameter(S)  # 仅对角线元素为可训练参数
        
        self.register_buffer('V', V.t()) # fixed 

        self.fc2 = nn.Linear(ffn_dim, model_dim) # ignore this fc2
        nn.init.xavier_uniform_(self.fc2.weight) # ignore this fc2
    
    
    def __init__(self, head_num, model_dim, ffn_dim, drop_out=0.1, post_norm=True):
        super().__init__()
        self.drop_out1 = nn.Dropout(drop_out)  
        self.drop_out2 = nn.Dropout(drop_out)
        self.drop_out3 = nn.Dropout(drop_out)
        self.sa_layer_norm = nn.LayerNorm(model_dim)
        self.en_de_layer_norm = nn.LayerNorm(model_dim)
        self.ffn_layer_norm = nn.LayerNorm(model_dim)
        self.post_norm = post_norm
        self.self_attn = MultiHeadAttention(head_num, model_dim)
        self.en_de_attn = MultiHeadAttention(head_num, model_dim)
        
        self.decoder_svdchange(model_dim, ffn_dim) # call function
    

    def forward(self,net_input: torch.Tensor,padding_mask: torch.Tensor,attn_mask: Optional[torch.Tensor] = None,
        prev_input: Optional[torch.Tensor] = None,prev_input_padding_mask: Optional[torch.Tensor] = None,):

        assert net_input.shape[0] == prev_input.shape[0]

        if self.post_norm:
            res, x = net_input, net_input
        else:
            res, x = net_input, self.sa_layer_norm(net_input)

        x, attn_weight = self.self_attn(x, padding_mask, attn_mask)
        x = self.drop_out1(x)
        x = res + x

        if self.post_norm:
            x = self.sa_layer_norm(x)
            res = x
        else:
            res, x = x, self.en_de_layer_norm(x)

        x, attn_weight = self.en_de_attn(x, padding_mask, None, prev_input, prev_input_padding_mask)
        x = self.drop_out2(x)
        x = res + x
        if self.post_norm:
            x = self.en_de_layer_norm(x)
            res = x
        else:
            res, x = x, self.ffn_layer_norm(x)

        E = torch.zeros(self.U.size(0), self.V.size(1), device=net_input.device)
        min_dim = min(E.size(0), E.size(1))
        E[torch.arange(min_dim), torch.arange(min_dim)] = self.diag_param

        weight = torch.mm(torch.mm(self.U, E), self.V) # Only this has been modified
        x = F.linear(net_input, weight.t()) # Only this has been modified
        x = F.relu(x) 
        x = self.fc2(x) 
        x = self.drop_out3(x) 
        x = x + res
        if self.post_norm:
            x = self.ffn_layer_norm(x)
        return x, attn_weight


class TransformerEncoderLayer(nn.Module):

    def encoder_svdchange(self, model_dim, ffn_dim):# All SVD operations can be done here
        temp_weight = torch.randn(model_dim, ffn_dim) # Random initialization parameters
        U, S, V = torch.svd(temp_weight, some=False) # decompose
        self.register_buffer('U', U) # fixed

        self.diag_param = nn.Parameter(S)# as shown above

        self.register_buffer('V', V.t()) # fixed

        self.fc2 = nn.Linear(ffn_dim, model_dim) # ignore this fc2
        nn.init.xavier_uniform_(self.fc2.weight) # ignore this fc2

    def __init__(self, head_num, model_dim, ffn_dim, drop_out=0.1, post_norm=True):
        super().__init__()
        self.drop_out = nn.Dropout(drop_out)
        self.sa_layer_norm = nn.LayerNorm(model_dim)
        self.ffn_layer_norm = nn.LayerNorm(model_dim)
        self.post_norm = post_norm
        self.self_attn = MultiHeadAttention(head_num, model_dim)

        self.encoder_svdchange(model_dim, ffn_dim) # call function

    def forward(self,net_input: torch.Tensor,padding_mask: torch.Tensor,attn_mask: Optional[torch.Tensor] = None,):
        if self.post_norm:
            res, x = net_input, net_input
        else:
            res, x = net_input, self.sa_layer_norm(net_input)

        x, attn_weight = self.self_attn(x, padding_mask, attn_mask)
        x = self.drop_out(x)
        x = res + x

        if self.post_norm:
            x = self.sa_layer_norm(x)
            res = x
        else:
            res, x = x, self.ffn_layer_norm(x)

        E = torch.zeros(self.U.size(0), self.V.size(1), device=net_input.device)
        min_dim = min(E.size(0), E.size(1))
        E[torch.arange(min_dim), torch.arange(min_dim)] = self.diag_param

        weight = torch.mm(torch.mm(self.U, E), self.V) # Only this has been modified
        x = F.linear(x, weight.t(), None) # Only this has been modified
        x = F.relu(x)
        x = self.fc2(x)
        x = self.drop_out(x)
        x = x + res
        if self.post_norm:
            x = self.ffn_layer_norm(x)
        return x, attn_weight
