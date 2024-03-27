import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
from Transformer.modules import MultiHeadAttention
# orthogonal initialization weight,weight fixed, gaussian initialization of bias, bias fixed,
def orth (m,n): 
    assert m>=n, "orthogonal matrix should be full rank"
    torch.manual_seed(1)
    raw_weight = torch.randn(m,n)
    q, _ = torch.linalg.qr(raw_weight)
    weight = q[:, :n]
    return weight

def biasinitialization(input):
    mean = 0
    std = 0.01
    output = torch.randn(input) * std +mean
    return output

class TransformerDecoderLayer(nn.Module):
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

        orth_weight = orth(ffn_dim,model_dim) # 使用orth产生正交矩阵 
        self.fc1_weight = nn.Parameter(orth_weight, requires_grad=False)  # 新权重不更新 
        self.fc1_bias = nn.Parameter(biasinitialization(ffn_dim), requires_grad=False) # 高斯初始化

        self.fc2 = nn.Linear(ffn_dim, model_dim)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self,net_input: torch.Tensor,padding_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        prev_input: Optional[torch.Tensor] = None,
        prev_input_padding_mask: Optional[torch.Tensor] = None,):

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

        x = F.relu(F.linear(x, self.fc1_weight, self.fc1_bias))# 传过来新矩阵和权重并通过relu

        x = self.fc2(x)
        x = self.drop_out3(x)
        x = x + res

        if self.post_norm:
            x = self.ffn_layer_norm(x)

        return x, attn_weight


class TransformerEncoderLayer(nn.Module):
    def __init__(self, head_num, model_dim, ffn_dim, drop_out=0.1, post_norm=True):
        super().__init__()

        self.drop_out = nn.Dropout(drop_out)
        self.sa_layer_norm = nn.LayerNorm(model_dim)
        self.ffn_layer_norm = nn.LayerNorm(model_dim)
        self.post_norm = post_norm

        self.self_attn = MultiHeadAttention(head_num, model_dim)

        orth_weight = orth(ffn_dim,model_dim) # 使用orth产生正交矩阵
        self.fc1_weight = nn.Parameter(orth_weight, requires_grad=False)  # 新权重不更新
        self.fc1_bias = nn.Parameter(biasinitialization(ffn_dim), requires_grad=False) # 高斯初始化

        self.fc2 = nn.Linear(ffn_dim, model_dim)

    def forward(self,net_input: torch.Tensor,padding_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,):

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
        
        x = F.relu(F.linear(x, self.fc1_weight, self.fc1_bias))# 传过来新矩阵和权重并通过relu

        x = self.fc2(x)
        x = self.drop_out(x)
        x = x + res

        if self.post_norm:
            x = self.ffn_layer_norm(x)

        return x, attn_weight
