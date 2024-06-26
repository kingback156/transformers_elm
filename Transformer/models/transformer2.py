import torch
import torch.nn as nn
from Transformer.modules import TransformerEncoderLayer, TransformerDecoderLayer
from Transformer.data import WordDict
from Transformer.handle import remove_bpe
from typing import Optional
import math


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

class TransformerEncoder(nn.Module):
    def __init__(self, model_dim, ffn_dim, head_num, encoder_layers, post_norm=True):
        super().__init__()

        self.layers = nn.ModuleList()
        if not post_norm:
            self.layer_norm = nn.LayerNorm(model_dim)

        self.layers.extend(
            [
                TransformerEncoderLayer(
                    head_num, model_dim, ffn_dim, post_norm=post_norm
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(
        self,
        net_input: torch.Tensor,
        padding_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):

        x = net_input

        for layer in self.layers:
            x, _ = layer(x, padding_mask, attn_mask)

        if hasattr(self, "layer_norm"):
            x = self.layer_norm(x)

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, model_dim, ffn_dim, head_num, decoder_layers, post_norm=True):
        super().__init__()

        self.layers = nn.ModuleList()

        if not post_norm:
            self.layer_norm = nn.LayerNorm(model_dim)

        self.layers.extend(
            [
                TransformerDecoderLayer(
                    head_num, model_dim, ffn_dim, post_norm=post_norm
                )
                for _ in range(decoder_layers)
            ]
        )

    def forward(
        self,
        dex: torch.Tensor,
        padding_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        prev_input: Optional[torch.Tensor] = None,
        prev_input_padding_mask: Optional[torch.Tensor] = None,
    ):

        x = dex

        future_mask = self.__generate_future_mask(x)
        if attn_mask == None:
            attn_mask = future_mask
        else:
            attn_mask = attn_mask.logical_or(future_mask)

        for layer in self.layers:
            x, _ = layer(
                x, padding_mask, attn_mask, prev_input, prev_input_padding_mask
            )

        if hasattr(self, "layer_norm"):
            x = self.layer_norm(x)

        return x

    def __generate_future_mask(self, net_input):
        attn_mask = torch.triu(
            net_input.new_ones(
                (net_input.shape[0], net_input.shape[1], net_input.shape[1]),
                dtype=torch.bool,
            ),
            diagonal=1,
        )  # Future mask

        return attn_mask

class Transformer(nn.Module):
    def __init__(self,vocab_info: WordDict,model_dim,ffn_dim,head_num,encoder_layers,decoder_layers,share_embeddings=True,post_norm=True,share_decoder_embedding=True,**kwargs):
        super().__init__()
        self.padding_idx = vocab_info.padding_idx
        self.share_embeddings = share_embeddings
        self.model_dim = model_dim
        self.vocab_info = vocab_info
        vocab_size = vocab_info.vocab_size
        if share_embeddings:
            self.embedding = Embedding(vocab_size, model_dim, padding_idx=self.padding_idx)
        else:
            self.encoder_emb = Embedding(vocab_size[0], model_dim, padding_idx=self.padding_idx)
            self.decoder_emb = Embedding(vocab_size[1], model_dim, padding_idx=self.padding_idx)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.encoder = TransformerEncoder(model_dim, ffn_dim, head_num, encoder_layers, post_norm=post_norm, **kwargs)
        self.decoder = TransformerDecoder(model_dim, ffn_dim, head_num, decoder_layers, post_norm=post_norm, **kwargs)

        self.fc = nn.Linear(model_dim, vocab_size if isinstance(vocab_size, int) else vocab_size[1])

        nn.init.xavier_uniform_(self.fc.weight)

        if share_decoder_embedding:
            if share_embeddings:
                self.fc.weight = self.embedding.weight
            else:
                self.fc.weight = self.decoder_emb.weight

    def forward(self, input_tokens, output_tokens) -> torch.Tensor:

        en_padding_mask = input_tokens.eq(self.padding_idx)

        if self.share_embeddings:
            x = self.embedding(input_tokens)
        else:
            x = self.encoder_emb(input_tokens)
        x = x * math.sqrt(self.model_dim)

        pos = self.__generate_pos_matrix(x)
        x = x + pos 
        x = self.dropout1(x)

        encoder_out = self.encoder(x, en_padding_mask)

        de_padding_mask = output_tokens.eq(self.padding_idx)

        if self.share_embeddings:
            dex = self.embedding(output_tokens)
        else:
            dex = self.decoder_emb(output_tokens)
        dex = dex * math.sqrt(self.model_dim)

        pos = self.__generate_pos_matrix(dex)
        dex = dex + pos
        dex = self.dropout2(dex)

        decoder_out = self.decoder(dex,de_padding_mask,prev_input=encoder_out,prev_input_padding_mask=en_padding_mask,)

        predict = self.fc(decoder_out)

        return predict

    def __generate_pos_matrix(self, x: torch.Tensor):
        pos = [
            [
                math.cos(pos / (10000 ** ((i - 1) / self.model_dim)))
                if i & 1
                else math.sin(pos / (10000 ** (i / self.model_dim)))
                for i in range(x.shape[-1])
            ]
            for pos in range(x.shape[-2])
        ]

        pos = x.new_tensor(pos, requires_grad=False)

        return pos

    @torch.no_grad()
    def inference(self, source, target=None, beam_size=5, device: torch.device = ...):
        source = self.vocab_info.tokenize(source)
        source = [self.vocab_info.bos_idx] + source + [self.vocab_info.eos_idx]

        if target != None:
            target = self.vocab_info.tokenize(target)
            target = [self.vocab_info.bos_idx] + target[:5]
            source = torch.tensor(source).unsqueeze(0).to(device)
            target = torch.tensor(target).unsqueeze(0).to(device)
            predict = self.forward(source, target)
            predict = -predict.log_softmax(-1)
            _, predict_tokens = predict.topk(1, -1, largest=False)
            predict_tokens = predict_tokens.view(1, -1)
            return

        source = torch.tensor(source).unsqueeze(0).to(device)  # 1 x L

        net_output = torch.tensor([[self.vocab_info.bos_idx]]).to(device)  # 1 x 1

        predict = self.forward(source, net_output)  #  1 x 1 x vocab_size
        predict = -predict.log_softmax(-1)

        predict_prob, predict_tokens = predict.topk(
            beam_size, -1, largest=False
        )  # 1 x 1 x beam_size

        net_output = torch.cat(
            [net_output.expand((beam_size, 1)), predict_tokens.view(beam_size, 1)],
            dim=-1,
        )  # beam_size x 2
        total_prob = predict_prob.view(-1)
        source = source.expand((beam_size, -1))

        while True:
            predict = self.forward(source, net_output)[:, -1, :].reshape(beam_size, -1)
            predict = -predict.log_softmax(-1)  # beam_size x vocab_size

            predict_prob, predict_tokens = predict.topk(
                beam_size, -1, largest=False
            )  # beam_size x beam_size

            net_output = (
                net_output.unsqueeze(1)
                .expand((beam_size, beam_size, net_output.shape[-1]))
                .reshape(beam_size * beam_size, -1)
            )  # beam_size*beam_size x L
            total_prob = (
                total_prob.unsqueeze(1).expand((beam_size, beam_size)).reshape(-1)
            )  # beam_size*beam_size
            predict_tokens = predict_tokens.view(beam_size * beam_size, 1)
            predict_prob = predict_prob.view(-1)

            net_output = torch.cat(
                [net_output, predict_tokens], dim=-1
            )  # beam_size*beam_size x L+1
            total_prob = total_prob + predict_prob  # beam_size*beam_size

            _, net_output_topk = total_prob.topk(
                beam_size, dim=-1, largest=False
            )  # beam_size

            net_output = net_output.index_select(0, net_output_topk)  # beam_size x L+1
            total_prob = total_prob.index_select(0, net_output_topk)
            sentences = self.vocab_info.detokenize(net_output)
            # print("\n".join([(remove_bpe(sent)) for sent in sentences]))

            for sentence in sentences:
                last_token = sentence.split()[-1]
                if last_token == self.vocab_info.idx2word(self.vocab_info.eos_idx):
                    return " ".join(sentence.split()[1:-1])
                elif len(sentence.split()) > 200:
                    return " ".join(sentence.split()[1:-1])
    
    # ===================收集在transformers中所有fc2的参数=========================
    def collect_fc2_params(self):
        fc2_params = []
        for layer in self.encoder.layers:
            fc2_params.append(layer.fc2.weight)
            fc2_params.append(layer.fc2.bias)
        for layer in self.decoder.layers:
            fc2_params.append(layer.fc2.weight)
            fc2_params.append(layer.fc2.bias)
        return fc2_params
     # ===================收集在transformers中所有fc2的参数=========================
    

    