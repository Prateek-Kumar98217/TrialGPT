from dataclasses import dataclass
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Register a static mask for efficiency
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.size()

        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)
        query = query.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        key = key.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        value = value.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        attn = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        y = attn @ value
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.perceptron = MultiLayerPerceptron(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.perceptron(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50304  # round of 50257 to a multiple of 64
    dropout: float = 0.05

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            word_token_embeddings=nn.Embedding(config.vocab_size, config.n_embd),
            word_position_embeddings=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.word_token_embeddings.weight = self.lm_head.weight

        # Optimized weight initialization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("Number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.word_position_embeddings.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        token_embeddings = self.transformer.word_token_embeddings(idx)
        position_embeddings = self.transformer.word_position_embeddings(pos)
        x = self.transformer.drop(token_embeddings + position_embeddings)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])  # Note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, k_top=None):
        for _ in range(max_new_tokens):
            idx_cont = idx if idx.size(1) < self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cont)
            logits = logits[:, -1, :] / temperature

            if k_top is not None:
                v, _ = torch.topk(logits, min(k_top, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
