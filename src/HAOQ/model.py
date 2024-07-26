from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from HAOQ.config import Config

class OrthogonalQueryDecomposition(nn.Module):
    """
    Decomposes the input sequence into two orthogonal query spaces:
    1. a local query space (LQS) for capturing local dependencies
    2. a global query space (GQS) for capturing long range dependencies

    The input sequence is projected onto the orthogonal bases of the LQS and GQS,
    merging the "Query Decomposition" and "Orthogonal Query Projection" into a single module.

    NOTE: the projection matrices are initialized as orthogonal.
    Without updates, the matrices will likely drift from orthogonality.
    This can be mitigated by orthogonalizing the matrices periodically.

    There are other ways to ensure orthogonality during training, such as using a custom optimizer.
    """
    def __init__(self, cfg: Config):
        super(OrthogonalQueryDecomposition, self).__init__()

        self.cfg = cfg
        self.update_freq = cfg.ortho_update_freq
        self.global_step = 0

        self.proj_local = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_model))
        self.proj_global = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_model))
        nn.init.orthogonal_(self.proj_local)
        nn.init.orthogonal_(self.proj_global)

    def orthogonalize(self):
        # QR decomposition
        q_local, _ = torch.linalg.qr(self.proj_local)
        q_global, _ = torch.linalg.qr(self.proj_global)

        # Update the projection matrices
        self.proj_local.data = q_local
        self.proj_global.data = q_global

    def forward(self, x):
        # x size: [batch_size, seq_len, d_model]

        # Orthogonalize the projection matrices
        # NOTE: this is not necessary during inference
        # this will occur during the first pass, and then every `update_freq` steps
        if self.training:
            if self.global_step % self.update_freq == 0:
                self.orthogonalize()

            self.global_step += 1

        # Project input onto orthogonal bases
        lqs = x @ self.proj_local
        gqs = x @ self.proj_global

        return lqs, gqs


class HierarchicalAttention(nn.Module):
    """
    Hierarchical Attention with Sliding Window Attention

    This module applies local and global attention mechanisms to the input sequence.
    The local attention mechanism is used to capture local dependencies, while the global attention mechanism captures long-range dependencies.
    joint attention is then computed by fusing the local and global attention outputs.

    NOTE: More than 2 attention mechanisms could be used, such as a middle-range attention mechanism.
    """
    def __init__(self, cfg: Config):
        super(HierarchicalAttention, self).__init__()
        self.cfg = cfg

        # NOTE: Using a non-standard MultiheadAttention module would allow for both
        # more flexibility *and* better opportunities for interpretability
        # For now...
        self.attn_l = nn.MultiheadAttention(self.cfg.d_model, self.cfg.n_heads, batch_first=True)
        self.attn_g = nn.MultiheadAttention(self.cfg.d_model, self.cfg.n_heads, batch_first=True)
        self.local_attn_window = cfg.local_attn_window
        self.global_attn_window = cfg.global_attn_window

    def sliding_window_attention(self, queries, attn, window_size):
        seq_len = queries.size(1)

        out = []
        # slide over the sequence with a window size
        for i in range(0, queries.size(1), window_size):
            boundary = min(i+window_size, seq_len) # handle the end of the sequence
            window = queries[:, i:boundary, :]
            attn_out, _ = attn(window, window, window)
            out.append(attn_out)

        return torch.cat(out, dim=1)

    def forward(self, lqs, gqs):
        # lqs size: (batch_size, seq_len, d_model)
        # gqs size: (batch_size, seq_len, d_model)

        # Local Attention
        local_out = self.sliding_window_attention(lqs, self.attn_l, self.local_attn_window)

        # Global Attention
        # NOTE: if wanted, this could be reduced to a single attention pass for the full context window.
        # set the size of the global attention window to the full sequence length to achieve this
        global_out = self.sliding_window_attention(gqs, self.attn_g, self.global_attn_window)

        return local_out, global_out


class QueryFusion(nn.Module):
    """
    This is the most simple fusion method, a weighted sum of the local and global attention outputs.

    We could move to a more complex fusion method, such as a multi-layer perceptron, or a gating mechanism.
    That way, interactions between the local and global attention outputs could be learned.
    """
    def __init__(self, cfg: Config):
        super(QueryFusion, self).__init__()
        self.fusion_weights = nn.Parameter(torch.ones(2))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, local_out, global_out):
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = weights[0] * local_out + weights[1] * global_out
        return self.norm(fused)


class HAOQAttention(nn.Module):
    def __init__(self, cfg: Config):
        super(HAOQAttention, self).__init__()
        self.cfg = cfg

        self.oqd = OrthogonalQueryDecomposition(cfg)
        self.attn = HierarchicalAttention(cfg)
        self.fuse = QueryFusion(cfg)

    def forward(self, x):
        # x size: (batch_size, seq_len, d_model)

        lqs, gqs = self.oqd(x)
        local_out, global_out = self.attn(lqs, gqs)
        fused = self.fuse(local_out, global_out)

        return fused

class MLP(nn.Module):
    def __init__(self, cfg, activation=F.gelu):
        super(MLP, self).__init__()

        self.cfg = cfg

        self.fc_in = nn.Linear(self.cfg.d_model, self.cfg.d_mlp)
        self.act = activation
        self.fc_out = nn.Linear(self.cfg.d_mlp, self.cfg.d_model)

    def forward(self, x):
        # x size: [batch_size, seq_len, d_model]

        pre = self.fc_in(x)
        post = self.act(pre)
        out = self.fc_out(post)

        return out


class Block(nn.Module):
    """
    Follows a similar structure to GPT2.

    Changing the ordering of layer norms may be appropriate.
    """
    def __init__(self, cfg: Config):
        super(Block, self).__init__()

        self.cfg = cfg

        self.ln_1 = nn.LayerNorm(self.cfg.d_model, eps=self.cfg.layer_norm_epsilon)
        self.attn = HAOQAttention(cfg)
        self.ln_2 = nn.LayerNorm(self.cfg.d_model, eps=self.cfg.layer_norm_epsilon)
        self.mlp = MLP(self.cfg)

    def forward(self, x):
        # x size: [batch_size, seq_len, d_model]
        residual = x
        attn_output = self.attn(self.ln_1(residual))
        residual += attn_output
        mlp_output = self.mlp(self.ln_2(residual))
        residual += mlp_output
        return residual


class HAOQModel(nn.Module):
    def __init__(self, cfg: Config):
        super(HAOQModel, self).__init__()
        self.cfg = cfg

        d_vocab = self.cfg.vocab_size
        n_ctx = self.cfg.n_ctx
        d_model = self.cfg.d_model
        n_layers = self.cfg.n_layers

        self.tok_embd = nn.Embedding(d_vocab, d_model)
        self.pos_embd = nn.Embedding(n_ctx, d_model)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(n_layers)])

        self.ln_f = nn.LayerNorm(d_model)


    def forward(self, input_ids):
        # input_seq size: [batch_size, seq_len]

        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        tok_embeds = self.tok_embd(input_ids)
        pos_embeds = self.pos_embd(position_ids)

        # this needs improvement
        hidden_states = tok_embeds + pos_embeds

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class HAOQLMHead(nn.Module):
    """
    Unembed the output sequence back into the vocabulary space.

    """
    def __init__(self, cfg: Config):
        super(HAOQLMHead, self).__init__()
        self.cfg = cfg
        self.decoder = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x):
        return self.decoder(x)


class HAOQLMHeadModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.transformer = HAOQModel(cfg)
        self.lm_head = HAOQLMHead(cfg)

    def forward(self, input_ids):
        hidden_states = self.transformer(input_ids)
        logits = self.lm_head(hidden_states)
        return logits

