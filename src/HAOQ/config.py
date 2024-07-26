from dataclasses import dataclass

@dataclass
class Config:
    debug: bool = True
    layer_norm_epsilon: float = 1e-5
    vocab_size: int = 50257
    d_model: int = 768
    d_mlp: int = 3072
    local_attn_window: int = 128
    global_attn_window: int = 512
    ortho_update_freq: int = 100
    n_ctx: int = 1024
    n_heads: int = 12
    n_layers: int = 2