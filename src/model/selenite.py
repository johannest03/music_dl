import jax
import flax.linen as nn
from params import MAX_SEQUENCE_LENGTH

class Selenite(nn.Module):
    vocab_size: int
    d_model: int = 128
    d_ff: int = 512
    n_heads: int = 4
    n_layers: int = 2
    
    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.d_model)

        self.pos_emb = nn.Embed(MAX_SEQUENCE_LENGTH, self.d_model)  

        self.transformer_blocks = [
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff
            ) for _ in range(self.n_layers)
        ]
        self.layernorm = nn.LayerNorm()
        self.projection = nn.Dense(self.vocab_size)

    @nn.compact
    def __call__(self, x, rng=None, train=True):
        
        # Input embedding
        x = self.embedding(x)
        
        # Positional encoding (simple learnable)
        seq_len = x.shape[1]
        positions = jax.numpy.arange(seq_len)[None, :]  # shape (1, seq_len)
        pos_emb = self.pos_emb(positions)
        x = x + pos_emb

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, rng, train=train)

        # Output projection
        logits = self.projection(x)
        return logits
    
    
class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    def setup(self):
        self.attention = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
        )
        self.mlp = nn.Sequential([
            nn.Dense(self.d_ff),
            nn.relu,
            nn.Dense(self.d_model)
        ])
        self.layernorm1 = nn.LayerNorm()
        self.layernorm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, rng, train=True):
        # Self-attention block
        attn_output = self.attention(self.layernorm1(x))
        attn_output = self.dropout(attn_output, rng=rng, deterministic=not train)
        x = x + attn_output

        # Feed-forward block
        mlp_output = self.mlp(self.layernorm2(x))
        mlp_output = self.dropout(mlp_output, rng=rng, deterministic=not train)
        x = x + mlp_output

        return x