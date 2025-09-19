import jax
import flax.linen as nn
from params import MAX_SEQUENCE_LENGTH
import jax.numpy as jnp

class Selenite(nn.Module):
    vocab_size: int
    pad_token_id: int
    d_model: int = 256
    d_ff: int = 1024
    n_heads: int = 8
    n_layers: int = 10
    dropout_rate: float = 0.2

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.d_model)
        self.dropout = nn.Dropout(rate=self.dropout_rate)

        self.transformer_blocks = [
            TransformerRoPEBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate
            ) for _ in range(self.n_layers)
        ]
        self.layernorm = nn.LayerNorm()
        self.projection = lambda x: x @ self.embedding.embedding.T

    @nn.compact
    def __call__(self, x, rng=None, train=True):
        # x is token IDs: (b, seq_len)
        padding_mask = (x != self.pad_token_id).astype(jnp.float32)  # (b, seq_len)

        # Input embedding
        x = self.embedding(x)

        x = self.dropout(x, rng=rng, deterministic=not train)
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, rng, train=train, padding_mask=padding_mask)

        # Output projection
        x = self.layernorm(x)
        logits = self.projection(x)
        return logits

class TransformerRoPEBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float

    def setup(self):
        self.attention = RoPEAttention(
            num_heads=self.n_heads,
            head_dim=self.d_model // self.n_heads,
            max_seq_len=MAX_SEQUENCE_LENGTH,
            dropout_rate=self.dropout_rate
        )
        
        self.mlp = SwiGLU(d_ff=self.d_ff, d_model=self.d_model)
        self.layernorm1 = nn.LayerNorm()
        self.layernorm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, rng, train=True, padding_mask=None):
        seq_len = x.shape[1]
        # Create causal mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        # Create combined mask: causal AND padding
        if padding_mask is not None:
            combined_mask = causal_mask * padding_mask[:, None, :] * padding_mask[:, :, None]  # (b, seq_len, seq_len)
        else:
            combined_mask = causal_mask  # (seq_len, seq_len), will broadcast
        combined_mask = combined_mask[:, None, :, :]  # (b, 1, seq_len, seq_len) for attention

        # Self-attention block
        attn_output = self.attention(self.layernorm1(x), rng=rng, mask=combined_mask, deterministic=not train)
        attn_output = self.dropout(attn_output, rng=rng, deterministic=not train)
        x = x + attn_output

        # Feed-forward block
        mlp_output = self.mlp(self.layernorm2(x))
        mlp_output = self.dropout(mlp_output, rng=rng, deterministic=not train)
        x = x + mlp_output

        return x
    
class RoPEAttention(nn.Module):
    num_heads: int
    head_dim: int
    max_seq_len: int
    dropout_rate: float
    
    def setup(self):
        self.qkv = nn.Dense(self.num_heads * self.head_dim * 3)
        self.out = nn.Dense(self.num_heads * self.head_dim)

        self.dropout = nn.Dropout(rate=self.dropout_rate)
        cos, sin = self._rope_freqs(self.head_dim, self.max_seq_len)
        self.cos = self.variable("constants", "cos", lambda: cos)
        self.sin = self.variable("constants", "sin", lambda: sin)
        
        
    @nn.compact
    def __call__(self, x, rng, mask=None, deterministic=False):
        b, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = jnp.clip(qkv, -1e4, 1e4)  # Clip after dense projection
        qkv = qkv.reshape(b, seq_len, self.num_heads, 3 * self.head_dim)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = jnp.swapaxes(q, 1, 2)  # (b, num_heads, seq_len, head_dim)
        k = jnp.swapaxes(k, 1, 2)  # (b, num_heads, seq_len, head_dim)
        v = jnp.swapaxes(v, 1, 2)  # (b, num_heads, seq_len, head_dim)
        
        # Clip q, k, v early
        q = jnp.clip(q, -1e4, 1e4)
        k = jnp.clip(k, -1e4, 1e4)
        v = jnp.clip(v, -1e4, 1e4)
        
        # Apply rotary to q, k
        cos = self.cos.value[:seq_len]  # (seq_len, dim/2)
        sin = self.sin.value[:seq_len]  # (seq_len, dim/2)
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)
        
        # Clip after rotary (prevents amplification)
        q = jnp.clip(q, -1e4, 1e4)
        k = jnp.clip(k, -1e4, 1e4)
        
        att_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(self.head_dim)
        if mask is not None:
            att_weights = jnp.where(mask > 0, att_weights, -1e10)
        att_weights = jnp.clip(att_weights, -1e4, 1e4)  # Reinforce
        attn_scores = nn.softmax(att_weights, axis=-1)
        attn_scores = self.dropout(attn_scores, rng=rng, deterministic=deterministic)
        out = jnp.einsum('bhqk,bhkd->bhqd', attn_scores, v)
        out = jnp.clip(out, -1e4, 1e4)  # Clip attention output
        out = jnp.swapaxes(out, 1, 2).reshape(b, seq_len, -1)
        out = self.out(out)
        out = jnp.clip(out, -1e4, 1e4)  # Clip final output
        return out
    
    def _rope_freqs(self, dim: int, max_seq_len: int, base: float = 10000.0):
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2) / dim))
        t = jnp.arange(max_seq_len)
        freqs = jnp.einsum('i , j -> i j', t, inv_freq)  # (seq_len, dim/2)
        return jnp.cos(freqs), jnp.sin(freqs)
    
    def _rotate_half(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return jnp.stack((-x2, x1), axis=-1).reshape(x.shape)
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        # q, k: (b, heads, seq_len, dim)
        # cos, sin: (seq_len, dim/2)
        cos = jnp.repeat(cos[None, None, :, :], 2, axis=-1).reshape(1, 1, cos.shape[0], -1)
        sin = jnp.repeat(sin[None, None, :, :], 2, axis=-1).reshape(1, 1, sin.shape[0], -1)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k
    
class SwiGLU(nn.Module):
    d_ff: int
    d_model: int
    
    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.d_ff)(x)
        gate = nn.swish(gate)  # Swish activation
        up = nn.Dense(self.d_ff)(x)
        out = gate * up  # Element-wise multiplication
        out = nn.Dense(self.d_model)(out)
        return out