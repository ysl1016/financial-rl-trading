import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class DeepSeekEncoderLayer(nn.Module):
    """
    DeepSeek-R1 based transformer encoder layer
    
    Extended standard transformer encoder layer optimized for financial time series data
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        layer_norm_eps: float = 1e-5,
        use_rotary_embeddings: bool = True,
        use_gated_mlp: bool = True,
        window_size: Optional[int] = None
    ):
        """
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward network inner dimension
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu')
            norm_first: True for Pre-LN, False for Post-LN
            layer_norm_eps: Layer normalization epsilon
            use_rotary_embeddings: Whether to use rotary positional embeddings
            use_gated_mlp: Whether to use gated MLP
            window_size: Local attention window size (None for global attention)
        """
        super(DeepSeekEncoderLayer, self).__init__()
        
        self.norm_first = norm_first
        self.use_rotary_embeddings = use_rotary_embeddings
        self.window_size = window_size
        
        # Self-attention module
        if use_rotary_embeddings:
            self.self_attn = RotaryMultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
        
        # Feedforward network
        if use_gated_mlp:
            self.feedforward = GatedMLP(
                d_model=d_model,
                d_ff=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                getattr(nn, activation.upper())(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Input sequence [batch_size, seq_length, d_model]
            src_mask: Attention mask (optional)
            src_key_padding_mask: Padding mask (optional)
            
        Returns:
            x: Processed sequence [batch_size, seq_length, d_model]
        """
        x = src
        
        # Create local attention mask if needed
        if self.window_size is not None and src_mask is None:
            seq_length = src.size(1)
            local_mask = create_local_attention_mask(seq_length, self.window_size).to(src.device)
            if src_mask is not None:
                src_mask = src_mask * local_mask
            else:
                src_mask = local_mask
        
        # Pre-LN or Post-LN structure
        if self.norm_first:
            # Pre-LN
            attn_output = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + attn_output
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-LN
            attn_output = self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm1(x + attn_output)
            x = self.norm2(x + self._ff_block(x))
        
        return x
    
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Self-attention block"""
        if self.use_rotary_embeddings:
            x = self.self_attn(x, attn_mask=attn_mask)
        else:
            x, _ = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
        return x
    
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feedforward block"""
        return self.feedforward(x)


class DeepSeekTransformerEncoder(nn.Module):
    """
    DeepSeek-R1 based transformer encoder
    
    Multiple encoder layers stack optimized for financial time series data
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        layer_norm_eps: float = 1e-5,
        use_rotary_embeddings: bool = True,
        use_gated_mlp: bool = True,
        use_different_window_sizes: bool = True,
        max_seq_length: int = 100
    ):
        """
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: Feedforward network inner dimension
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu')
            norm_first: Pre-LN or Post-LN structure
            layer_norm_eps: Layer normalization epsilon
            use_rotary_embeddings: Whether to use rotary positional embeddings
            use_gated_mlp: Whether to use gated MLP
            use_different_window_sizes: Whether to use different window sizes per layer
            max_seq_length: Maximum sequence length
        """
        super(DeepSeekTransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        if not use_rotary_embeddings:
            self.positional_encoding = PositionalEncoding(
                d_model=d_model,
                dropout=dropout,
                max_len=max_seq_length
            )
        
        # Encoder layers stack
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Use different window sizes if specified
            if use_different_window_sizes:
                window_size = None if i == num_layers - 1 else 2 ** (i + 2)  # 4, 8, 16, ..., None
            else:
                window_size = None
            
            layer = DeepSeekEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                layer_norm_eps=layer_norm_eps,
                use_rotary_embeddings=use_rotary_embeddings,
                use_gated_mlp=use_gated_mlp,
                window_size=window_size
            )
            self.layers.append(layer)
        
        # Final normalization layer
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Weight initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Input sequence [batch_size, seq_length, input_dim]
            mask: Attention mask (optional)
            src_key_padding_mask: Padding mask (optional)
            
        Returns:
            output: Encoded sequence [batch_size, seq_length, d_model]
        """
        # Input embedding
        x = self.input_embedding(src) * math.sqrt(self.d_model)
        
        # Positional encoding (if not using rotary)
        if not hasattr(self.layers[0], 'use_rotary_embeddings') or not self.layers[0].use_rotary_embeddings:
            x = self.positional_encoding(x)
        
        # Process through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        # Final normalization
        output = self.norm(x)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_length, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RotaryMultiheadAttention(nn.Module):
    """
    Multihead attention with rotary positional embeddings
    
    Reference: RoPE (Rotary Position Embedding) - Su et al., 2021
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super(RotaryMultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # QKV projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Rotary positional embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence [batch_size, seq_length, embed_dim]
            attn_mask: Attention mask (optional)
            
        Returns:
            output: Attention output [batch_size, seq_length, embed_dim]
        """
        batch_size, seq_length, _ = x.shape
        
        # Calculate QKV
        qkv = self.qkv_proj(x).reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_length, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply rotary positional embeddings
        q, k = self.rotary_emb(q, k, seq_length)
        
        # Calculate attention
        output = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        # Output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.out_proj(output)
        
        return output


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embeddings
    
    Reference: RoPE (Rotary Position Embedding) - Su et al., 2021
    """
    
    def __init__(self, dim: int, base: int = 10000):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", self.inv_freq)
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor [batch_size, num_heads, seq_length, head_dim]
            k: Key tensor [batch_size, num_heads, seq_length, head_dim]
            seq_len: Sequence length
            
        Returns:
            q_rot: Query with rotary embeddings applied
            k_rot: Key with rotary embeddings applied
        """
        t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Expand dimensions [1, 1, seq_length, head_dim]
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        
        # Apply rotary transformations
        q_rot = self._rotate_half(q)
        k_rot = self._rotate_half(k)
        
        q = q * cos + q_rot * sin
        k = k * cos + k_rot * sin
        
        return q, k
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the tensor's dimensions"""
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class GatedMLP(nn.Module):
    """
    Gated MLP (Gated Multi-Layer Perceptron)
    
    Reference: GLU (Gated Linear Unit) variant - Dauphin et al., 2017
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super(GatedMLP, self).__init__()
        
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.value_proj = nn.Linear(d_model, d_ff)
        self.output_proj = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.act_fn = getattr(F, activation) if hasattr(F, activation) else getattr(nn, activation.upper())()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_length, d_model]
            
        Returns:
            output: Processed tensor [batch_size, seq_length, d_model]
        """
        # GLU mechanism
        if callable(self.act_fn):
            gate_output = self.act_fn(self.gate_proj(x))
        else:
            gate_output = self.act_fn.forward(self.gate_proj(x))
            
        value_output = self.value_proj(x)
        
        # Gate * Value
        intermediate_output = gate_output * value_output
        
        # Output projection
        output = self.output_proj(intermediate_output)
        output = self.dropout(output)
        
        return output


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0
) -> torch.Tensor:
    """
    Calculate scaled dot-product attention
    
    Args:
        q: Query tensor [batch_size, num_heads, seq_length, head_dim]
        k: Key tensor [batch_size, num_heads, seq_length, head_dim]
        v: Value tensor [batch_size, num_heads, seq_length, head_dim]
        attn_mask: Attention mask (optional)
        dropout_p: Dropout probability
        
    Returns:
        output: Attention output [batch_size, num_heads, seq_length, head_dim]
    """
    d_k = q.size(-1)
    
    # Q @ K^T / sqrt(d_k)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if attn_mask is not None:
        attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
    
    # Softmax
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    # Weighted sum with values
    output = torch.matmul(attn_weights, v)
    
    return output


def create_local_attention_mask(seq_length: int, window_size: int) -> torch.Tensor:
    """
    Create mask for local attention
    
    Args:
        seq_length: Sequence length
        window_size: Attention window size
        
    Returns:
        mask: Local attention mask [seq_length, seq_length]
    """
    mask = torch.zeros(seq_length, seq_length)
    
    for i in range(seq_length):
        start = max(0, i - window_size // 2)
        end = min(seq_length, i + window_size // 2 + 1)
        mask[i, start:end] = 1
    
    return mask
