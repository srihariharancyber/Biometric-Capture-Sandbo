import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequential biometric tokens.
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class SelfAttentionBlock(nn.Module):
    """
    Transformer Encoder Block that explicitly returns per-head self-attention weights.
    """
    def __init__(self, d_model: int = 64, nhead: int = 4, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # x: [B, T, D]
        # MultiheadAttention with average_attn_weights=False returns [B, nhead, T, T]
        attn_out, attn_weights = self.mha(
            query=x, key=x, value=x,
            need_weights=True,
            average_attn_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        if return_attn:
            return x, attn_weights
        return x


class BiometricSelfAttentionNet(nn.Module):
    """
    Self-Attention Deep Network for Continuous Behavioral Biometrics Authentication.
    Processes sequences of keystroke and mouse kinematic tokens into a compact biometric embedding.
    """
    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        embed_dim: int = 32,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.embed_dim = embed_dim
        
        # 1. Multi-modal Token Projection Layer
        self.token_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Positional Encoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=256)
        
        # 3. Stacked Transformer Self-Attention Blocks
        self.layers = nn.ModuleList([
            SelfAttentionBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 4. Temporal Attention Pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=1,
            batch_first=True
        )
        
        # 5. Metric Learning Biometric Latent Embedding
        self.embed_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, embed_dim)
        )
        
        # 6. Optional Classification Head for multi-user identity classification
        self.classifier = nn.Linear(embed_dim, num_classes)

    def extract_embedding(self, x: torch.Tensor, return_attention: bool = False):
        """
        Extract normalized biometric embedding and attention weight matrices.
        x: [Batch, SeqLen, InputDim]
        """
        B, T, _ = x.shape
        h = self.token_proj(x)
        h = self.pos_encoder(h)
        
        all_attn_weights = []
        for i, layer in enumerate(self.layers):
            if return_attention:
                h, attn = layer(h, return_attn=True)
                all_attn_weights.append(attn) # [B, nhead, T, T]
            else:
                h = layer(h, return_attn=False)
                
        # Attention Pooling over time
        q = self.pool_query.repeat(B, 1, 1) # [B, 1, d_model]
        pooled_out, _ = self.pool_attn(query=q, key=h, value=h) # [B, 1, d_model]
        pooled = pooled_out.squeeze(1) # [B, d_model]
        
        # Latent projection & L2 normalization
        raw_emb = self.embed_head(pooled)
        normalized_emb = F.normalize(raw_emb, p=2, dim=-1)
        
        if return_attention:
            return normalized_emb, all_attn_weights
        return normalized_emb

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if return_attention:
            emb, attn = self.extract_embedding(x, return_attention=True)
            logits = self.classifier(emb)
            return logits, emb, attn
        else:
            emb = self.extract_embedding(x, return_attention=False)
            logits = self.classifier(emb)
            return logits, emb


class TripletBiometricLoss(nn.Module):
    """
    Triplet margin loss for behavioral biometrics deep metric learning.
    Encourages sequences from the legitimate user to cluster together,
    while pushing imposter/attacker sequences away beyond margin alpha.
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # Distance metric: 1 - cosine similarity
        pos_dist = 1.0 - F.cosine_similarity(anchor, positive, dim=-1)
        neg_dist = 1.0 - F.cosine_similarity(anchor, negative, dim=-1)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
