"""
GPT-2 (124M) Model Implementation
Following Andrej Karpathy's "Let's build GPT-2"

Key Design Decisions:
1. Variable naming matches HuggingFace GPT-2 for weight loading compatibility
2. Pre-Norm architecture (LayerNorm before Attention/MLP)
3. GELU activation with tanh approximation (matches original TF implementation)
4. Flash Attention via F.scaled_dot_product_attention
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# GPT2Config: Model Configuration
# =============================================================================

@dataclass
class GPT2Config:
    """Configuration for GPT-2 model.
    
    Default values correspond to GPT-2 (124M) "small" variant.
    """
    vocab_size: int = 50257       # GPT-2 vocabulary size (BPE tokens)
    n_positions: int = 1024       # Maximum sequence length
    n_embd: int = 768             # Embedding dimension
    n_layer: int = 12             # Number of transformer blocks
    n_head: int = 12              # Number of attention heads
    n_inner: Optional[int] = None # Inner dimension of MLP (default: 4 * n_embd)
    dropout: float = 0.0          # Dropout probability (0.0 for inference)
    bias: bool = True             # Use bias in Linear layers and LayerNorm
    
    def __post_init__(self):
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd


# =============================================================================
# CausalSelfAttention: Multi-Head Self-Attention with Causal Mask
# =============================================================================

class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention.
    
    Naming matches HuggingFace GPT-2:
    - c_attn: Combined QKV projection (Conv1D in TF, Linear in PyTorch)
    - c_proj: Output projection
    
    Note: Original GPT-2 uses TensorFlow Conv1D with shape (input, output).
    PyTorch Linear uses shape (output, input). This requires transpose when
    loading pretrained weights.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Combined QKV projection: (n_embd) -> (3 * n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection: (n_embd) -> (n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        B, T, C = x.size()  # batch_size, seq_len, n_embd
        
        # Combined QKV projection and split
        qkv = self.c_attn(x)  # (B, T, 3 * n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention: (B, T, n_embd) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Flash Attention with causal mask
        # F.scaled_dot_product_attention automatically handles:
        # 1. Causal masking (is_causal=True)
        # 2. Efficient memory usage
        # 3. Dropout during training
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        
        # Reshape back: (B, n_head, T, head_dim) -> (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection with residual dropout
        y = self.resid_dropout(self.c_proj(y))
        
        return y


# =============================================================================
# MLP: Feed-Forward Network with GELU Activation
# =============================================================================

class MLP(nn.Module):
    """Feed-Forward Network (MLP) with GELU activation.
    
    Architecture: Linear -> GELU -> Linear -> Dropout
    
    Naming matches HuggingFace GPT-2:
    - c_fc: First linear projection (expand)
    - c_proj: Second linear projection (contract)
    
    Note: Uses GELU with tanh approximation to match original TF implementation.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        
        # Expand: (n_embd) -> (n_inner), typically 4x
        self.c_fc = nn.Linear(config.n_embd, config.n_inner, bias=config.bias)
        # GELU with tanh approximation (matches original GPT-2)
        self.gelu = nn.GELU(approximate='tanh')
        # Contract: (n_inner) -> (n_embd)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd, bias=config.bias)
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# =============================================================================
# Block: Transformer Block with Pre-Norm Architecture
# =============================================================================

class Block(nn.Module):
    """Transformer Block with Pre-Norm architecture.
    
    Structure:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    
    Naming matches HuggingFace GPT-2:
    - ln_1: LayerNorm before attention
    - attn: CausalSelfAttention
    - ln_2: LayerNorm before MLP
    - mlp: Feed-forward network
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        
        # Pre-Norm: LayerNorm before Attention
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        # Pre-Norm: LayerNorm before MLP
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5, bias=config.bias)
        self.mlp = MLP(config)
        
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Pre-Norm Attention with residual connection
        x = x + self.attn(self.ln_1(x), attention_mask)
        # Pre-Norm MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x


# =============================================================================
# GPT: Full GPT-2 Model
# =============================================================================

class GPT(nn.Module):
    """GPT-2 Language Model.
    
    Architecture:
        Input -> Token Embedding + Position Embedding -> N x Block -> LayerNorm -> LM Head
    
    Naming matches HuggingFace GPT-2 (using ModuleDict for transformer):
    - transformer.wte: Token embedding
    - transformer.wpe: Position embedding
    - transformer.h: List of transformer blocks
    - transformer.ln_f: Final layer norm
    - lm_head: Language model head (output projection, no bias)
    
    Note: wte (word token embedding) and lm_head share weights (weight tying).
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        
        # Transformer components using ModuleDict (matches HF naming)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),      # Token embedding
            wpe = nn.Embedding(config.n_positions, config.n_embd),    # Position embedding
            drop = nn.Dropout(config.dropout),                         # Embedding dropout
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer blocks
            ln_f = nn.LayerNorm(config.n_embd, eps=1e-5, bias=config.bias),     # Final LayerNorm
        ))
        
        # Language Model Head (no bias, weight tied with wte)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: Share weights between token embedding and output projection
        # This is standard practice in GPT-2 and reduces model size
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled initialization to residual projections
        # This helps with training stability for deep networks
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        # Report number of parameters
        print(f"GPT-2 model initialized with {self.get_num_params()/1e6:.2f}M parameters")
        
    def _init_weights(self, module):
        """Initialize weights following GPT-2 paper."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get the number of parameters.
        
        Args:
            non_embedding: If True, exclude position embeddings from count.
                          Token embeddings are counted since they're tied to lm_head.
        
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            labels: Optional labels for language modeling loss
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if labels provided, else None
        """
        device = input_ids.device
        B, T = input_ids.size()
        
        assert T <= self.config.n_positions, f"Sequence length {T} exceeds maximum {self.config.n_positions}"
        
        # Create position indices
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)
        
        # Token and position embeddings
        tok_emb = self.transformer.wte(input_ids)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)        # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.transformer.h:
            x = block(x, attention_mask)
            
        # Final LayerNorm
        x = self.transformer.ln_f(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Compute logits
            logits = self.lm_head(x)  # (B, T, vocab_size)
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        else:
            # Only compute logits for the last position for efficient inference
            # During training, compute all logits
            logits = self.lm_head(x)  # (B, T, vocab_size)
        
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_name: str = 'gpt2') -> 'GPT':
        """Load pretrained GPT-2 weights from HuggingFace.
        
        CRITICAL: TensorFlow Conv1D weights have shape (input, output).
        PyTorch Linear weights have shape (output, input).
        The following layers require transposition:
        - attn.c_attn.weight
        - attn.c_proj.weight
        - mlp.c_fc.weight
        - mlp.c_proj.weight
        
        Args:
            model_name: HuggingFace model name ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
            
        Returns:
            GPT model with pretrained weights
        """
        from transformers import GPT2LMHeadModel
        
        print(f"Loading pretrained weights from '{model_name}'...")
        
        # Model configuration mapping
        config_mapping = {
            'gpt2':        GPT2Config(n_layer=12, n_head=12, n_embd=768),   # 124M
            'gpt2-medium': GPT2Config(n_layer=24, n_head=16, n_embd=1024),  # 350M
            'gpt2-large':  GPT2Config(n_layer=36, n_head=20, n_embd=1280),  # 774M
            'gpt2-xl':     GPT2Config(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }
        
        assert model_name in config_mapping, f"Unknown model: {model_name}"
        config = config_mapping[model_name]
        
        # Initialize our model
        model = cls(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn_mask')]
        
        # Load HuggingFace model
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        hf_sd = hf_model.state_dict()
        
        # Keys that need transposition (TF Conv1D -> PyTorch Linear)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # Copy weights with proper handling
        hf_keys = [k for k in hf_sd.keys() if not k.endswith('.attn.masked_bias') 
                   and not k.endswith('.attn.bias')]
        
        # Verify we have matching parameter counts
        assert len(hf_keys) == len(sd_keys), f"Mismatched keys: {len(hf_keys)} vs {len(sd_keys)}"
        
        for k in hf_keys:
            # Handle weight transposition for Conv1D -> Linear
            if any(k.endswith(w) for w in transposed):
                assert hf_sd[k].dim() == 2, f"Expected 2D tensor for {k}"
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k].t())
            else:
                # Direct copy for other weights
                assert hf_sd[k].shape == sd[k].shape, f"Shape mismatch for {k}: {hf_sd[k].shape} vs {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k])
        
        print(f"Successfully loaded pretrained weights!")
        return model
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate text autoregressively.
        
        Args:
            input_ids: Starting token indices of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = neutral)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling with this threshold
            
        Returns:
            Generated token indices of shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop input if it exceeds max sequence length
            idx_cond = input_ids if input_ids.size(1) <= self.config.n_positions else input_ids[:, -self.config.n_positions:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Get logits for the last position only
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, idx_next], dim=1)
        
        return input_ids


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    # Test model initialization
    print("=" * 60)
    print("Testing GPT-2 (124M) Model")
    print("=" * 60)
    
    config = GPT2Config()
    model = GPT(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    logits, loss = model(input_ids)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {config.vocab_size})")
    
    # Test with labels
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(input_ids, labels=labels)
    print(f"Loss: {loss.item():.4f}")
    
    print("\n✓ All tests passed!")

