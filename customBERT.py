import torch
import torch.nn as nn
import math
from transformers.activations import gelu


# -------------------------------------------------
# Embeddings
# -------------------------------------------------

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def embedding_single(self, input_embeds):

        batch_size, seq_length, _ = input_embeds.size()
        device = input_embeds.device

        # Position ids
        position_ids = torch.arange(seq_length, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)

        # Position + token embeddings
        positions = self.position_embeddings(position_ids)

        embeddings = input_embeds + positions
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

    def forward(self, embed1, embed2, embed3):

        out1 = self.embedding_single(embed1)
        out2 = self.embedding_single(embed2)
        out3 = self.embedding_single(embed3)

        return out1, out2, out3
    
# -------------------------------------------------
# Projection for Residual Stream
# -------------------------------------------------
    
class TripleSubspaceProjection(nn.Module):
    def __init__(self, input_dim=768, subspace_dim=256):
        super().__init__()

        # Three independent projection matrices
        self.proj1 = nn.Linear(input_dim, subspace_dim, bias=False)
        self.proj2 = nn.Linear(input_dim, subspace_dim, bias=False)
        self.proj3 = nn.Linear(input_dim, subspace_dim, bias=False)

    def forward(self, embed1, embed2, embed3):
        """
        embedX: (batch_size, seq_length, 768)
        returns:
            projX: (batch_size, seq_length, 256)
        """

        out1 = self.proj1(embed1)
        out2 = self.proj2(embed2)
        out3 = self.proj3(embed3)

        combined = torch.cat([out1, out2, out3], dim=-1)

        return combined
    
# -------------------------------------------------
# Self Attention
# -------------------------------------------------

class BertSelfAttention0(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.ModuleList([
            nn.Linear(config.hidden_size, self.head_dim)
            for _ in range(self.num_heads)
        ])
        self.key = nn.ModuleList([
            nn.Linear(config.hidden_size, self.head_dim)
            for _ in range(self.num_heads)
        ])
        self.value = nn.ModuleList([
            nn.Linear(config.hidden_size, self.head_dim)
            for _ in range(self.num_heads)
        ])

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, embeds1, embeds2, embeds3):  

        head_outputs = []

        for h in range(self.num_heads):

            inputs = 0
            if h < 4:
                inputs = embeds1
            elif h < 8:
                inputs = embeds2
            elif h < 12:
                inputs = embeds3 

            Q = self.query[h](inputs)  
            K = self.key[h](inputs)   
            V = self.value[h](inputs)

            scores = torch.matmul(Q, K.transpose(-1, -2))
            scores = scores / math.sqrt(self.head_dim)

            probs = torch.softmax(scores, dim=-1)
            probs = self.dropout(probs)

            context = torch.matmul(probs, V)

            head_outputs.append(context)

        context = torch.cat(head_outputs, dim=-1)

        return context

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim
        block_dim = config.hidden_size // 3

        self.query = nn.ModuleList([
            nn.Linear(block_dim, self.head_dim)
            for _ in range(config.num_attention_heads)
        ])
        self.key = nn.ModuleList([
            nn.Linear(block_dim, self.head_dim)
            for _ in range(config.num_attention_heads)
        ])
        self.value = nn.ModuleList([
            nn.Linear(block_dim, self.head_dim)
            for _ in range(config.num_attention_heads)
        ])

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, inputs):

        block_dim = inputs.size(-1) // 3

        x1 = inputs[..., :block_dim]
        x2 = inputs[..., block_dim:2*block_dim]
        x3 = inputs[..., 2*block_dim:]

        head_outputs = []
        heads_per_modality = self.num_heads // 3

        for h in range(self.num_heads):

            if h < heads_per_modality:
                x = x1
            elif h < 2 * heads_per_modality:
                x = x2
            else:
                x = x3

            Q = self.query[h](x)
            K = self.key[h](x)
            V = self.value[h](x)

            scores = torch.matmul(Q, K.transpose(-1, -2))
            scores = scores / math.sqrt(self.head_dim)
            probs = torch.softmax(scores, dim=-1)
            probs = self.dropout(probs)

            context = torch.matmul(probs, V)
            head_outputs.append(context)

        context = torch.cat(head_outputs, dim=-1)
        return context
    
class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.block_dim = config.hidden_size // 3

        self.num_modality_heads = self.num_heads // 3
        self.half_modality_heads = self.num_modality_heads // 2

        self.query = nn.ModuleList([
            nn.Linear(self.block_dim, self.head_dim)
            for _ in range(config.num_attention_heads)
        ])
        self.key = nn.ModuleList([
            nn.Linear(self.block_dim, self.head_dim)
            for _ in range(config.num_attention_heads)
        ])
        self.value = nn.ModuleList([
            nn.Linear(self.block_dim, self.head_dim)
            for _ in range(config.num_attention_heads)
        ])

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def attend(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        return torch.matmul(probs, V)

    def forward(self, inputs):

        block_dim = inputs.size(-1) // 3

        x = [
            inputs[..., :block_dim],                    
            inputs[..., block_dim:2*block_dim],         
            inputs[..., 2*block_dim:]                  
        ]

        head_outputs = []

        for h in range(self.num_heads):

            # Determine query modality
            q_mod = h // self.num_modality_heads

            # Determine which half inside modality
            inner_idx = h % self.num_modality_heads
            half = inner_idx // self.half_modality_heads

            # Determine key modality (cross only)
            if half == 0:
                k_mod = (q_mod + 1) % 3
            else:
                k_mod = (q_mod + 2) % 3

            Q = self.query[h](x[q_mod])
            K = self.key[h](x[k_mod])
            V = self.value[h](x[k_mod])

            head_outputs.append(self.attend(Q, K, V))

        return torch.cat(head_outputs, dim=-1)

class BlockDiagonalLinear(nn.Module):
    def __init__(self, in_size, out_size, block_size=256, bias=True):
        super().__init__()

        self.block_size_in = block_size
        self.num_blocks = in_size // block_size
        self.block_size_out = out_size // self.num_blocks

        self.weight = nn.Parameter(
            torch.randn(self.num_blocks, block_size, self.block_size_out)
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.num_blocks, self.block_size_out)
            )
        else:
            self.bias = None

    def forward(self, x):

        original_shape = x.shape
        hidden_size = original_shape[-1]

        # Flatten everything except hidden dimension
        x = x.view(-1, hidden_size)

        # Reshape into blocks
        x = x.view(-1, self.num_blocks, self.block_size_in)

        # Block-wise matmul
        out = torch.einsum('bnd,ndh->bnh', x, self.weight)

        if self.bias is not None:
            out = out + self.bias

        # Flatten blocks back
        out = out.reshape(out.shape[0], -1)

        # Restore original dimensions
        new_hidden = out.shape[-1]
        out = out.view(*original_shape[:-1], new_hidden)

        return out
    
class BlockLayerNorm(nn.Module):
    def __init__(self, hidden_size, block_size=256, eps=1e-12):
        super().__init__()
        
        self.block_size = block_size
        self.num_blocks = hidden_size // block_size
        self.eps = eps
        
        # Separate gamma and beta per block
        self.weight = nn.Parameter(
            torch.ones(self.num_blocks, block_size)
        )
        self.bias = nn.Parameter(
            torch.zeros(self.num_blocks, block_size)
        )

    def forward(self, x):
        # x: (batch, seq_len, hidden_size) OR (batch, hidden_size)
        original_shape = x.shape
        
        # Merge all dims except last
        x = x.view(-1, self.num_blocks, self.block_size)
        
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        x = x * self.weight + self.bias
        
        return x.view(original_shape)

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = BlockDiagonalLinear(config.hidden_size, config.hidden_size, block_size=256)
        self.LayerNorm = BlockLayerNorm(
            config.hidden_size,
            block_size=256,
            eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# -------------------------------------------------
# Feed Forward
# -------------------------------------------------

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = BlockDiagonalLinear(config.hidden_size, config.intermediate_size, block_size=256)

    def forward(self, hidden_states):
        return gelu(self.dense(hidden_states))


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = BlockDiagonalLinear(config.intermediate_size, config.hidden_size, block_size=1024)
        self.LayerNorm = BlockLayerNorm(
            config.hidden_size,
            block_size=256,
            eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# -------------------------------------------------
# Encoder Layer
# -------------------------------------------------

class BertLayer0(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.ModuleDict({
            "self": BertSelfAttention0(config),
            "output": BertSelfOutput(config)
        })
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, input_embeds1, input_embeds2, input_embeds3, residual_stream):
        attn_output = self.attention["self"](input_embeds1, input_embeds2, input_embeds3)
        hidden_states = self.attention["output"](attn_output, residual_stream)

        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output(intermediate_output, hidden_states)

        return layer_output

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.ModuleDict({
            "self": BertSelfAttention(config),
            "output": BertSelfOutput(config)
        })
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states):
        attn_output = self.attention["self"](hidden_states)
        hidden_states = self.attention["output"](attn_output, hidden_states)

        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output(intermediate_output, hidden_states)

        return layer_output
    
class BertXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.ModuleDict({
            "cross": BertCrossAttention(config),
            "output": BertSelfOutput(config)
        })
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states):
        attn_output = self.attention["cross"](hidden_states)
        hidden_states = self.attention["output"](attn_output, hidden_states)

        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output(intermediate_output, hidden_states)

        return layer_output

# -------------------------------------------------
# Full BERT
# -------------------------------------------------

class CustomBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embeddings = BertEmbeddings(config)
        self.subspace_proj = TripleSubspaceProjection()

        self.encoder = nn.ModuleDict({
            "layer": nn.ModuleList([
                BertLayer0(config) if i == 0
                else BertLayer(config) if i <= 4
                else BertXLayer(config)
                for i in range(config.num_hidden_layers)
            ])
        })

        self.pooler = nn.ModuleDict({
            "dense": BlockDiagonalLinear(config.hidden_size, config.hidden_size, block_size=256)
        })
        self.pooler_activation = nn.Tanh()

        self.classifier = nn.Linear(config.hidden_size, 10)

    def forward(self, input_embeds1, input_embeds2, input_embeds3):

        input_embeds1, input_embeds2, input_embeds3 = self.embeddings(input_embeds1, input_embeds2, input_embeds3)
        residual_stream = self.subspace_proj(input_embeds1, input_embeds2, input_embeds3)

        hidden_states = self.encoder["layer"][0](input_embeds1, input_embeds2, input_embeds3, residual_stream)

        for layer_module in self.encoder["layer"][1:]:
            hidden_states = layer_module(hidden_states)

        cls_token = hidden_states.mean(dim=1)
        pooled_output = self.pooler_activation(
            self.pooler["dense"](cls_token)
        )

        logits = self.classifier(pooled_output)

        return logits