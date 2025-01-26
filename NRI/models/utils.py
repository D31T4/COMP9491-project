import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

# [send_edges: [E], recv_edges: [E]]
GraphInfo = torch.LongTensor

def create_graph(
    num_nodes: int
) -> GraphInfo:
    '''
    Args:
    ---
    - num_nodes: no. of nodes

    Returns:
    ---
    - send_edges: indices of source nodes [E]
    - recv_edges: indices of target nodes [E]
    '''
    # fully connected graph without self-loops
    adj_matrix = torch.ones(num_nodes) - torch.eye(num_nodes)
    source_nodes, target_nodes = torch.where(adj_matrix)

    # matrix to project adge features to node features
    node_indices = torch.arange(num_nodes)
    edge2node_projector = (node_indices[:, None] == target_nodes[None, :]).to(torch.float32)

    return torch.stack([source_nodes, target_nodes], dim=0)

def node2edge(
    node_embedding: torch.FloatTensor, 
    send_edges: torch.LongTensor, 
    recv_edges: torch.LongTensor,
    dgvae: bool = False,
    temporal: bool = True
) -> torch.FloatTensor:
    '''
    node-to-edge operator in NRI

    Args:
    ---
    - node_embedding: [..., A, (t)?, dim]
    - send_edges: [E]
    - recv_edges: [E]
    - dgvae: use dG-VAE operator
    - temporal: existence temporal axis (t)

    Returns:
    ---
    - edge_embedding: [..., E, (t)?, dim]
    '''
    if temporal:
        send_embeddings = node_embedding[..., send_edges, :, :]
        recv_embeddings = node_embedding[..., recv_edges, :, :]
    else:
        send_embeddings = node_embedding[..., send_edges, :]
        recv_embeddings = node_embedding[..., recv_edges, :]

    if dgvae:
        # Eqn (3). in dG-VAE
        send_embeddings = send_embeddings - recv_embeddings

    return torch.cat([send_embeddings, recv_embeddings], dim=-1)

def edge2node(
    edge_embedding: torch.FloatTensor, 
    recv_edges: torch.LongTensor, 
    node_embedding: torch.FloatTensor, 
    edge_weight: torch.Tensor | None = None,
    temporal: bool = True
):
    '''
    edge-to-node operator in NRI

    Args:
    ---
    - edge_embedding: [..., E, (t), dim]
    - recv_edges
    - out: [..., A, (t), dim]. Output tensor for accumulation
    - edge_weight: [..., E, (t)]. Weight of each edge, you can put a boolean mask here as well.
    - temporal: existence of temporal axis (t)
    '''
    if edge_weight is None:
        edge_weight = torch.tensor(1.0)

    edge_embedding = edge_embedding * edge_weight[..., None]
    
    index = -3 if temporal else -2
    node_embedding.index_add_(index, recv_edges, edge_embedding)

def avg_logits(
    logits: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
):
    '''
    average logits

    Args:
    ---
    - logits: [..., N, dim]
    - mask: [..., N]

    Returns:
    ---
    - logits: [..., N, dim]
    '''
    if mask is None:
        mask = torch.ones(logits.shape[:-1], device=logits.device, dtype=torch.bool)

    logits = logits.masked_fill(~mask[..., None], 0)
    # masked average
    logits = logits.sum(dim=-2, keepdim=True) / mask.sum(dim=-1, keepdim=True).clip_(min=1)[..., None] # clip to prevent div by 0

    repeat_size = [1] * logits.dim()
    repeat_size[-2] = mask.size(-1)
    logits = logits.repeat(*repeat_size)
    
    return logits

def ema_logits(
    logits: torch.FloatTensor, 
    alpha: float,
    mask: Optional[torch.BoolTensor] = None,
):
    '''
    exponential moving average logits
    https://en.wikipedia.org/wiki/Exponential_smoothing

    Args:
    ---
    - logits: [..., N, dim]
    - alpha
    - mask: [..., N]

    Returns:
    ---
    - logits: [..., N, dim]
    '''
    assert 0.0 <= alpha and alpha <= 1.0

    if mask is None:
        mask = torch.ones(logits.shape[:-1], device=logits.device, dtype=torch.bool)

    logits = logits.masked_fill(~mask[..., None], 0)

    for timestamp in range(1, logits.size(-2)):
        current_mask = mask[..., timestamp - 1] & mask[..., timestamp]
        logits[..., timestamp, :][current_mask] = (1 - alpha) * logits[..., timestamp - 1, :][current_mask] + alpha * logits[..., timestamp, :][current_mask]

    return logits

class SinusoidalPositionalEncoding(nn.Module):
    '''
    sinusoid positional encoding from Attention is All You Need.

    stolen from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
    '''

    def __init__(self, 
        dim: int,
        max_len: int,
        do_prob: float = 0.
    ):
        '''
        Args:
        ---
        - dim: model dimension
        - max_len: max sequence length
        - do_prob: dropout probability
        '''
        super().__init__()
        
        self.dropout = nn.Dropout(p=do_prob)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))

        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Args:
        ---
        - x: [B, L, dim]

        Returns:
        ---
        - x + pe [B, L, dim]
        '''
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MLP(nn.Module):
    '''
    multi-layer perceptron block
    '''

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        do_prob: float = 0.0
    ):
        '''
        Args:
        ---
        - in_dim: input dim
        - hid_dim: hidden dim
        - out_dim: output dim
        - do_prob: dropout prob.
        '''
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ELU(inplace=True),
            nn.Dropout(do_prob),
            nn.Linear(hid_dim, out_dim),
            nn.ELU(inplace=True)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Args:
        ---
        - x: [..., in_dim]

        Returns:
        ---
        - output: [..., out_dim]
        '''
        return self.model(x)