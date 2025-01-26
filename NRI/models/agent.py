import torch
import torch.nn as nn

from typing import Optional

from NRI.models.utils import MLP

from SinD.dataset.type import AgentType

class AgentEmbedding(nn.Module):
    '''
    agent embedding
    '''
    
    def __init__(
        self,
        dim: int,
        do_prob: float = 0.0,
        n_classes: Optional[int] = None
    ):
        '''
        Args:
        ---
        - dim: embedding dimension
        - do_prob: dropout prob.
        - n_classes: no. of agent classes
        '''
        super().__init__()

        if n_classes is None:
            n_classes = len(AgentType)

        self.agent_type_embedding = nn.Embedding(n_classes, dim)
        
        self.agent_feat_embedding = nn.Sequential(
            nn.Linear(2, dim),
            nn.ELU(inplace=True)
        )
        
        self.agent_embedding_mlp = MLP(dim * 2, dim, dim, do_prob)

    def forward(
        self,
        agents: torch.FloatTensor,
        agent_types: torch.IntTensor
    ) -> torch.FloatTensor:
        '''
        Args:
        ---
        - agents
        - agent_types

        Returns:
        ---
        - agent embedding
        '''
        agent_feat_embedding = self.agent_feat_embedding.forward(agents)
        agent_type_embedding = self.agent_type_embedding.forward(agent_types)
        
        if agents.is_nested:
            agent_embedding = torch.nested.as_nested_tensor([
                torch.concat([
                    agent_feat_embedding[batch_index],
                    agent_type_embedding[batch_index]
                ], dim=-1)
                for batch_index in range(agents.size(0))
            ], device=agents.device)
        else:
            agent_embedding = torch.concat([
                agent_feat_embedding,
                agent_type_embedding
            ], dim=-1)

        return self.agent_embedding_mlp.forward(agent_embedding)