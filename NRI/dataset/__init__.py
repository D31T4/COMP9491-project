import torch

from SinD.dataset import SignalizedIntersectionDataset, SignalizedIntersectionDatasetConfig
from NRI.models.utils import create_graph, GraphInfo

class SignalizedIntersectionDatasetForNRI(SignalizedIntersectionDataset):
    def __getitem__(
        self, 
        index: int, 
        remove_future_agents: bool = True
    ):
        '''
        get item

        Args:
        ---
        - idx: index
        - remove_future_agents: remove agents not appearing in `0:config.obs_len`

        Returns:
        ---
        - trajectory: [A, T, TRAJ_DIM]
        - trajectory_mask: [A, T]
        - agent_mask: [A]
        - agent_records: [A, AGENT_DIM]
        - agent_flags: [A, FLAG_DIM]
        - signals: signal records [A, SIG_DIM]
        - graph_info: [send_edges [E], recv_edges [E], edge2node_projector [A, E]]
        '''
        (
            trajectory,
            trajectory_mask,
            agent_mask,
            agent_records,
            agent_flags,
            signals
        ) = super().__getitem__(index, remove_future_agents=remove_future_agents)

        graph_info = create_graph(agent_records.size(0))

        return (
            trajectory,
            trajectory_mask,
            agent_mask,
            agent_records,
            agent_flags,
            signals,
            graph_info
        )
    
    @staticmethod
    def collate_nested(batches) -> tuple[
        torch.FloatTensor, 
        torch.BoolTensor, 
        torch.BoolTensor, 
        torch.FloatTensor, 
        torch.IntTensor, 
        torch.IntTensor,
        list[GraphInfo]
    ]:
        return (
            torch.nested.nested_tensor([batch[0] for batch in batches]),
            torch.nested.nested_tensor([batch[1] for batch in batches]),
            torch.nested.nested_tensor([batch[2] for batch in batches]),
            torch.nested.nested_tensor([batch[3] for batch in batches]),
            torch.nested.nested_tensor([batch[4] for batch in batches]).int(),
            torch.stack([batch[5] for batch in batches]).int(),
            torch.nested.nested_tensor([batch[6] for batch in batches]),
        )
    
    @staticmethod
    def collate_padded(batches) -> tuple[
        torch.FloatTensor, 
        torch.BoolTensor, 
        torch.BoolTensor, 
        torch.FloatTensor, 
        torch.IntTensor, 
        torch.IntTensor,
        list[GraphInfo]
    ]:
        out = [
            torch.nested.nested_tensor([batch[0] for batch in batches]).to_padded_tensor(0),
            torch.nested.nested_tensor([batch[1] for batch in batches]).to_padded_tensor(0),
            torch.nested.nested_tensor([batch[2] for batch in batches]).to_padded_tensor(0),
            torch.nested.nested_tensor([batch[3] for batch in batches]).to_padded_tensor(0),
            torch.nested.nested_tensor([batch[4] for batch in batches]).to_padded_tensor(0).int(),
            torch.stack([batch[5] for batch in batches]).int(),
        ]

        out.append(create_graph(out[0].size(1)))

        return tuple(out)