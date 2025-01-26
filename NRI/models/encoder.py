import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from NRI.models.utils import MLP, GraphInfo, node2edge, edge2node

LSTMHiddenState = tuple[torch.FloatTensor, torch.FloatTensor]

class Encoder(nn.Module):
    '''
    dNRI encoder
    '''
    
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        n_edges: int,
        rnn_hid_dim: Optional[int] = None,
        rnn_num_layers: int = 1,
        do_prob: float = 0.0,
        dgvae: bool = False,
        readout_head_dim: Optional[int] = None,
        edge_embedding_dim: Optional[int] = None
    ):
        '''
        Args:
        ---
        - in_dim: input dimension
        - hid_dim: hidden state dimension
        - n_edges: no. of edge types
        - rnn_hid_dim: rnn hidden state dimension
        - do_prob: dropout prob.
        - dgvae: use dG-VAE architecture if `True`
        '''
        super().__init__()

        if rnn_hid_dim is None:
            rnn_hid_dim = hid_dim

        if readout_head_dim is None:
            readout_head_dim = hid_dim

        if edge_embedding_dim is None:
            edge_embedding_dim = hid_dim

        self.hid_dim = hid_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.n_edges = n_edges
        self.dgvae = dgvae

        # forward rnn
        self.forward_rnn = nn.LSTM(
            hid_dim, 
            rnn_hid_dim, 
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=do_prob
        )
        
        self.prior_mlp = nn.Sequential(
            MLP(rnn_hid_dim, readout_head_dim, readout_head_dim, do_prob=do_prob),
            nn.Linear(readout_head_dim, n_edges)
        )

        # backward rnn
        self.backward_rnn = nn.LSTM(
            hid_dim, 
            rnn_hid_dim, 
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=do_prob
        )
        
        self.prior_teacher_mlp = nn.Sequential(
            MLP(2 * rnn_hid_dim, readout_head_dim, readout_head_dim, do_prob=do_prob),
            nn.Linear(readout_head_dim, n_edges)
        )

        if self.dgvae:
            self.edge_feature_mlp = MLP(rnn_hid_dim, readout_head_dim, edge_embedding_dim, do_prob=do_prob)

        # node mlp
        self.node_mlp1 = MLP(in_dim, hid_dim, hid_dim, do_prob=do_prob)
        self.node_mlp2 = MLP(hid_dim, hid_dim, hid_dim, do_prob=do_prob)

        # edge mlp
        self.edge_mlp1 = MLP(hid_dim * 2, hid_dim, hid_dim, do_prob=do_prob)
        self.edge_mlp2 = MLP(hid_dim * 3, hid_dim, hid_dim, do_prob=do_prob)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
        

    def get_initial_hidden(
        self, 
        repeat: int, 
        device: Optional[torch.device] = None
    ) -> LSTMHiddenState:
        '''
        Args:
        ---
        - repeat: no. of repetitions
        - device: tensor device

        Returns:
        ---
        - LSTM hidden state
        '''
        if device is None:
            device = torch.get_default_device()

        return (
            torch.zeros((self.forward_rnn.num_layers, repeat, self.forward_rnn.hidden_size), dtype=torch.float32, device=device),
            torch.zeros((self.forward_rnn.num_layers, repeat, self.forward_rnn.hidden_size), dtype=torch.float32, device=device)
        )

    def compute_edge_embedding(
        self, 
        observation: torch.FloatTensor,
        graph_info: GraphInfo,
        edge_mask: torch.BoolTensor,
        debug: bool = False,
        temporal: bool = True
    ) -> torch.FloatTensor:
        '''
        compute edge embedding

        Args:
        ---
        - observation: [..., A, (t), dim]
        - graph_info
        - edge_mask: [..., E, (t)]
        - debug: debug mode
        - temporal: existence of temporal axis (t)
        '''
        send_edges, recv_edges = graph_info

        node_embedding = self.node_mlp1(observation)

        edge_embedding = node2edge(node_embedding, send_edges, recv_edges, dgvae=self.dgvae, temporal=temporal)
        edge_embedding = self.edge_mlp1.forward(edge_embedding)

        node_embedding = torch.zeros_like(node_embedding)
        edge2node(edge_embedding, recv_edges, node_embedding, edge_mask, temporal=temporal)
        node_embedding = self.node_mlp2.forward(node_embedding)
        
        edge_embedding = torch.concat([node2edge(node_embedding, send_edges, recv_edges, temporal=temporal), edge_embedding], dim=-1)
        edge_embedding = self.edge_mlp2.forward(edge_embedding)

        if debug:
            edge_embedding = edge_embedding.masked_fill(~edge_mask[..., None], torch.nan)
        
        return edge_embedding


    def burn_in(
        self,
        trajectories: torch.FloatTensor,
        trajectory_mask: torch.FloatTensor,
        graph_info: GraphInfo,
        hidden_state: Optional[LSTMHiddenState] = None,
        put_nan: bool = True
    ) -> tuple[
        torch.FloatTensor, 
        LSTMHiddenState,
        torch.BoolTensor,
        Optional[torch.FloatTensor]
    ]:
        '''
        Args:
        ---
        - trajectories: [..., A, t, dim]
        - trajectory_mask: [..., A, t]
        - graph_info
        - hidden_state: [rnn_num_layers, batch_size * E, rnn_hid_dim]
        - put_nan: use nan as padding in output

        Returns:
        ---
        - all_logits: [..., E, t, n_edges]
        - hidden_state: [rnn_num_layers, batch_size * E, rnn_hid_dim]
        - edge_mask: [..., E, t]
        - edge_feats: edge features [..., E, t, dim]
        '''
        num_edges = graph_info[0].size(0)
        send_edges, recv_edges = graph_info
        
        seq_len = trajectory_mask.size(-1)

        is_batch = trajectories.dim() > 3

        edge_mask = trajectory_mask[..., send_edges, :] * trajectory_mask[..., recv_edges, :]
        batch_size = trajectory_mask.size(0) if is_batch else 1

        # run gnn
        edge_embeddings = self.compute_edge_embedding(
            trajectories,
            graph_info,
            edge_mask,
            temporal=True
        )

        flattened_edge_masks = edge_mask.view(batch_size * num_edges, seq_len)
        flattened_edge_embedding = edge_embeddings.view(batch_size * num_edges, seq_len, self.hid_dim)

        forward_embedding = torch.zeros((batch_size * num_edges, seq_len, self.forward_rnn.hidden_size), device=trajectories.device)

        if hidden_state is None:
            hidden_state = self.get_initial_hidden(batch_size * num_edges, device=trajectories.device)

        for timestamp in range(seq_len):
            forward_embedding[..., timestamp:(timestamp + 1), :], next_forward_hidden = self.forward_rnn.forward(
                flattened_edge_embedding[..., timestamp:(timestamp + 1), :], 
                hidden_state
            )

            hidden_state = (
                hidden_state[0].clone(),
                hidden_state[1].clone()
            )

            hidden_state[0][:, flattened_edge_masks[..., timestamp]] = next_forward_hidden[0][:, flattened_edge_masks[..., timestamp]]
            hidden_state[1][:, flattened_edge_masks[..., timestamp]] = next_forward_hidden[1][:, flattened_edge_masks[..., timestamp]]

            
        # reshape to original
        if is_batch:
            forward_embedding = forward_embedding.reshape((batch_size, num_edges, seq_len, self.forward_rnn.hidden_size))

        all_logits: torch.FloatTensor = self.prior_mlp.forward(forward_embedding)

        if put_nan:
            all_logits = all_logits.masked_fill(~edge_mask[..., None], torch.nan)

        # edge feats for dG-VAE
        edge_features: Optional[torch.FloatTensor] = None

        if self.dgvae:
            edge_features = self.edge_feature_mlp.forward(forward_embedding)
            edge_features = edge_features.masked_fill(~edge_mask[..., None], 0)

        return all_logits, hidden_state, edge_mask, edge_features


    def predict_next_step(
        self,
        observation: torch.FloatTensor,
        observation_mask: torch.BoolTensor,
        hidden_state: Optional[LSTMHiddenState],
        graph_info: GraphInfo,
        put_nan: bool = True
    ) -> tuple[
        torch.FloatTensor,
        LSTMHiddenState,
        Optional[torch.FloatTensor]
    ]:
        '''
        Args:
        ---
        - observation: [..., A, dim]
        - observation_mask: [..., A]
        - hidden_state: [rnn_num_layer, batch * E, dim]
        - graph_info
        - put_nan: use nan as padding in output

        Returns:
        ---
        - logits: edge logits [..., E, n_edges]
        - hidden_state: edge hidden state [rnn_num_layers, batch_size * num_agents, rnn_hid_dim]
        - edge_mask: edge mask [..., E]
        - edge_feats: edge features for dG-VAE [..., E, dim]. `None` if `self.dgvae == False`
        '''
        is_batch = observation.dim() > 2

        send_edges, recv_edges = graph_info
        edge_mask = observation_mask[..., send_edges] & observation_mask[..., recv_edges]

        batch_size = observation_mask.size(0) if is_batch else 1

        if hidden_state is None:
            hidden_state = self.get_initial_hidden(batch_size * edge_mask.size(0), observation.device)

        edge_embedding = self.compute_edge_embedding(observation, graph_info, edge_mask, temporal=False)

        edge_embedding, new_hidden_state = self.forward_rnn.forward(edge_embedding.view(-1, 1, self.hid_dim), hidden_state)

        if is_batch:
            edge_embedding = edge_embedding.view(batch_size, -1, 1, self.forward_rnn.hidden_size)[..., 0, :]
        else:
            edge_embedding = edge_embedding.view(-1, 1, self.forward_rnn.hidden_size)[..., 0, :]


        inv_edge_mask = ~edge_mask
        flattened_edge_mask = inv_edge_mask.view(-1)

        new_hidden_state[0][:, flattened_edge_mask] = hidden_state[0][:, flattened_edge_mask]
        new_hidden_state[1][:, flattened_edge_mask] = hidden_state[1][:, flattened_edge_mask]


        logits: torch.FloatTensor = self.prior_mlp.forward(edge_embedding)

        if put_nan:
            logits = logits.masked_fill(inv_edge_mask[..., None], torch.nan)

        edge_features: Optional[torch.FloatTensor] = None

        if self.dgvae:
            edge_features = self.edge_feature_mlp.forward(edge_embedding)
            edge_features = edge_features.masked_fill(inv_edge_mask[..., None], 0)

        return logits, new_hidden_state, edge_mask, edge_features
    
    
    def kl_loss(
        self,
        student_logits: torch.FloatTensor,
        teacher_logits: torch.FloatTensor,
        edge_mask: torch.BoolTensor
    ) -> torch.FloatTensor:
        '''
        compute KL-divergence loss of teacher prior model and student prior model

        Args:
        ---
        - student_logits: [..., n_edges]
        - teacher_logits: [..., n_edges]
        - edge_mask: [...]

        Returns:
        ---
        - KL-divergence loss
        '''
        student_logits = student_logits[edge_mask, :].reshape(1, -1, self.n_edges).log_softmax(dim=-1)
        teacher_logits = teacher_logits[edge_mask, :].reshape(1, -1, self.n_edges).log_softmax(dim=-1)
        return F.kl_div(student_logits, teacher_logits, log_target=True, reduction='batchmean')


    def training_loss(
        self,
        trajectory: torch.FloatTensor,
        trajectory_mask: torch.BoolTensor,
        graph_info: list[GraphInfo],
        debug: bool = False,
        put_nan: bool = True,
        initial_hidden_state: Optional[list[LSTMHiddenState]] = None
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.BoolTensor,
        Optional[torch.FloatTensor]
    ]:
        '''
        Compute training loss. Accepts nested tensor or padded tensor.

        Args:
        ---
        - trajectory: [B, A[i], t, d]
        - trajectory_mask: [B, A[i], t]
        - graph_info
        - debug: debug mode
        - put_nan: put nan in invalid indices if `True`
        - initial_hidden_state: initial hidden state for RNNs

        Returns:
        ---
        - loss
        - logits: [B, E[i], t, n_edges]
        - edge_masks: [B, E[i], t]
        - edge_feats: edge features for dG-VAE [B, E[i], t, dim]
        '''
        batch_size = trajectory.size(0)

        batch_weights = torch.tensor([
            (trajectory_mask[batch_index][graph_info[batch_index][0]] & trajectory_mask[batch_index][graph_info[batch_index][1]]).sum() 
            for batch_index in range(batch_size)
        ])
        
        batch_weights = batch_weights / batch_weights.sum()

        loss = torch.tensor(0.0, device=trajectory.device)

        teacher_logits: list[torch.FloatTensor] = [None] * batch_size
        edge_masks: list[torch.BoolTensor] = [None] * batch_size

        edge_features = None

        if self.dgvae:
            edge_features: list[torch.FloatTensor] = [None] * batch_size

        for batch_index in range(batch_size):
            batch_loss, current_edge_logits, current_edge_mask, current_edge_feats = self.batch_training_loss(
                trajectory[batch_index].unsqueeze(0),
                trajectory_mask[batch_index].unsqueeze(0),
                graph_info[batch_index],
                debug=debug,
                put_nan=put_nan,
                initial_hidden_state=initial_hidden_state[batch_index] if initial_hidden_state is not None else None
            )

            teacher_logits[batch_index] = current_edge_logits[0]
            edge_masks[batch_index] = current_edge_mask[0]
            
            if self.dgvae:
                edge_features[batch_index] = current_edge_feats[0]

            loss += batch_loss * batch_weights[batch_index]

        assert not loss.isnan().any()

        if trajectory.is_nested:
            teacher_logits = torch.nested.as_nested_tensor(teacher_logits)
            edge_masks = torch.nested.as_nested_tensor(edge_masks)

            if self.dgvae:
                edge_features = torch.nested.as_nested_tensor(edge_features)

        else:
            teacher_logits = torch.stack(teacher_logits, dim=0)
            edge_masks = torch.stack(edge_masks, dim=0)

            if self.dgvae:
                edge_features = torch.stack(edge_features, dim=0)

        return loss, teacher_logits, edge_masks, edge_features
    
    def batch_training_loss(
        self,
        trajectory: torch.FloatTensor,
        trajectory_mask: torch.BoolTensor,
        graph_info: GraphInfo,
        debug: bool = False,
        put_nan: bool = True,
        initial_hidden_state: Optional[LSTMHiddenState] = None
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.BoolTensor,
        Optional[torch.FloatTensor]
    ]:
        '''
        Batch compute training loss. Accepts padded tensors only.

        Args:
        ---
        - trajectory: [B, A, t, d]
        - trajectory_mask: [B, A, t]
        - graph_info
        - debug: debug mode
        - put_nan: put nan in invalid indices if `True`
        - initial_hidden_state: initial hidden state for RNNs

        Returns:
        ---
        - loss
        - logits: [B, E, t, n_edges]
        - edge_masks: [B, E, t]
        - edge_feats: edge features for dG-VAE [B, E, t, dim]
        '''
        send_edges, recv_edges = graph_info

        num_edges = send_edges.size(0)
        batch_size = trajectory.size(0)
        seq_len = trajectory_mask.size(2)

        edge_masks = trajectory_mask[:, send_edges, :] & trajectory_mask[:, recv_edges, :]

        edge_embedding = self.compute_edge_embedding(
            trajectory,
            graph_info,
            edge_masks,
            debug=debug,
            temporal=True
        )


        flattened_edge_masks = edge_masks.view(batch_size * num_edges, seq_len)
        flattened_edge_embedding = edge_embedding.view(batch_size * num_edges, seq_len, self.hid_dim)

        forward_embedding = torch.zeros((batch_size * num_edges, seq_len, self.forward_rnn.hidden_size), device=trajectory.device)
        backward_embedding = torch.zeros((batch_size * num_edges, seq_len, self.forward_rnn.hidden_size), device=trajectory.device)


        if initial_hidden_state is None:
            initial_hidden_state = self.get_initial_hidden(batch_size * num_edges, trajectory.device)
        

        forward_hidden = initial_hidden_state
        backward_hidden = initial_hidden_state

        for timestamp in range(seq_len):
            forward_embedding[..., timestamp:(timestamp + 1), :], next_forward_hidden = self.forward_rnn.forward(
                flattened_edge_embedding[..., timestamp:(timestamp + 1), :], 
                forward_hidden
            )

            forward_hidden = (
                forward_hidden[0].clone(),
                forward_hidden[1].clone()
            )

            forward_hidden[0][:, flattened_edge_masks[..., timestamp]] = next_forward_hidden[0][:, flattened_edge_masks[..., timestamp]]
            forward_hidden[1][:, flattened_edge_masks[..., timestamp]] = next_forward_hidden[1][:, flattened_edge_masks[..., timestamp]]

            
            backward_embedding[..., (seq_len - timestamp - 1):(seq_len - timestamp), :], next_backward_hidden = self.backward_rnn.forward(
                flattened_edge_embedding[..., (seq_len - timestamp - 1):(seq_len - timestamp), :],
                backward_hidden
            )

            backward_hidden = (
                backward_hidden[0].clone(),
                backward_hidden[1].clone()
            )

            backward_hidden[0][:, flattened_edge_masks[..., seq_len - timestamp - 1]] = next_backward_hidden[0][:, flattened_edge_masks[..., seq_len - timestamp - 1]]
            backward_hidden[1][:, flattened_edge_masks[..., seq_len - timestamp - 1]] = next_backward_hidden[1][:, flattened_edge_masks[..., seq_len - timestamp - 1]]
            


        forward_embedding = forward_embedding.reshape((batch_size, num_edges, seq_len, self.forward_rnn.hidden_size))
        backward_embedding = backward_embedding.reshape((batch_size, num_edges, seq_len, self.forward_rnn.hidden_size))

        teacher_logits: torch.FloatTensor = self.prior_teacher_mlp.forward(
            torch.concat([forward_embedding, backward_embedding], dim=-1)
        )

        student_logits: torch.FloatTensor = self.prior_mlp.forward(forward_embedding)

        # put nan to invalid indices
        if put_nan:
            teacher_logits = teacher_logits.masked_fill(
                ~edge_masks[..., None],
                torch.nan
            )

        edge_features: Optional[torch.FloatTensor] = None

        if self.dgvae:
            edge_features = self.edge_feature_mlp.forward(forward_embedding)

            # zero pad instead of nan due to accumulation in decoder
            edge_features = edge_features.masked_fill(
                ~edge_masks[..., None],
                0
            )

        loss = self.kl_loss(student_logits, teacher_logits, edge_masks) / edge_masks.sum().clip_(min=1) # clip to avoid nan for empty sequences

        return loss, teacher_logits, edge_masks, edge_features
