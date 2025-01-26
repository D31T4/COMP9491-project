import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Optional

from NRI.models.encoder import Encoder, LSTMHiddenState
from NRI.models.decoder import Decoder, DISTRIBUTION_DIM
from NRI.models.agent import AgentEmbedding
from NRI.models.traffic_signal import TrafficSignalSeq2SeqModel
from NRI.models.utils import MLP, GraphInfo, create_graph, avg_logits, ema_logits, node2edge

from SinD.dataset.dataset import TRAJECTORY_FEATURE_DIM

class DynamicNeuralRelationalInference(nn.Module):
    '''
    dynamic neural relational inference
    '''
    
    def __init__(
        self,
        hid_dim: int,
        n_edges: int,
        traffic_signal_model_hid_dim: int = 32,
        traffic_signal_model_num_layers: int = 2,
        encoder_rnn_hid_dim: Optional[int] = None,
        decoder_rnn_hid_dim: Optional[int] = None,
        do_prob: float = 0.0,
        dgvae: bool = False,
        readout_head_dim: Optional[int] = None,
        edge_embedding_dim: Optional[int] = None
    ):
        '''
        Args:
        ---
        - hid_dim: hidden state dim
        - n_edges: no. of edge types
        - traffic_signal_model_hid_dim: traffic signal model hidden state dim
        - traffic_signal_model_num_layers: traffic signal model no. of rnn layers
        - rnn_hid_dim: encoder rnn hidden state dim
        - do_prob: dropout prob.
        - dgvae: use dG-VAE arhitecture if `True`
        '''
        super().__init__()

        if encoder_rnn_hid_dim is None:
            encoder_rnn_hid_dim = hid_dim
        
        if decoder_rnn_hid_dim is None:
            decoder_rnn_hid_dim = hid_dim

        # embed observation
        self.observation_embedding = MLP(
            in_dim=TRAJECTORY_FEATURE_DIM,
            hid_dim=hid_dim,
            out_dim=hid_dim,
            do_prob=do_prob
        )

        # agent embedding
        self.agent_embedding = AgentEmbedding(
            dim=hid_dim,
            do_prob=do_prob
        )

        self.encoder_initial_hidden_mlp = MLP(
            in_dim=2 * hid_dim,
            hid_dim=hid_dim,
            out_dim=2 * encoder_rnn_hid_dim,
            do_prob=do_prob
        )

        self.decoder_initial_hidden_mlp = MLP(
            in_dim=hid_dim,
            hid_dim=hid_dim,
            out_dim=decoder_rnn_hid_dim,
            do_prob=do_prob
        )

        self.init_trajectory_embedding_modules(hid_dim, traffic_signal_model_hid_dim, do_prob)

        # forecast traffic signal
        self.traffic_signal_model = TrafficSignalSeq2SeqModel(
            hid_dim=traffic_signal_model_hid_dim,
            num_layers=traffic_signal_model_num_layers,
            do_prob=do_prob,
            autoregression=False
        )

        self.encoder = Encoder(
            in_dim=hid_dim,
            hid_dim=hid_dim,
            n_edges=n_edges,
            do_prob=do_prob,
            rnn_hid_dim=encoder_rnn_hid_dim,
            rnn_num_layers=1,
            dgvae=dgvae,
            readout_head_dim=readout_head_dim,
            edge_embedding_dim=edge_embedding_dim
        )

        self.decoder = Decoder(
            in_dim=hid_dim,
            hid_dim=decoder_rnn_hid_dim,
            n_edges=n_edges,
            do_prob=do_prob,
            ignore_edge0=True,
            dgvae=dgvae,
            readout_head_dim=readout_head_dim,
            edge_embedding_dim=edge_embedding_dim,
        )

    def init_trajectory_embedding_modules(self, hid_dim: int, traffic_signal_model_hid_dim: int, do_prob: float):
        self.input_mixer = MLP(
            in_dim=hid_dim + traffic_signal_model_hid_dim,
            hid_dim=hid_dim,
            out_dim=hid_dim,
            do_prob=do_prob
        )

    def compute_initial_hidden_state(
        self, 
        agent_features: torch.FloatTensor,
        agent_types: torch.IntTensor,
        graph_info: GraphInfo,
    ) -> tuple[LSTMHiddenState, torch.FloatTensor]:
        '''
        compute initial hidden state

        Args:
        ---
        - agent_features: [..., A, agent_feat_dim]
        - agent_types: [..., A]
        - graph_info

        Returns:
        - lstm hidden state: [1, batch_size * E, dim]
        - gru hidden state: [..., A, dim]
        '''
        agent_embeddings = self.agent_embedding.forward(agent_features, agent_types)
        
        rnn_hid_dim = self.encoder.forward_rnn.hidden_size

        lstm_hidden = agent_embeddings[..., :rnn_hid_dim]
        lstm_cell = agent_embeddings[..., :2 * rnn_hid_dim]
        
        interaction_embeddings = node2edge(agent_embeddings, graph_info[0], graph_info[1], temporal=False)
        encoder_hidden = self.encoder_initial_hidden_mlp.forward(interaction_embeddings)

        lstm_hidden = encoder_hidden[..., :rnn_hid_dim].reshape(1, -1, rnn_hid_dim).contiguous()
        lstm_cell = encoder_hidden[..., rnn_hid_dim:].reshape(1, -1, rnn_hid_dim).contiguous()

        decoder_hidden = self.decoder_initial_hidden_mlp.forward(agent_embeddings)

        return (lstm_hidden, lstm_cell), decoder_hidden

    def sample_edges(
        self,
        edge_logits: torch.FloatTensor,
        edge_masks: torch.BoolTensor,
        hard: bool,
        sample: bool,
        gumbel_temp: float = 0.5,
        average: Optional[Literal['avg', 'ema']] = None,
        ema_alpha: Optional[float] = None,
    ):
        '''
        sample edges from encoder logits

        Args:
        ---
        - edge_logits: [..., E, t, n_edges]
        - edge_masks: [..., E, t]
        - hard: hard edge types
        - sample: use Gumbel-Softmax sampling
        - average: Average method. Use `avg` for average; `ema` for exponential moving average
        - ema_alpha: parameter for ema
        '''
        match average:
            case 'avg':
                edge_logits = avg_logits(edge_logits, edge_masks)
            case 'ema':
                assert ema_alpha is not None
                edge_logits = ema_logits(edge_logits, ema_alpha, edge_masks)
            case _:
                if average is not None:
                    raise ValueError(f'unrecognized average_logits value: {average}')
            
        probs = torch.zeros_like(edge_logits)

        if sample:
            probs[edge_masks] = F.gumbel_softmax(edge_logits[edge_masks], tau=gumbel_temp, hard=hard, dim=-1)
        else:
            probs[edge_masks] = edge_logits[edge_masks].softmax(dim=-1)

            if hard:
                hard_probs = torch.zeros_like(probs).scatter_(-1, probs.argmax(dim=-1, keepdim=True), 1.0)
                probs = hard_probs - probs.detach() + probs

        probs = probs.masked_fill(~edge_masks[..., None], 0)

        return probs

    def compute_trajectory_embedding(
        self, 
        trajectories: torch.FloatTensor, 
        signals: torch.FloatTensor,
        temporal: bool
    ) -> torch.FloatTensor:
        '''
        compute trajectory embedding

        Args:
        ---
        - trajectories: trajectory embedding [..., A, (t), dim]
        - signals: signal embedding [..., (t), dim]
        - temporal: existence of temporal axis

        Returns:
        ---
        - mixed embeddings
        '''
        repeat_args = [1] * trajectories.dim()
        
        if temporal:
            signals = signals[..., None, :, :]
            repeat_args[-3] = trajectories.size(-3)
        else:
            signals = signals[..., None, :]
            repeat_args[-2] = trajectories.size(-2)

        signals = signals.repeat(*repeat_args)

        out = self.input_mixer.forward(
            torch.concat([trajectories, signals], dim=-1)
        )

        return out



    def predict_future(
        self,
        trajectories: torch.FloatTensor,
        trajectory_mask: torch.BoolTensor,
        agent_feats: torch.FloatTensor,
        agent_types: torch.IntTensor,
        signals: torch.IntTensor,
        n_steps: int,
        graph_info: Optional[GraphInfo] = None,
        put_nan: bool = False,
        ema_alpha: Optional[float] = None,
        output_mode: Optional[Literal['point', 'distribution']] = 'point',
        true_signals: Optional[torch.IntTensor] = None
    ):
        '''
        predict future trajectories

        Args:
        ---
        - trajectories: past trajectory [..., A, t, dim]
        - trajectory_mask: [..., A, t]
        - agent_feats: agent features [..., A, dim]
        - agent_types: agent classes [..., A]
        - signals: traffic signals [..., t]
        - n_steps: no. of steps into the future to be predicted
        - graph_info
        - put_nan: use nan as padding in output
        - ema_alpha: use ema if provided
        - output_mode: `point | distribution`. Return trajectory if `point`; return distributions if `distribution`

        Returns:
        ---
        - output: [..., A, dim]
        - edge_types: [..., E, n_steps]
        - edge_masks: [..., E, n_steps]
        - predicted_signals: [..., n_steps]
        '''
        num_agents = trajectories.size(-3)
        obs_len = trajectories.size(-2)
        
        if graph_info is None:
            graph_info = create_graph(num_agents).to(device=trajectories.device)

        num_edges = graph_info[0].size(0)

        if true_signals is None:
            predicted_signals = self.traffic_signal_model.most_probable_outcome(signals, n_steps)
        else:
            predicted_signals = true_signals    
            
        combined_signals = torch.cat([signals, predicted_signals], dim=-1)

        # compute embeddings
        signal_embedding = self.traffic_signal_model.embedding.forward(combined_signals)

        # trajectory embedding
        trajectory_embeddings = self.compute_trajectory_embedding(
            self.observation_embedding.forward(trajectories),
            signal_embedding[..., :obs_len, :],
            temporal=True
        )
        
        encoder_hidden, decoder_hidden = self.compute_initial_hidden_state(
            agent_feats,
            agent_types,
            graph_info
        )
        
        edge_logits, encoder_hidden, edge_masks, edge_feats = self.encoder.burn_in(
            trajectory_embeddings[..., :-1, :],
            trajectory_mask[..., :-1],
            graph_info,
            hidden_state=encoder_hidden,
            put_nan=put_nan
        )

        edge_probs = self.sample_edges(
            edge_logits,
            edge_masks,
            hard=True,
            sample=False,
            average='ema' if ema_alpha is not None else None,
            ema_alpha=ema_alpha
        )

        _, decoder_hidden = self.decoder.burn_in(
            trajectory_embeddings[..., :-1, :],
            trajectories[..., :-1, :],
            trajectory_mask[..., :-1],
            edge_probs,
            edge_masks,
            graph_info,
            put_nan=put_nan,
            edge_features=edge_feats,
            zero_edge_probs=False,
            hidden_state=decoder_hidden,
        )

        prev_observation = trajectories[..., -1, :]
        prev_observation_embedding = trajectory_embeddings[..., -1, :]
        prev_edge_logits = edge_logits[..., -1, :]
        prev_edge_masks = edge_masks[..., -1]

        all_predicted = torch.zeros((*trajectories.shape[:-2], n_steps, TRAJECTORY_FEATURE_DIM if output_mode == 'point' else DISTRIBUTION_DIM), device=trajectories.device)
        all_edge_types = torch.zeros((*trajectories.shape[:-3], num_edges, n_steps, self.encoder.n_edges), device=trajectories.device)
        all_edge_masks = torch.zeros((*trajectories.shape[:-3], num_edges, n_steps), device=trajectories.device)

        for timestamp in range(n_steps):
            edge_logits, encoder_hidden, edge_masks, edge_feats = self.encoder.predict_next_step(
                prev_observation_embedding,
                trajectory_mask[..., -1],
                encoder_hidden,
                graph_info,
                put_nan=put_nan
            )

            # ema
            if ema_alpha is not None:
                ema_mask = prev_edge_masks & edge_masks
                edge_logits[ema_mask] = (1 - ema_alpha) * prev_edge_logits[ema_mask] + ema_alpha * edge_logits[ema_mask]
            
            edge_probs = self.sample_edges(
                edge_logits,
                edge_masks,
                hard=True,
                sample=False,
            )

            assert edge_probs[edge_masks].isnan().any() == False

            predicted_distribution, decoder_hidden = self.decoder.predict_next_step(
                prev_observation_embedding,
                prev_observation,
                trajectory_mask[..., -1],
                edge_probs,
                edge_masks,
                decoder_hidden,
                graph_info,
                put_nan=False,
                edge_features=edge_feats,
                zero_edge_probs=False
            )

            prev_observation = self.decoder.most_probable_trajectory(predicted_distribution)
            
            if timestamp < n_steps - 1:
                prev_observation_embedding = self.compute_trajectory_embedding(
                    self.observation_embedding.forward(prev_observation),
                    signal_embedding[..., obs_len + timestamp, :],
                    temporal=False
                )
                
            prev_edge_logits = edge_logits
            prev_edge_masks = edge_masks

            if output_mode == 'point':
                all_predicted[..., timestamp, :] = prev_observation
            else:
                all_predicted[..., timestamp, :] = predicted_distribution

            all_edge_types[..., timestamp, :] = edge_probs
            all_edge_masks[..., timestamp] = edge_masks

        return all_predicted, all_edge_types, all_edge_masks, predicted_signals

    def training_loss(
        self,
        trajectories: torch.FloatTensor,
        trajectory_mask: torch.BoolTensor,
        agent_feats: torch.FloatTensor,
        agent_types: torch.IntTensor,
        graph_info: list[GraphInfo],
        signals: torch.IntTensor,
        obs_len: int,
        pred_len: int,
        include_decoder_burn_in: bool = False,
        put_nan: bool = True,
        debug: bool = False,
        avg_logits: Optional[Literal['avg', 'ema']] = None,
        ema_alpha: Optional[float] = None,
        teacher_forcing_steps: int = float('inf')
    ):
        '''
        Compute training loss. Accepts nested tensors and padded tensors.
        Inputs must not contain nan values.
        
        Args:
        ---
        - trajectories: [B, A[i], t, trajectory_dim]
        - trajectory_mask: [B, A[i], t]
        - agent_feats: agent features [B, A[i], feat_dim]
        - agent_types: agent classes [B, A[i]]
        - graph_info
        - signals: traffic signals [B, t]
        - obs_len: observation length
        - pred_len: predicted length
        - signal_loss_weight: signal prediction loss weight
        - encoder_loss_weight: encoder loss weight
        - decoder_loss_weight: decoder loss weight
        - decoder_burn_in_loss_weight: decoder burn-in loss weight. Default `0`.
        - put_nan: use nan as padding in output. Use `False` if use autoregressively with encoder.
        - debug: debug mode
        - avg_logits: logits mode
        - ema_alpha: ema param

        Returns:
        ---
        - loss
        - predicted_trajectories: [B, A[i], pred_len, dim]
        - edge_logits: [B, E[i], pred_len, n_edges]
        - edge_masks: [B, E[i], pred_len]
        - predicted_signals: [B, pred_len]
        '''
        batch_size = trajectories.size(0)

        predicted_signals, signal_prediction_loss = self.traffic_signal_model.training_loss(
            signals, 
            obs_len, 
            pred_len
        )

        # [B, t, dim]
        signal_embedding = self.traffic_signal_model.embedding.forward(signals)

        trajectory_embeddings: torch.FloatTensor = [None] * batch_size

        encoder_initial_hidden: list[LSTMHiddenState] = [None] * batch_size
        decoder_initial_hidden: list[torch.FloatTensor] = [None] * batch_size

        # compute embeddings and initial hidden state
        for batch_index in range(batch_size):
            trajectory_embeddings[batch_index] = self.compute_trajectory_embedding(
                self.observation_embedding.forward(trajectories[batch_index]),
                signal_embedding[batch_index],
                temporal=True
            )
            
            encoder_initial_hidden[batch_index], decoder_initial_hidden[batch_index] = self.compute_initial_hidden_state(
                agent_feats[batch_index],
                agent_types[batch_index],
                graph_info[batch_index]
            )

        if trajectories.is_nested:
            trajectory_embeddings = torch.nested.as_nested_tensor(trajectory_embeddings)
        else:
            trajectory_embeddings = torch.stack(trajectory_embeddings, dim=0)

        # encoder
        encoder_loss, edge_logits, edge_masks, edge_features = self.encoder.training_loss(
            trajectory_embeddings, 
            trajectory_mask, 
            graph_info,
            initial_hidden_state=encoder_initial_hidden,
            put_nan=put_nan,
            debug=debug
        )

        # sample edges
        edge_probs: list[torch.FloatTensor] = [None] * batch_size

        for batch_index in range(batch_size):
            edge_probs[batch_index] = self.sample_edges(
                edge_logits[batch_index],
                edge_masks[batch_index],
                hard=False,
                sample=True,
                gumbel_temp=0.5,
                average=avg_logits,
                ema_alpha=ema_alpha
            )

            if debug:
                assert not edge_probs[batch_index].isnan().any()

        if trajectories.is_nested:
            edge_probs = torch.nested.as_nested_tensor(edge_probs)
        else:
            edge_probs = torch.stack(edge_probs, dim=0)

        # decoder
        def compute_embedding(x: torch.FloatTensor, batch_index: int, timestamp: int) -> torch.FloatTensor:
            '''
            Args:
            ---
            - x: [A, dim]
            - batch_index
            - timestmap
            '''
            return self.compute_trajectory_embedding(
                self.observation_embedding.forward(x),
                signal_embedding[batch_index][..., timestamp, :],
                temporal=False
            )            
        
        decoder_loss, burn_in_loss, predictions = self.decoder.training_loss(
            trajectory_embeddings,
            trajectories,
            trajectory_mask,
            edge_probs,
            edge_masks,
            graph_info,
            obs_len=obs_len,
            pred_len=pred_len,
            compute_embedding=compute_embedding,
            put_nan=put_nan,
            debug=debug,
            calc_burn_in_loss=include_decoder_burn_in,
            zero_edge_probs=False,
            edge_features=edge_features,
            initial_hidden_state=decoder_initial_hidden,
            teacher_forcing_steps=teacher_forcing_steps
        )


        loss = signal_prediction_loss + encoder_loss + decoder_loss

        if include_decoder_burn_in > 0.0:
            loss += burn_in_loss

        assert not loss.isnan().any()
        
        return signal_prediction_loss, encoder_loss, decoder_loss, burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals
    
    def batch_training_loss(
        self,
        trajectories: torch.FloatTensor,
        trajectory_mask: torch.BoolTensor,
        agent_feats: torch.FloatTensor,
        agent_types: torch.IntTensor,
        graph_info: GraphInfo,
        signals: torch.IntTensor,
        obs_len: int,
        pred_len: int,
        include_decoder_burn_in: bool = False,
        put_nan: bool = True,
        debug: bool = False,
        avg_logits: Optional[Literal['avg', 'ema']] = None,
        ema_alpha: Optional[float] = None,
        teacher_forcing_steps: int = float('inf')
    ):
        '''
        Batch compute training loss. Only accepts padded tensors.
        Input must not contains nan values.

        Args:
        ---
        - trajectories: [B, A[i], t, trajectory_dim]
        - trajectory_mask: [B, A[i], t]
        - agent_feats: agent features [B, A[i], feat_dim]
        - agent_types: agent classes [B, A[i]]
        - graph_info
        - signals: traffic signals [B, t]
        - obs_len: observation length
        - pred_len: predicted length
        - include_decoder_burn_in: compute decoder burn-in. Default `False`.
        - put_nan: use nan as padding in output. Use `False` if use autoregressively with encoder.
        - debug: debug mode
        - avg_logits: logits mode
        - ema_alpha: ema param

        Returns:
        ---
        - loss
        - predicted_trajectories: [B, A[i], pred_len, dim]
        - edge_logits: [B, E[i], pred_len, n_edges]
        - edge_masks: [B, E[i], pred_len]
        - predicted_signals: [B, pred_len]
        '''
        predicted_signals, signal_prediction_loss = self.traffic_signal_model.training_loss(
            signals, 
            obs_len, 
            pred_len
        )

        # [B, t, dim]
        signal_embedding = self.traffic_signal_model.embedding.forward(signals)

        # [B, A[i], t, dim]
        trajectory_embeddings = self.compute_trajectory_embedding(
            self.observation_embedding.forward(trajectories),
            signal_embedding,
            temporal=True
        )
        
        encoder_initial_hidden, decoder_initial_hidden = self.compute_initial_hidden_state(
            agent_feats,
            agent_types,
            graph_info,
        )

        encoder_loss, edge_logits, edge_masks, edge_features = self.encoder.batch_training_loss(
            trajectory_embeddings, 
            trajectory_mask, 
            graph_info,
            initial_hidden_state=encoder_initial_hidden,
            put_nan=put_nan,
            debug=debug
        )

        edge_probs = self.sample_edges(
            edge_logits,
            edge_masks,
            hard=False,
            sample=True,
            gumbel_temp=0.5,
            average=avg_logits,
            ema_alpha=ema_alpha
        )

        if debug:
            assert not edge_probs.isnan().any()

        def compute_embedding(x: torch.FloatTensor, timestamp: int) -> torch.FloatTensor:
            '''
            Args:
            ---
            - x: [A, dim]
            - batch_index
            - timestmap
            '''
            return self.compute_trajectory_embedding(
                self.observation_embedding.forward(x),
                signal_embedding[..., timestamp, :],
                temporal=False
            )
        
        decoder_loss, decoder_burn_in_loss, predictions = self.decoder.batch_training_loss(
            trajectory_embeddings,
            trajectories,
            trajectory_mask,
            edge_probs,
            edge_masks,
            graph_info,
            obs_len=obs_len,
            pred_len=pred_len,
            compute_embedding=compute_embedding,
            put_nan=put_nan,
            debug=debug,
            calc_burn_in_loss=include_decoder_burn_in,
            zero_edge_probs=False,
            edge_features=edge_features,
            initial_hidden_state=decoder_initial_hidden,
            teacher_forcing_steps=teacher_forcing_steps
        )


        loss = signal_prediction_loss + encoder_loss + decoder_loss
        
        if include_decoder_burn_in > 0.0:
            loss += include_decoder_burn_in * decoder_burn_in_loss

        assert not loss.isnan().any()
        
        return signal_prediction_loss, encoder_loss, decoder_loss, decoder_burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals
