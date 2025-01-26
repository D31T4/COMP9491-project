import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from typing import Callable, Literal, Optional

from NRI.models.utils import GraphInfo, node2edge, edge2node

from SinD.dataset.dataset import TRAJECTORY_FEATURE_DIM

# parametrized distribution dimension
OUTPUT_DIM = 31

# distribution params dimension
DISTRIBUTION_DIM = 29

TrajectoryDistributionTuple = tuple[
    td.MultivariateNormal,
    td.MultivariateNormal,
    td.MultivariateNormal,
    td.VonMises,
    td.VonMises,
    td.MultivariateNormal,
    td.MultivariateNormal
]

class TrajectoryDistribution:
    '''
    batched distributions of trajectory
    '''
    COL_SELECTORS: list[int | list[int]] = [
        [0, 1],
        [2, 3],
        [4, 5],
        6,
        7,
        [8, 9],
        [10, 11]
    ]
    
    def __init__(self, dist: TrajectoryDistributionTuple):
        self.dist = dist

    def log_prob(self, trajectories: torch.FloatTensor):
        '''
        compute log likelihood for each feature group

        Args:
        ---
        - trajectories: [..., dim]

        Returns:
        ---
        - log-likelihood: [..., 7]
        '''
        log_p = torch.zeros((*trajectories.shape[:-1], 7), device=trajectories.device)

        for col in range(len(TrajectoryDistribution.COL_SELECTORS)):
            log_p[..., col] = self.dist[col].log_prob(trajectories[..., TrajectoryDistribution.COL_SELECTORS[col]])

        return log_p
    
    def most_probable_trajectory(self, device: Optional[torch.device] = None):
        '''
        compute most probable trajectory

        Args:
        ---
        - device: output device
        '''
        if device is None:
            device = torch.get_default_device()

        trajectories = torch.zeros((*self.dist[0].loc.shape[:-1], 12), device=device)

        for col in range(7):
            trajectories[..., TrajectoryDistribution.COL_SELECTORS[col]] = self.dist[col].mode

        return trajectories

class Decoder(nn.Module):
    '''
    dNRI decoder
    '''

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        n_edges: int,
        do_prob: float = 0.0,
        ignore_edge0: bool = True,
        dgvae: bool = False,
        readout_head_dim: Optional[int] = None,
        edge_embedding_dim: Optional[int] = None
    ):
        '''
        Args:
        ---
        - in_dim: input dim
        - hid_dim: hidden state dim
        - n_edges: no. of edge types
        - do_prob: dropout prob.
        - ignore_edge0: interpret edge type-0 as no edge
        - variance_prediction: Predict covariance matrix if `cov`. Predict Cholesky decomposition if `chol`.
        - dgvae: use dG-VAE architecture if `True`
        '''
        super().__init__()

        if readout_head_dim is None:
            readout_head_dim = hid_dim

        if edge_embedding_dim is None:
            edge_embedding_dim = hid_dim

        self.hid_dim = hid_dim

        self.ignore_edge0 = ignore_edge0
        self.n_edges = n_edges

        self.dgvae = dgvae

        # no. of modules to be inited
        n_edge_mlps = n_edges

        if ignore_edge0:
            n_edge_mlps -= 1

        if self.dgvae:
            # Eqn (6) in dG-VAE
            edge_feature_dim = 2 * hid_dim + edge_embedding_dim
        else:
            edge_feature_dim = 2 * hid_dim

        self.edge_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(edge_feature_dim, hid_dim),
                nn.Tanh(),
                nn.Dropout(do_prob),
                nn.Linear(hid_dim, hid_dim),
                nn.Tanh()
            ) 
            for _ in range(n_edge_mlps)
        ])

        # GRU
        self.input_r = nn.Linear(in_dim, hid_dim)
        self.input_i = nn.Linear(in_dim, hid_dim)
        self.input_n = nn.Linear(in_dim, hid_dim)

        self.hidden_r = nn.Linear(hid_dim, hid_dim)
        self.hidden_i = nn.Linear(hid_dim, hid_dim)
        self.hidden_h = nn.Linear(hid_dim, hid_dim)

        # output
        self.delta_layer = nn.Sequential(
            nn.Linear(hid_dim, readout_head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(do_prob),
            nn.Linear(readout_head_dim, readout_head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(do_prob),
            nn.Linear(readout_head_dim, OUTPUT_DIM)
        )

        self.init_params()


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)


    def get_initial_hidden(self, repeat: tuple[int], device: Optional[torch.device] = None):
        '''
        get initial hidden state

        Args:
        ---
        - repeat: repetitions
        - device: tensor device

        Returns:
        ---
        - initial hidden state
        '''
        if device is None:
            device = torch.get_default_device()

        return torch.zeros((*repeat, self.hid_dim), device=device)

    def compute_node_embedding(
        self,
        node_embedding: torch.FloatTensor,
        edge_probs: torch.FloatTensor,
        graph_info: GraphInfo,
        node_masks: torch.BoolTensor,
        edge_features: torch.FloatTensor = None, # not Optional for type hinting
    ):
        '''
        compute node embedding

        Args:
        ---
        - node_embedding: [B, A, dim]
        - edge_probs: categorical distribution of each edge [B, E, n_edges]. Probs should be 0s for invalid edges.
        - graph_info
        - node_masks: [B, A]
        - edge_features: edge features used in dG-VAE [B,E, dim]
        '''
        send_edges, recv_edges = graph_info
        new_node_embedding = torch.zeros_like(node_embedding)

        if not self.ignore_edge0:
            offset = 0
            start_index = 0
        else:
            offset = -1
            start_index = 1
        
        edge_embedding = node2edge(node_embedding, send_edges, recv_edges, temporal=False)

        if self.dgvae:
            edge_embedding = torch.concat([edge_embedding, edge_features], dim=-1)

        for edge_index in range(start_index, self.n_edges):
            edge2node(
                self.edge_mlps[edge_index + offset].forward(edge_embedding),
                recv_edges,
                new_node_embedding,
                edge_probs[..., edge_index],
                temporal=False
            )

        return new_node_embedding

    def burn_in(
        self,
        trajectory_embeddings: torch.FloatTensor,
        trajectories: torch.FloatTensor,
        trajectory_mask: torch.BoolTensor,
        edge_probs: torch.FloatTensor,
        edge_masks: torch.BoolTensor,
        graph_info: GraphInfo,
        put_nan: bool = True,
        debug: bool = False,
        edge_features: Optional[torch.FloatTensor] = None,
        zero_edge_probs: bool = True,
        hidden_state: Optional[torch.FloatTensor] = None,
        update_hidden_state: Optional[Callable[[torch.FloatTensor, int], torch.FloatTensor]] = None
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor
    ]:
        '''
        burn in hidden states

        Args:
        ---
        - trajectory_embeddings: [..., A, t, dim]
        - trajectories: [..., A, t, dim]
        - trajectory_mask: [..., A, t]
        - edge_probs: [..., E, t, n_edges]
        - edge_masks: [..., E, t]
        - graph_info
        - put_nan: put nan in output if `True`
        - debug: debug mode
        - edge_features: edge features used in dG-VAE [E, t, dim]
        - zero_edge_probs: put 0s in invalid mask locations if `True`

        Returns:
        ---
        - predictions: [..., A, t, dim]
        - hidden_state: [..., A, dim]
        '''
        if update_hidden_state is None:
            update_hidden_state = lambda x, *args: x

        if hidden_state is None:
            hidden_state = self.get_initial_hidden(trajectory_embeddings.shape[:-2], trajectory_embeddings.device)

        obs_len = trajectory_embeddings.size(-2)

        if zero_edge_probs:
            edge_probs = edge_probs.masked_fill(~edge_masks[..., None], 0)

        predictions = torch.zeros((*trajectory_embeddings.shape[:-2], obs_len, DISTRIBUTION_DIM), device=trajectory_embeddings.device)

        current_edge_features: torch.FloatTensor = None

        for timestamp in range(obs_len):
            if self.dgvae:
                current_edge_features = edge_features[..., timestamp, :]

            hidden_state = update_hidden_state(hidden_state, timestamp)

            predictions[..., timestamp, :], hidden_state = self.predict_next_step(
                trajectory_embeddings[..., timestamp, :],
                trajectories[..., timestamp, :],
                trajectory_mask[..., timestamp],
                edge_probs[..., timestamp, :],
                edge_masks[..., timestamp],
                hidden_state,
                graph_info,
                put_nan=False,
                debug=debug,
                edge_features=current_edge_features,
                zero_edge_probs=False,
            )

        if put_nan:
            predictions = predictions.masked_fill(~trajectory_mask[..., None], torch.nan)

        if debug:
            assert not predictions[trajectory_mask].isnan().any()

        return predictions, hidden_state


    def predict_next_step(
        self,
        observation_embedding: torch.FloatTensor,
        observation: torch.FloatTensor,
        observation_mask: torch.BoolTensor,
        edge_prob: torch.FloatTensor,
        edge_mask: torch.BoolTensor,
        hidden_state: Optional[torch.FloatTensor],
        graph_info: GraphInfo,
        put_nan: bool,
        debug: bool = False,
        edge_features: Optional[torch.FloatTensor] = None,
        zero_edge_probs: bool = True
    ):
        '''
        predict next step

        Args:
        ---
        - observation_embedding: [..., A, dim]
        - observation: [..., A, dim]
        - observation_mask: [..., A]
        - edge_prob: [..., E, n_edges]
        - edge_mask: [..., E]
        - hidden_state: [..., A, rnn_dim]
        - graph_info
        - put_nan: put nan in output
        - debug: debug mode
        - edge_features: [..., E, dim]
        - zero_edge_probs: put 0s in invalid mask locations if set to `True`

        Returns:
        ---
        - predicted distribution
        - hidden state
        '''
        if zero_edge_probs:
            edge_prob = edge_prob.masked_fill(~edge_mask[..., None], 0)

        if hidden_state is None:
            hidden_state = self.get_initial_hidden(observation_embedding.shape[:-1], observation_embedding.device)

        node_embedding = self.compute_node_embedding(
            hidden_state, 
            edge_prob, 
            graph_info, 
            observation_mask, 
            edge_features,
        )


        # GRU
        r = F.sigmoid(self.input_r(observation_embedding) + self.hidden_r(node_embedding))
        i = F.sigmoid(self.input_i(observation_embedding) + self.hidden_i(node_embedding))
        n = F.tanh(self.input_n(observation_embedding) + r * self.hidden_h(node_embedding))
        
        hidden_state = hidden_state * ~observation_mask[..., None] + ((1 - i) * n + i * hidden_state) * observation_mask[..., None]
        #hidden_state = hidden_state.clone()
        #hidden_state[observation_mask, :] = (1 - i) * n + i * hidden_state[observation_mask, :]

        delta: torch.FloatTensor = self.delta_layer.forward(hidden_state)
        
        predicted = self.predict_trajectory_distribution_params(delta)
        predicted[..., [0, 1, 5, 6, 10, 11, 15, 17, 19, 20, 24, 25]] += observation
        
        if put_nan:
            predicted = predicted.masked_fill(~observation_mask[..., None], torch.nan)

        if debug:
            assert not predicted[observation_mask, ...].isnan().any()
        
        return predicted, hidden_state


    def training_loss(
        self, 
        trajectory_embeddings: torch.FloatTensor, 
        trajectories: torch.FloatTensor,
        trajectory_masks: torch.BoolTensor,
        edge_probs: torch.FloatTensor,
        edge_masks: torch.BoolTensor,
        graph_info: list[GraphInfo],
        obs_len: int,
        pred_len: int,
        compute_embedding: Callable[[torch.FloatTensor, int, int], torch.FloatTensor],
        put_nan: bool = True,
        debug: bool = False,
        calc_burn_in_loss: bool = False,
        edge_features: Optional[torch.FloatTensor] = None,
        zero_edge_probs: bool = True,
        initial_hidden_state: Optional[torch.FloatTensor] = None,
        update_hidden_state: Optional[Callable[[torch.FloatTensor, int, int], torch.FloatTensor]] = None,
        teacher_forcing_steps: int = float('inf')
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor
    ]:
        '''
        compute training loss.

        Args:
        ---
        - trajectory_embeddings: [B, A[i], t, dim]
        - trajectories: [B, A[i], t, dim]
        - trajectory_masks: [B, A[i], t]
        - edge_probs: [B, E[i], t, n_edges]
        - edge_masks: [B, E[i], t]
        - graph_info
        - obs_len
        - pred_len
        - compute_embedding: compute embedding (observation, batch_index, timestamp) => embedding
        - put_nan: put nan in output if set to `True`. default `True`
        - debug: debug mode
        - burn_in_loss_weight: include `0 -> obs_len` section when computing NLL loss
        - edge_features: edge features used for dG-VAE
        - zero_edge_probs: put 0s in invalid mask locations in `edge_probs` if `True`.
        - initial_hidden_state: [B, A[i], dim]

        Returns:
        ---
        - NLL loss
        - parameters of predicted distributions
        '''
        if update_hidden_state is None:
            update_hidden_state = lambda x, *args: x

        batch_size = trajectory_embeddings.size(0)

        loss = torch.tensor(0.0, device=trajectory_embeddings.device)
        burn_in_loss = torch.tensor(0.0, device=trajectories.device)

        # calc batch weights
        batch_weights = torch.tensor([
            trajectory_masks[batch_index][..., obs_len:].sum()
            for batch_index in range(batch_size)
        ])

        batch_weights = batch_weights / batch_weights.sum()

        batch_burn_in_weights = torch.tensor([
            (trajectory_masks[batch_index][..., :obs_len - 1] & trajectory_masks[batch_index][..., 1:obs_len]).sum()
            for batch_index in range(batch_size)
        ])

        batch_burn_in_weights = batch_burn_in_weights / batch_burn_in_weights.sum()

        predicted_distributions: list[torch.FloatTensor] = [None] * batch_size

        current_edge_features: torch.FloatTensor = None

        for batch_index in range(batch_size):
            current_edge_masks = edge_masks[batch_index]

            if zero_edge_probs:
                current_edge_probs = edge_probs[batch_index].masked_fill(~current_edge_masks[..., None], 0)
            else:
                current_edge_probs = edge_probs[batch_index]

            # get edge features for dG-VAE
            if self.dgvae:
                current_edge_features = edge_features[batch_index].unsqueeze(0)
            
            batch_loss, batch_burn_in_loss, distributions = self.batch_training_loss(
                trajectory_embeddings[batch_index].unsqueeze(0),
                trajectories[batch_index].unsqueeze(0),
                trajectory_masks[batch_index].unsqueeze(0),
                current_edge_probs.unsqueeze(0),
                edge_masks[batch_index].unsqueeze(0),
                graph_info[batch_index],
                obs_len,
                pred_len,
                compute_embedding=lambda x, timestamp: compute_embedding(x[0], batch_index, timestamp).unsqueeze(0),
                put_nan=put_nan,
                debug=debug,
                calc_burn_in_loss=calc_burn_in_loss,
                edge_features=current_edge_features,
                zero_edge_probs=zero_edge_probs,
                initial_hidden_state=initial_hidden_state[batch_index].unsqueeze(0) if initial_hidden_state is not None else None,
                update_hidden_state=lambda x, timestamp: update_hidden_state(x[0], batch_index, timestamp).unsqueeze(0),
                teacher_forcing_steps=teacher_forcing_steps
            )

            predicted_distributions[batch_index] = distributions[0]

            loss += batch_loss * batch_weights[batch_index]
            burn_in_loss += batch_burn_in_loss * batch_burn_in_weights[batch_index]

        
        assert not loss.isnan().any()

        if trajectories.is_nested:
            predicted_distributions = torch.nested.as_nested_tensor(predicted_distributions)
        else:
            predicted_distributions = torch.stack(predicted_distributions, dim=0)

        return loss, burn_in_loss, predicted_distributions
    
    def batch_training_loss(
        self, 
        trajectory_embeddings: torch.FloatTensor, 
        trajectories: torch.FloatTensor,
        trajectory_masks: torch.BoolTensor,
        edge_probs: torch.FloatTensor,
        edge_masks: torch.BoolTensor,
        graph_info: GraphInfo,
        obs_len: int,
        pred_len: int,
        compute_embedding: Callable[[torch.FloatTensor, int], torch.FloatTensor],
        put_nan: bool = True,
        debug: bool = False,
        calc_burn_in_loss: bool = False,
        edge_features: Optional[torch.FloatTensor] = None,
        zero_edge_probs: bool = True,
        initial_hidden_state: Optional[torch.FloatTensor] = None,
        update_hidden_state: Optional[Callable[[torch.FloatTensor, int], torch.FloatTensor]] = None,
        teacher_forcing_steps: int = float('inf')
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor
    ]:
        '''
        batch compute training loss

        Args:
        ---
        - trajectory_embeddings: [B, A, t, dim]
        - trajectories: [B, A, t, dim]
        - trajectory_masks: [B, A, t]
        - edge_probs: [B, E, t, n_edges]
        - edge_masks: [B, E, t]
        - graph_info
        - obs_len
        - pred_len
        - compute_embedding: compute embedding (observation, timestamp) => embedding
        - put_nan: put nan in output if set to `True`. default `True`
        - debug: debug mode
        - calc_burn_in_loss: compute NLL loss in burn-in segment
        - edge_features: edge features used for dG-VAE
        - zero_edge_probs: put 0s in invalid mask locations in `edge_probs` if `True`.
        - initial_hidden_state: [B, A, dim]

        Returns:
        ---
        - NLL loss
        - parameters of predicted distributions
        '''
        if update_hidden_state is None:
            update_hidden_state = lambda x, *args: x

        if zero_edge_probs:
            edge_probs = edge_probs.masked_fill(~edge_masks[..., None], 0)

        batch_size, num_agents = trajectory_embeddings.shape[:2]
        
        burn_in_predictions, hidden_state = self.burn_in(
            trajectory_embeddings[..., :obs_len - 1, :],
            trajectories[..., :obs_len - 1, :],
            trajectory_masks[..., :obs_len - 1],
            edge_probs[..., :obs_len - 1, :],
            edge_masks[..., :obs_len - 1],
            graph_info,
            put_nan=False,
            debug=debug,
            edge_features=edge_features,
            zero_edge_probs=False,
            hidden_state=initial_hidden_state,
            update_hidden_state=update_hidden_state
        )

        predicted_distributions = torch.zeros((batch_size, num_agents, pred_len, DISTRIBUTION_DIM), dtype=torch.float32, device=trajectory_embeddings.device)

        edge_features_at_t: torch.FloatTensor = None

        for timestamp in range(pred_len):
            if self.dgvae:
                edge_features_at_t = edge_features[..., obs_len + timestamp - 1, :]

            if timestamp % teacher_forcing_steps == 0:
                prev_observation = trajectories[..., obs_len + timestamp - 1, :]
            
            prev_observation_embedding = compute_embedding(prev_observation, obs_len + timestamp - 1)

            hidden_state = update_hidden_state(hidden_state, timestamp)

            predicted_distributions[..., timestamp, :], hidden_state = self.predict_next_step(
                prev_observation_embedding,
                prev_observation,
                trajectory_masks[..., obs_len + timestamp - 1],
                edge_probs[..., obs_len + timestamp - 1, :],
                edge_masks[..., obs_len + timestamp - 1],
                hidden_state,
                graph_info,
                put_nan=False,
                debug=debug,
                edge_features=edge_features_at_t,
                zero_edge_probs=False
            )

            prev_observation = self.most_probable_trajectory(predicted_distributions[..., timestamp, :])


        if debug:
            assert not predicted_distributions[trajectory_masks[..., obs_len:], ...].isnan().any()

        if put_nan:
            predicted_distributions = predicted_distributions.masked_fill(
                ~trajectory_masks[..., obs_len:, None],
                torch.nan
            )

        loss = self.trajectory_nll(
            predicted_distributions, 
            trajectories[..., obs_len:, :], 
            trajectory_masks[..., obs_len:],
            debug=debug,
        )

        burn_in_loss = torch.tensor(0.0, device=trajectories.device)

        if calc_burn_in_loss:
            burn_in_prediction_mask = trajectory_masks[..., :obs_len - 1] & trajectory_masks[..., 1:obs_len]

            burn_in_loss = self.trajectory_nll(
                burn_in_predictions,
                trajectories[..., 1:obs_len, :],
                burn_in_prediction_mask,
                debug=debug,
            )
            
        assert not loss.isnan().any()
        assert not burn_in_loss.isnan().any()

        return loss, burn_in_loss, predicted_distributions

    #region distribution ctors
    @staticmethod
    def create_bivariate_normal(
        loc: torch.FloatTensor,
        covariance: torch.FloatTensor,
        eps: float = 1e-10,
    ):
        '''
        create bivariate normal distribution from Cholesky decomposition of covariance matrix

        Args:
        ---
        - loc: mean [...][x, y]
        - covariance: [...][sd_x: standard deviation of x, sd_y: standard deviation of y, corr_xy: correlation coefficient of x-y]
        - eps: epsilon
        '''
        sigma = torch.zeros((*loc.shape[:-1], 2, 2), dtype=torch.float32, device=loc.device)

        sigma_x = covariance[..., 0] + eps
        sigma_y = covariance[..., 1] + eps
        corr_xy = covariance[..., 2]

        sigma[..., 0, 0] = sigma_x
        sigma[..., 1, 1] = sigma_y + torch.sqrt(1 - corr_xy.pow(2))
        sigma[..., 1, 0] = corr_xy * sigma_y

        return td.MultivariateNormal(
            loc=loc, 
            scale_tril=sigma, 
            validate_args=False
        )
    
    @staticmethod
    def create_von_mises(
        loc: torch.FloatTensor,
        concentration: torch.FloatTensor,
        eps: float = 1e-5,
    ) -> td.VonMises:
        '''
        create von Mises distribution

        Args:
        ---
        - direction: [...][angle]
        - concentration: [...]
        '''
        return td.VonMises(
            loc=loc.fmod(torch.pi), 
            concentration=concentration + eps, 
            validate_args=False
        )
    #endregion

    def predict_trajectory_distribution_params(self, x: torch.FloatTensor):
        '''
        Predict distribution parameters. Use Trajectron++ parametrization.

        Args:
        ---
        - x: [...][
            mu_x, mu_y, sigma_x, sigma_y, corr_xy, 
            mu_vx, mu_vy, sigma_vx, sigma_vy, corr_vxy, 
            mu_ax, mu_ay, sigma_ax, sigma_ay, corr_axy,
            yaw_x, yaw_y, yaw_conc,
            heading_x, heading_y, heading_conc,
            mu_vlon, mu_vlat, sigma_vlon, sigma_vlat, corr_vlonlat,
            mu_alon, mu_alat, sigma_alon, sigma_alat, corr_alonlat
        ]

        Returns:
        ---
        - y: [...][
            mu_x, mu_y, sigma_x, sigma_y, corr_xy, 
            mu_vx, mu_vy, sigma_vx, sigma_vy, corr_vxy, 
            mu_ax, mu_ay, sigma_ax, sigma_ay, corr_axy,
            yaw_rad, yaw_conc,
            heading_rad, heading_conc,
            mu_vlon, mu_vlat, sigma_vlon, sigma_vlat, corr_vlonlat,
            mu_alon, mu_alat, sigma_alon, sigma_alat, corr_alonlat
        ]
        '''
        y = torch.zeros((*x.shape[:-1], DISTRIBUTION_DIM), dtype=x.dtype, device=x.device)

        y[..., :15] = x[..., :15]
        y[..., 19:] = x[..., 21:]
            
        # angle parametrization
        normed = F.normalize(x[..., [15, 16]], dim=-1)
        y[..., 15] = torch.atan2(normed[..., 0], normed[..., 1])
        y[..., 16] = x[..., 17]

        normed = F.normalize(x[..., [18, 19]], dim=-1)
        y[..., 17] = torch.atan2(normed[..., 0], normed[..., 1])
        y[..., 18] = x[..., 20]

        # standard deviations and concentrations
        y[..., [2, 3, 7, 8, 12, 13, 16, 18, 21, 22, 26, 27]] = y[..., [2, 3, 7, 8, 12, 13, 16, 18, 21, 22, 26, 27]].exp()
        y[..., [2, 3, 7, 8, 12, 13, 21, 22, 26, 27]].clip_(0.01, 1)

        # correlation coef
        y[..., [4, 9, 14, 23, 28]] = y[..., [4, 9, 14, 23, 28]].tanh()

        return y
    
    def to_distribution(self, params: torch.FloatTensor, debug: bool = False) -> TrajectoryDistribution:
        '''
        Convert parameters to `TrajectoryDistribution` sinstance

        Args:
        ---
        - params: [...][
            mu_x, mu_y, sigma_x, sigma_y, corr_xy, 
            mu_vx, mu_vy, sigma_vx, sigma_vy, corr_vxy, 
            mu_ax, mu_ay, sigma_ax, sigma_ay, corr_axy,
            yaw_rad, yaw_sigma,
            heading_rad, heading_sigma,
            mu_vlon, mu_vlat, sigma_vlon, sigma_vlat, corr_vlonlat,
            mu_alon, mu_alat, sigma_alon, sigma_alat, corr_alonlat
        ]
        - debug: debug mode
        '''
        distributions = (
            Decoder.create_bivariate_normal(loc=params[..., [0, 1]], covariance=params[..., [2, 3, 4]]),
            Decoder.create_bivariate_normal(loc=params[..., [5, 6]], covariance=params[..., [7, 8, 9]]),
            Decoder.create_bivariate_normal(loc=params[..., [10, 11]], covariance=params[..., [12, 13, 14]]),
            Decoder.create_von_mises(loc=params[..., 15], concentration=params[..., 16]),
            Decoder.create_von_mises(loc=params[..., 17], concentration=params[..., 18]),
            Decoder.create_bivariate_normal(loc=params[..., [19, 20]], covariance=params[..., [21, 22, 23]]),
            Decoder.create_bivariate_normal(loc=params[..., [25, 25]], covariance=params[..., [26, 27, 28]]),
        )

        return TrajectoryDistribution(distributions)

    def most_probable_trajectory(self, x: torch.FloatTensor):
        '''
        get most likely trajectory (mean)

        Args:
        ---
        - x: [...][
            mu_x, mu_y, sigma_x, sigma_y, corr_xy, 
            mu_vx, mu_vy, sigma_vx, sigma_vy, corr_vxy, 
            mu_ax, mu_ay, sigma_ax, sigma_ay, corr_axy,
            yaw_rad, yaw_sigma,
            heading_rad, heading_sigma,
            mu_vlon, mu_vlat, sigma_vlon, sigma_vlat, corr_vlonlat,
            mu_alon, mu_alat, sigma_alon, sigma_alat, corr_alonlat
        ]

        Returns:
        ---
        - y: [...][
            mu_x, mu_y,
            mu_vx, mu_vy,
            mu_ax, mu_ay,
            yaw_rad,
            heading_rad,
            mu_vlon, mu_vlat,
            mu_alon, mu_alat
        ]
        '''
        return x[..., [0, 1, 5, 6, 10, 11, 15, 17, 19, 20, 24, 25]].clone()
    
    def trajectory_nll(
        self, 
        distribution: torch.FloatTensor,
        ground_truth: torch.FloatTensor,
        mask: torch.BoolTensor,
        debug: bool = False,
    ):
        '''
        negative log-likelihood loss

        Args:
        ---
        - distribution: predicted distributions
        - ground_truth: ground truth
        - mask
        - debug: debug mode
        - reduction: reduction mode

        Returns:
        - NLL loss
        '''
        distribution = distribution[mask, ...].reshape(-1, DISTRIBUTION_DIM)
        ground_truth = ground_truth[mask, ...].reshape(-1, TRAJECTORY_FEATURE_DIM)

        if distribution.size(0) == 0:
            return torch.tensor(0.0)
        
        nll = -(self.to_distribution(distribution, debug=debug).log_prob(ground_truth))
        return nll.sum(dim=-1).mean()