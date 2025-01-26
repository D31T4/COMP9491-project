import unittest

import torch

from NRI.models import DynamicNeuralRelationalInference
from NRI.models.encoder import Encoder
from NRI.models.decoder import Decoder
from NRI.models.agent import AgentEmbedding
from NRI.models.traffic_signal import TrafficSignalSeq2SeqModel, TrafficSignalTransformerModel
from NRI.models.utils import create_graph, avg_logits, ema_logits

from SinD.dataset.dataset import TRAJECTORY_FEATURE_DIM, AGENT_FEATURE_DIM
from SinD.dataset.type import EncodedTrafficSignal, AgentType

#region dNRI tests
class EncoderTest(unittest.TestCase):
    def test_training_loss(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        seq_len = 6

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, in_dim)),
            torch.randn((3, seq_len, in_dim)),
            torch.randn((1, seq_len, in_dim))
        ])

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool),
            torch.ones((1, seq_len), dtype=torch.bool)
        ])

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        trajectory_mask[2][0, 4:] = False
        trajectory_mask[2][0, :1] = False

        graph_info = [
            create_graph(5),
            create_graph(3),
            create_graph(1)
        ]

        loss, edge_logits, edge_masks, _ = encoder.training_loss(trajectories, trajectory_mask, graph_info, debug=True)
        loss.backward()

        self.assertFalse(loss.isnan().item())
        
        for batch_index in range(edge_logits.size(0)):
            self.assertFalse(edge_logits[batch_index][edge_masks[batch_index]].isnan().any())

            num_agents = trajectories[batch_index].size(0)
            num_edges = num_agents * (num_agents - 1)

            self.assertEqual(num_edges, edge_logits[batch_index].size(0))
            self.assertEqual(seq_len, edge_logits[batch_index].size(1))
            self.assertEqual(n_edges, edge_logits[batch_index].size(-1))

            self.assertEqual(num_edges, edge_masks[batch_index].size(0))
            self.assertEqual(seq_len, edge_masks[batch_index].size(-1))

    def test_training_loss_cuda(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        seq_len = 6

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, in_dim)),
            torch.randn((3, seq_len, in_dim))
        ]).cuda()

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).cuda()

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = torch.nested.nested_tensor([
            create_graph(5),
            create_graph(3)
        ]).cuda()

        encoder.cuda()

        loss, edge_logits, edge_masks, _ = encoder.training_loss(trajectories, trajectory_mask, graph_info, debug=True)
        loss.backward()

        self.assertFalse(loss.isnan().item())
        
        for batch_index in range(edge_logits.size(0)):
            self.assertFalse(edge_logits[batch_index][edge_masks[batch_index]].isnan().any())

            num_agents = trajectories[batch_index].size(0)
            num_edges = num_agents * (num_agents - 1)

            self.assertEqual(num_edges, edge_logits[batch_index].size(0))
            self.assertEqual(seq_len, edge_logits[batch_index].size(1))
            self.assertEqual(n_edges, edge_logits[batch_index].size(-1))

            self.assertEqual(num_edges, edge_masks[batch_index].size(0))
            self.assertEqual(seq_len, edge_masks[batch_index].size(-1))

    def test_training_loss_dgvae(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        seq_len = 6

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges,
            dgvae=True
        )

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, in_dim)),
            torch.randn((3, seq_len, in_dim))
        ])

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ])

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = [
            create_graph(5),
            create_graph(3)
        ]

        loss, edge_logits, edge_masks, edge_features = encoder.training_loss(trajectories, trajectory_mask, graph_info, debug=True, put_nan=True)
        loss.backward()

        self.assertFalse(loss.isnan().item())
        
        for batch_index in range(edge_logits.size(0)):
            self.assertFalse(edge_logits[batch_index][edge_masks[batch_index]].isnan().any())

            num_agents = trajectories[batch_index].size(0)
            num_edges = num_agents * (num_agents - 1)

            self.assertEqual(num_edges, edge_logits[batch_index].size(0))
            self.assertEqual(seq_len, edge_logits[batch_index].size(1))
            self.assertEqual(n_edges, edge_logits[batch_index].size(-1))

            self.assertEqual(num_edges, edge_masks[batch_index].size(0))
            self.assertEqual(seq_len, edge_masks[batch_index].size(-1))

            self.assertEqual(num_edges, edge_features[batch_index].size(0))
            self.assertFalse(edge_features[batch_index][edge_masks[batch_index]].isnan().any())

    def test_training_loss_dgvae_cuda(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        seq_len = 6

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges,
            dgvae=True
        )

        encoder.cuda()

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, in_dim)),
            torch.randn((3, seq_len, in_dim))
        ]).cuda()

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).cuda()

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = torch.nested.nested_tensor([
            create_graph(5),
            create_graph(3)
        ]).cuda()

        loss, edge_logits, edge_masks, edge_features = encoder.training_loss(trajectories, trajectory_mask, graph_info, debug=True, put_nan=True)
        loss.backward()

        self.assertFalse(loss.isnan().item())
        
        for batch_index in range(edge_logits.size(0)):
            self.assertFalse(edge_logits[batch_index][edge_masks[batch_index]].isnan().any())

            num_agents = trajectories[batch_index].size(0)
            num_edges = num_agents * (num_agents - 1)

            self.assertEqual(num_edges, edge_logits[batch_index].size(0))
            self.assertEqual(seq_len, edge_logits[batch_index].size(1))
            self.assertEqual(n_edges, edge_logits[batch_index].size(-1))

            self.assertEqual(num_edges, edge_masks[batch_index].size(0))
            self.assertEqual(seq_len, edge_masks[batch_index].size(-1))

            self.assertEqual(num_edges, edge_features[batch_index].size(0))
            self.assertFalse(edge_features[batch_index][edge_masks[batch_index]].isnan().any())

    def test_batch_training_loss(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        seq_len = 6

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges,
            dgvae=True
        )

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, in_dim)),
            torch.randn((3, seq_len, in_dim))
        ])

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ])

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        nested_loss, nested_edge_logits, nested_edge_masks, nested_edge_feats = encoder.training_loss(
            trajectories, 
            trajectory_mask, 
            [create_graph(5), create_graph(3)], 
            debug=True
        )

        batch_loss, batch_edge_logits, batch_edge_masks, batch_edge_feats = encoder.batch_training_loss(
            trajectories.to_padded_tensor(0), 
            trajectory_mask.to_padded_tensor(0), 
            create_graph(5), 
            debug=True
        )

        with self.subTest('test predictions are close'):
            for i in range(2):
                # this may fail...
                #self.assertTrue(
                #    nested_edge_feats[i][nested_edge_masks[i]].isclose(
                #        batch_edge_feats[i, batch_edge_masks[i]],
                #    ).all()
                #)

                self.assertTrue(
                    nested_edge_logits[i][nested_edge_masks[i]].isclose(
                        batch_edge_logits[i, batch_edge_masks[i]],
                    ).all()
                )

        with self.subTest('test losses are close'):
            self.assertTrue(batch_loss.isclose(nested_loss).all())
        
        batch_loss.backward()

    def test_batch_training_loss_cuda(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        seq_len = 6

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges,
            dgvae=True
        )

        encoder.cuda()

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, in_dim)),
            torch.randn((3, seq_len, in_dim))
        ]).cuda()

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).cuda()

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        batch_loss, batch_edge_logits, batch_edge_masks, batch_edge_feats = encoder.batch_training_loss(
            trajectories.to_padded_tensor(0), 
            trajectory_mask.to_padded_tensor(0), 
            create_graph(5).cuda(), 
            debug=True
        )

        self.assertFalse(batch_loss.isnan().any())
        batch_loss.backward()

    def test_burn_in(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        seq_len = 6
        num_agents = 5

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        trajectories = torch.randn((num_agents, seq_len, in_dim))
        trajectory_mask = torch.ones((num_agents, seq_len), dtype=torch.bool)

        trajectory_mask[1, 3:] = False
        trajectory_mask[2, :2] = False

        graph_info = create_graph(num_agents)

        logits, hidden_state, edge_mask, _ = encoder.burn_in(
            trajectories,
            trajectory_mask,
            graph_info
        )

        self.assertFalse(logits[edge_mask].isnan().any())

    def test_burn_in_one_agent(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        seq_len = 6
        num_agents = 1

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        trajectories = torch.randn((num_agents, seq_len, in_dim))
        trajectory_mask = torch.ones((num_agents, seq_len), dtype=torch.bool)

        trajectory_mask[0, 3:] = False

        graph_info = create_graph(num_agents)

        logits, hidden_state, edge_mask, _ = encoder.burn_in(
            trajectories,
            trajectory_mask,
            graph_info
        )

        self.assertFalse(logits[edge_mask].isnan().any())

    def test_batch_burn_in(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        seq_len = 6

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        encoder.cuda()

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, in_dim)),
            torch.randn((3, seq_len, in_dim))
        ]).to_padded_tensor(0).cuda()

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).to_padded_tensor(0).cuda()

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = create_graph(5).cuda()

        logits, hidden_state, edge_mask, _ = encoder.burn_in(
            trajectories,
            trajectory_mask,
            graph_info
        )

        self.assertFalse(logits[edge_mask].isnan().any())

    def test_predict_next_step(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        seq_len = 6
        num_agents = 5
        num_edges = num_agents * (num_agents - 1)

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        trajectories = torch.randn((num_agents, seq_len, in_dim))
        trajectory_mask = torch.ones((num_agents, seq_len), dtype=torch.bool)

        trajectory_mask[1, 3:] = False
        trajectory_mask[2, :2] = False

        graph_info = create_graph(num_agents)

        logits, hidden_state, edge_mask, _ = encoder.predict_next_step(
            trajectories[:, 0],
            trajectory_mask[:, 0],
            hidden_state=None,
            graph_info=graph_info
        )

        self.assertEqual(logits.shape, (num_edges, n_edges))

    def test_predict_next_step_one_agent(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        seq_len = 6
        num_agents = 1
        num_edges = num_agents * (num_agents - 1)

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        trajectories = torch.randn((num_agents, seq_len, in_dim))
        trajectory_mask = torch.ones((num_agents, seq_len), dtype=torch.bool)

        trajectory_mask[0, 3:] = False

        graph_info = create_graph(num_agents)

        logits, hidden_state, edge_mask, _ = encoder.predict_next_step(
            trajectories[:, 0],
            trajectory_mask[:, 0],
            hidden_state=None,
            graph_info=graph_info
        )

        self.assertEqual(logits.shape, (num_edges, n_edges))

class DecoderTest(unittest.TestCase):
    def test_training_loss(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        obs_len = 6
        pred_len = 6
        seq_len = obs_len + pred_len

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        decoder = Decoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, in_dim)),
            torch.randn((3, seq_len, in_dim)),
            torch.randn((1, seq_len, in_dim))
        ])

        real_trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((1, seq_len, TRAJECTORY_FEATURE_DIM))
        ])

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool),
            torch.ones((1, seq_len), dtype=torch.bool)
        ])

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        trajectory_mask[2][0, 4:] = False
        trajectory_mask[2][0, :1] = False

        graph_info = [
            create_graph(5),
            create_graph(3),
            create_graph(1)
        ]

        _, edge_logits, edge_masks, _ = encoder.training_loss(
            trajectories, 
            trajectory_mask, 
            graph_info, 
            debug=True, 
            put_nan=True
        )

        edge_probs = []

        for batch_index in range(trajectories.size(0)):
            current_edge_probs = torch.zeros_like(edge_logits[batch_index])
            current_edge_mask = edge_masks[batch_index]
            current_edge_probs[current_edge_mask] = edge_logits[batch_index][current_edge_mask].softmax(dim=-1)
            edge_probs.append(current_edge_probs)

        edge_probs = torch.nested.as_nested_tensor(edge_probs)

        loss, burn_in_loss, predicted_distributions = decoder.training_loss(
            trajectories,
            real_trajectories,
            trajectory_mask,
            edge_probs=edge_probs,
            edge_masks=edge_masks,
            graph_info=graph_info,
            obs_len=obs_len,
            pred_len=pred_len,
            compute_embedding=lambda trajectories, *args: torch.randn((trajectories.size(0), in_dim), device=trajectories.device),
            put_nan=True,
            debug=True,
            calc_burn_in_loss=True
        )

        for batch_index in range(predicted_distributions.size(0)):
            assert not predicted_distributions[batch_index][trajectory_mask[batch_index][:, obs_len:], :].isnan().any()

        loss = loss + burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_training_loss_cuda(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        obs_len = 6
        pred_len = 6
        seq_len = obs_len + pred_len

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        decoder = Decoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, in_dim)),
            torch.randn((3, seq_len, in_dim))
        ]).cuda()

        real_trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ]).cuda()

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).cuda()

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = torch.nested.nested_tensor([
            create_graph(5),
            create_graph(3)
        ]).cuda()

        encoder.cuda()
        decoder.cuda()

        _, edge_logits, edge_masks, _ = encoder.training_loss(trajectories, trajectory_mask, graph_info, debug=True)
        
        loss, burn_in_loss, predicted_distributions = decoder.training_loss(
            trajectories,
            real_trajectories,
            trajectory_mask,
            edge_probs=edge_logits.softmax(dim=-1),
            edge_masks=edge_masks,
            graph_info=graph_info,
            obs_len=obs_len,
            pred_len=pred_len,
            compute_embedding=(lambda trajectories, *args: torch.randn((trajectories.size(0), in_dim), device=trajectories.device)),
            debug=True,
            calc_burn_in_loss=True
        )

        for batch_index in range(predicted_distributions.size(0)):
            assert not predicted_distributions[batch_index][trajectory_mask[batch_index][:, obs_len:], :].isnan().any()

        loss = loss + burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_training_loss_dgvae(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        obs_len = 6
        pred_len = 6
        seq_len = obs_len + pred_len

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges,
            dgvae=True
        )

        decoder = Decoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges,
            dgvae=True
        )

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, in_dim)),
            torch.randn((3, seq_len, in_dim))
        ]).to_padded_tensor(0)

        real_trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ]).to_padded_tensor(0)

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).to_padded_tensor(0)

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = [
            create_graph(5),
            create_graph(5)
        ]

        _, edge_logits, edge_masks, edge_features = encoder.training_loss(
            trajectories, 
            trajectory_mask, 
            graph_info, 
            put_nan=True, 
            debug=True
        )

        edge_probs = torch.zeros_like(edge_logits)

        for batch_index in range(trajectories.size(0)):
            edge_probs[batch_index][edge_masks[batch_index]] = edge_logits[batch_index][edge_masks[batch_index]].softmax(dim=-1)
        
        loss, burn_in_loss, predicted_distributions = decoder.training_loss(
            trajectories,
            real_trajectories,
            trajectory_mask,
            edge_probs=edge_probs,
            edge_masks=edge_masks,
            graph_info=graph_info,
            obs_len=obs_len,
            pred_len=pred_len,
            compute_embedding=lambda trajectories, *args: torch.randn((trajectories.size(0), in_dim), device=trajectories.device),
            put_nan=True,
            debug=True,
            edge_features=edge_features,
            calc_burn_in_loss=True
        )

        for batch_index in range(predicted_distributions.size(0)):
            assert not predicted_distributions[batch_index][trajectory_mask[batch_index][:, obs_len:], :].isnan().any()

        loss = loss + burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_training_loss_dgvae_cuda(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        obs_len = 6
        pred_len = 6
        seq_len = obs_len + pred_len

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges,
            dgvae=True
        )

        decoder = Decoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges,
            dgvae=True
        )

        encoder.cuda()
        decoder.cuda()

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, in_dim)),
            torch.randn((3, seq_len, in_dim))
        ]).cuda()

        real_trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ]).cuda()

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).cuda()

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = torch.nested.nested_tensor([
            create_graph(5),
            create_graph(3)
        ]).cuda()

        _, edge_logits, edge_masks, edge_features = encoder.training_loss(trajectories, trajectory_mask, graph_info, put_nan=True, debug=True)
        
        loss, burn_in_loss, predicted_distributions = decoder.training_loss(
            trajectories,
            real_trajectories,
            trajectory_mask,
            edge_probs=edge_logits.softmax(dim=-1),
            edge_masks=edge_masks,
            graph_info=graph_info,
            obs_len=obs_len,
            pred_len=pred_len,
            compute_embedding=lambda trajectories, *args: torch.randn((trajectories.size(0), in_dim), device=trajectories.device),
            debug=True,
            edge_features=edge_features,
            calc_burn_in_loss=True
        )

        for batch_index in range(predicted_distributions.size(0)):
            assert not predicted_distributions[batch_index][trajectory_mask[batch_index][:, obs_len:], :].isnan().any()

        loss = loss + burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_batch_training_loss(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        obs_len = 6
        pred_len = 6
        seq_len = obs_len + pred_len

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges,
            dgvae=True
        )

        decoder = Decoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges,
            dgvae=True
        )

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, in_dim)),
            torch.randn((3, seq_len, in_dim))
        ])

        real_trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ])

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ])

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = [
            create_graph(5),
            create_graph(3)
        ]

        _, nested_edge_logits, nested_edge_masks, nested_edge_feats = encoder.training_loss(
            trajectories, 
            trajectory_mask, 
            graph_info, 
            debug=True, 
            put_nan=True
        )

        _, batch_edge_logtis, batch_edge_masks, batch_edge_feats = encoder.batch_training_loss(
            trajectories.to_padded_tensor(0), 
            trajectory_mask.to_padded_tensor(0), 
            create_graph(5), 
            debug=True, 
            put_nan=True
        )

        nested_edge_probs = []

        for batch_index in range(trajectories.size(0)):
            current_edge_probs = torch.zeros_like(nested_edge_logits[batch_index])
            current_edge_mask = nested_edge_masks[batch_index]
            current_edge_probs[current_edge_mask] = nested_edge_logits[batch_index][current_edge_mask].softmax(dim=-1)

            nested_edge_probs.append(current_edge_probs)
        
        nested_edge_probs = torch.nested.as_nested_tensor(nested_edge_probs)

        batch_edge_probs = torch.zeros_like(batch_edge_logtis)
        batch_edge_probs[batch_edge_masks] = batch_edge_logtis[batch_edge_masks].softmax(dim=-1)

        nested_loss, nested_burn_in_loss, nested_predicted_distributions = decoder.training_loss(
            trajectories,
            real_trajectories,
            trajectory_mask,
            edge_probs=nested_edge_probs,
            edge_masks=nested_edge_masks,
            graph_info=graph_info,
            obs_len=obs_len,
            pred_len=pred_len,
            compute_embedding=lambda trajectories, *args: torch.zeros((*trajectories.shape[:-1], in_dim), device=trajectories.device),
            put_nan=True,
            debug=True,
            calc_burn_in_loss=True,
            edge_features=nested_edge_feats
        )

        batch_loss, batch_burn_in_loss, batch_predicted_distributions = decoder.batch_training_loss(
            trajectories.to_padded_tensor(0),
            real_trajectories.to_padded_tensor(0),
            trajectory_mask.to_padded_tensor(0),
            edge_probs=batch_edge_probs,
            edge_masks=batch_edge_masks,
            graph_info=create_graph(5),
            obs_len=obs_len,
            pred_len=pred_len,
            compute_embedding=lambda trajectories, *args: torch.zeros((*trajectories.shape[:-1], in_dim), device=trajectories.device),
            put_nan=True,
            debug=True,
            calc_burn_in_loss=True,
            edge_features=batch_edge_feats
        )

        padded_trajectory_mask = trajectory_mask.to_padded_tensor(0)

        with self.subTest('test predictions are close'):
            for batch_index in range(trajectories.size(0)):
                current_nested_predictions = nested_predicted_distributions[batch_index][trajectory_mask[batch_index][:, obs_len:]]
                current_batch_predictions = batch_predicted_distributions[batch_index, padded_trajectory_mask[batch_index, :, obs_len:]]

                # this may fail...
                self.assertTrue(
                    current_nested_predictions.isclose(
                        current_batch_predictions
                    ).all()
                )

        batch_loss = batch_loss + batch_burn_in_loss
        nested_loss = nested_loss + nested_burn_in_loss

        with self.subTest('test losses are close'):
            assert batch_loss.isclose(nested_loss).all()

        with self.subTest('test autograd'):
            batch_loss.backward()

    def test_batch_training_loss_cuda(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        obs_len = 6
        pred_len = 6
        seq_len = obs_len + pred_len

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges,
            dgvae=True
        )

        decoder = Decoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges,
            dgvae=True
        )

        encoder.cuda()
        decoder.cuda()

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, in_dim)),
            torch.randn((3, seq_len, in_dim))
        ]).to_padded_tensor(0).cuda()

        real_trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ]).to_padded_tensor(0).cuda()

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).to_padded_tensor(0).cuda()

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = create_graph(5).cuda()

        _, batch_edge_logits, batch_edge_masks, batch_edge_feats = encoder.batch_training_loss(
            trajectories, 
            trajectory_mask, 
            graph_info, 
            debug=True, 
            put_nan=True
        )

        batch_edge_probs = torch.zeros_like(batch_edge_logits)
        batch_edge_probs[batch_edge_masks] = batch_edge_logits[batch_edge_masks].softmax(dim=-1)

        batch_loss, batch_burn_in_loss, batch_predicted_distributions = decoder.batch_training_loss(
            trajectories,
            real_trajectories,
            trajectory_mask,
            edge_probs=batch_edge_probs,
            edge_masks=batch_edge_masks,
            graph_info=graph_info,
            obs_len=obs_len,
            pred_len=pred_len,
            compute_embedding=lambda trajectories, *args: torch.randn((*trajectories.shape[:-1], in_dim), device=trajectories.device),
            put_nan=True,
            debug=True,
            calc_burn_in_loss=True,
            edge_features=batch_edge_feats
        )

        batch_loss = batch_loss + batch_burn_in_loss

        assert not batch_loss.isnan().any()
        batch_loss.backward()

    def test_burn_in(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        seq_len = 6
        num_agents = 5

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        decoder = Decoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        trajectories = torch.randn((num_agents, seq_len, in_dim))
        real_trajectories = torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM))

        trajectory_mask = torch.ones((num_agents, seq_len), dtype=torch.bool)

        trajectory_mask[1, 3:] = False
        trajectory_mask[2, :2] = False

        graph_info = create_graph(num_agents)

        logits, hidden_state, edge_mask, _ = encoder.burn_in(
            trajectories,
            trajectory_mask,
            graph_info
        )

        _, hidden_state = decoder.burn_in(
            trajectories,
            real_trajectories,
            trajectory_mask,
            logits.softmax(dim=-1),
            edge_mask,
            graph_info,
            put_nan=True,
            debug=True
        )

    def test_burn_in_one_agent(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        seq_len = 6
        num_agents = 1

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        decoder = Decoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        trajectories = torch.randn((num_agents, seq_len, in_dim))
        real_trajectories = torch.randn((1, seq_len, TRAJECTORY_FEATURE_DIM))

        trajectory_mask = torch.ones((num_agents, seq_len), dtype=torch.bool)

        trajectory_mask[0, 3:] = False

        graph_info = create_graph(num_agents)

        logits, hidden_state, edge_mask, _ = encoder.burn_in(
            trajectories,
            trajectory_mask,
            graph_info
        )

        _, hidden_state = decoder.burn_in(
            trajectories,
            real_trajectories,
            trajectory_mask,
            logits.softmax(dim=-1),
            edge_mask,
            graph_info,
            put_nan=True,
            debug=True
        )

    def test_predict_next_step(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        seq_len = 6
        num_agents = 5
        num_edges = num_agents * (num_agents - 1)

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        decoder = Decoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        trajectories = torch.randn((num_agents, seq_len, in_dim))
        real_trajectories = torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM))

        trajectory_mask = torch.ones((num_agents, seq_len), dtype=torch.bool)
        trajectory_mask[1, 3:] = False
        trajectory_mask[2, :2] = False

        graph_info = create_graph(num_agents)
        edge_mask = trajectory_mask[graph_info[0]] & trajectory_mask[graph_info[1]]

        logits = torch.randn((num_edges, n_edges)).softmax(dim=-1)

        _, hidden_state = decoder.predict_next_step(
            trajectories[:, 0],
            real_trajectories[:, 0],
            trajectory_mask[:, 0],
            logits.softmax(dim=-1),
            edge_mask[:, 0],
            None,
            graph_info,
            put_nan=True,
            debug=True
        )

    def test_predict_next_step_one_agent(self):
        in_dim = 16
        hid_dim = 32
        n_edges = 4
        seq_len = 6
        num_agents = 1
        num_edges = num_agents * (num_agents - 1)

        encoder = Encoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        decoder = Decoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_edges=n_edges
        )

        trajectories = torch.randn((num_agents, seq_len, in_dim))
        real_trajectories = torch.randn((1, seq_len, TRAJECTORY_FEATURE_DIM))

        trajectory_mask = torch.ones((num_agents, seq_len), dtype=torch.bool)
        trajectory_mask[0, 3:] = False

        graph_info = create_graph(num_agents)
        edge_mask = trajectory_mask[graph_info[0]] & trajectory_mask[graph_info[1]]

        logits = torch.randn((num_edges, n_edges)).softmax(dim=-1)

        _, hidden_state = decoder.predict_next_step(
            trajectories[:, 0],
            real_trajectories[:, 0],
            trajectory_mask[:, 0],
            logits.softmax(dim=-1),
            edge_mask[:, 0],
            None,
            graph_info,
            put_nan=True,
            debug=True
        )

class DynamicNeuralRelationalInferenceTest(unittest.TestCase):
    def test_training_loss(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
        )

        obs_len = 12
        pred_len = 12
        seq_len = obs_len + pred_len

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((1, seq_len, TRAJECTORY_FEATURE_DIM))
        ])

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool),
            torch.ones((1, seq_len), dtype=torch.bool)
        ])

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM)),
            torch.randn((1, AGENT_FEATURE_DIM))
        ])

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (1,), dtype=torch.int32)
        ])

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        trajectory_mask[2][0, 4:] = False
        trajectory_mask[2][0, :1] = False

        graph_info = [
            create_graph(5),
            create_graph(3),
            create_graph(1)
        ]

        signals = torch.randint(0, len(EncodedTrafficSignal), (3, seq_len), dtype=torch.int32)

        signal_prediction_loss, encoder_loss, decoder_loss, decoder_burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals = model.training_loss(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            graph_info,
            signals,
            obs_len,
            pred_len,
            include_decoder_burn_in=True,
            put_nan=True,
            debug=True
        )

        loss = signal_prediction_loss + encoder_loss + decoder_loss + decoder_burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_training_loss_cuda(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
        )

        model.cuda()

        obs_len = 12
        pred_len = 12
        seq_len = obs_len + pred_len

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ]).cuda()

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).cuda()

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM))
        ]).cuda()

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32)
        ]).cuda()

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = torch.nested.as_nested_tensor([
            create_graph(5),
            create_graph(3)
        ]).cuda()

        signals = torch.randint(0, len(EncodedTrafficSignal), (2, seq_len), dtype=torch.int32).cuda()

        signal_prediction_loss, encoder_loss, decoder_loss, decoder_burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals = model.training_loss(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            graph_info,
            signals,
            obs_len,
            pred_len,
            include_decoder_burn_in=True,
            put_nan=True,
            debug=True
        )

        loss = signal_prediction_loss + encoder_loss + decoder_loss + decoder_burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_batch_training_loss(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
        )

        obs_len = 12
        pred_len = 12
        seq_len = obs_len + pred_len

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ]).to_padded_tensor(0)

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).to_padded_tensor(0)

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM))
        ]).to_padded_tensor(0)

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32)
        ]).to_padded_tensor(0)

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = create_graph(5)

        signals = torch.randint(0, len(EncodedTrafficSignal), (2, seq_len), dtype=torch.int32)

        signal_prediction_loss, encoder_loss, decoder_loss, decoder_burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals = model.batch_training_loss(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            graph_info,
            signals,
            obs_len,
            pred_len,
            include_decoder_burn_in=True,
            put_nan=True,
            debug=True
        )

        loss = signal_prediction_loss + encoder_loss + decoder_loss + decoder_burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_batch_training_loss_cuda(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
        )

        obs_len = 12
        pred_len = 12
        seq_len = obs_len + pred_len

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ]).to_padded_tensor(0).cuda()

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).to_padded_tensor(0).cuda()

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM))
        ]).to_padded_tensor(0).cuda()

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32)
        ]).to_padded_tensor(0).cuda()

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = create_graph(5).cuda()

        signals = torch.randint(0, len(EncodedTrafficSignal), (2, seq_len), dtype=torch.int32).cuda()

        model.cuda()

        signal_prediction_loss, encoder_loss, decoder_loss, decoder_burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals = model.batch_training_loss(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            graph_info,
            signals,
            obs_len,
            pred_len,
            include_decoder_burn_in=True,
            put_nan=True,
            debug=True
        )

        loss = signal_prediction_loss + encoder_loss + decoder_loss + decoder_burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_training_loss_dgvae(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
            dgvae=True
        )

        obs_len = 12
        pred_len = 12
        seq_len = obs_len + pred_len

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ])

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ])

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM))
        ])

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32)
        ])

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = [
            create_graph(5),
            create_graph(3)
        ]

        signals = torch.randint(0, len(EncodedTrafficSignal), (2, seq_len), dtype=torch.int32)

        signal_prediction_loss, encoder_loss, decoder_loss, decoder_burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals = model.training_loss(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            graph_info,
            signals,
            obs_len,
            pred_len,
            include_decoder_burn_in=True,
            put_nan=True,
            debug=True
        )

        loss = signal_prediction_loss + encoder_loss + decoder_loss + decoder_burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_training_loss_dgvae_cuda(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
            dgvae=True
        )

        model.cuda()

        obs_len = 12
        pred_len = 12
        seq_len = obs_len + pred_len

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ]).cuda()

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).cuda()

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM))
        ]).cuda()

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32)
        ]).cuda()

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = torch.nested.as_nested_tensor([
            create_graph(5),
            create_graph(3)
        ]).cuda()

        signals = torch.randint(0, len(EncodedTrafficSignal), (2, seq_len), dtype=torch.int32).cuda()

        signal_prediction_loss, encoder_loss, decoder_loss, decoder_burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals = model.training_loss(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            graph_info,
            signals,
            obs_len,
            pred_len,
            include_decoder_burn_in=True,
            put_nan=True,
            debug=True
        )

        loss = signal_prediction_loss + encoder_loss + decoder_loss + decoder_burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_batch_training_loss_dgvae(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
            dgvae=True
        )

        obs_len = 12
        pred_len = 12
        seq_len = obs_len + pred_len

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ]).to_padded_tensor(0)

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).to_padded_tensor(0)

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM))
        ]).to_padded_tensor(0)

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32)
        ]).to_padded_tensor(0)

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = create_graph(5)

        signals = torch.randint(0, len(EncodedTrafficSignal), (2, seq_len), dtype=torch.int32)

        signal_prediction_loss, encoder_loss, decoder_loss, decoder_burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals = model.batch_training_loss(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            graph_info,
            signals,
            obs_len,
            pred_len,
            include_decoder_burn_in=True,
            put_nan=True,
            debug=True
        )

        loss = signal_prediction_loss + encoder_loss + decoder_loss + decoder_burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_batch_training_loss_dgvae_cuda(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
            dgvae=True
        )

        obs_len = 12
        pred_len = 12
        seq_len = obs_len + pred_len

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ]).to_padded_tensor(0).cuda()

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).to_padded_tensor(0).cuda()

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM))
        ]).to_padded_tensor(0).cuda()

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32)
        ]).to_padded_tensor(0).cuda()

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = create_graph(5).cuda()

        signals = torch.randint(0, len(EncodedTrafficSignal), (2, seq_len), dtype=torch.int32).cuda()

        model.cuda()

        signal_prediction_loss, encoder_loss, decoder_loss, decoder_burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals = model.batch_training_loss(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            graph_info,
            signals,
            obs_len,
            pred_len,
            include_decoder_burn_in=True,
            put_nan=True,
            debug=True
        )

        loss = signal_prediction_loss + encoder_loss + decoder_loss + decoder_burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_training_loss_dgvae_avg(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
            dgvae=True
        )

        obs_len = 12
        pred_len = 12
        seq_len = obs_len + pred_len

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ])

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ])

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM))
        ])

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32)
        ])

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = [
            create_graph(5),
            create_graph(3)
        ]

        signals = torch.randint(0, len(EncodedTrafficSignal), (2, seq_len), dtype=torch.int32)

        signal_prediction_loss, encoder_loss, decoder_loss, decoder_burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals = model.training_loss(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            graph_info,
            signals,
            obs_len,
            pred_len,
            include_decoder_burn_in=True,
            put_nan=True,
            debug=True,
            avg_logits='avg'
        )

        loss = signal_prediction_loss + encoder_loss + decoder_loss + decoder_burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_batch_training_loss_dgvae_avg(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
            dgvae=True
        )

        obs_len = 12
        pred_len = 12
        seq_len = obs_len + pred_len

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ]).to_padded_tensor(0)

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).to_padded_tensor(0)

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM))
        ]).to_padded_tensor(0)

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32)
        ]).to_padded_tensor(0)

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = create_graph(5)

        signals = torch.randint(0, len(EncodedTrafficSignal), (2, seq_len), dtype=torch.int32)

        signal_prediction_loss, encoder_loss, decoder_loss, decoder_burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals = model.batch_training_loss(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            graph_info,
            signals,
            obs_len,
            pred_len,
            include_decoder_burn_in=True,
            put_nan=True,
            debug=True,
            avg_logits='avg'
        )

        loss = signal_prediction_loss + encoder_loss + decoder_loss + decoder_burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_training_loss_dgvae_ema(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
            dgvae=True
        )

        obs_len = 12
        pred_len = 12
        seq_len = obs_len + pred_len

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ])

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ])

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM))
        ])

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32)
        ])

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = [
            create_graph(5),
            create_graph(3)
        ]

        signals = torch.randint(0, len(EncodedTrafficSignal), (2, seq_len), dtype=torch.int32)

        signal_prediction_loss, encoder_loss, decoder_loss, decoder_burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals = model.training_loss(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            graph_info,
            signals,
            obs_len,
            pred_len,
            include_decoder_burn_in=True,
            put_nan=True,
            debug=True,
            avg_logits='ema',
            ema_alpha=0.9
        )

        loss = signal_prediction_loss + encoder_loss + decoder_loss + decoder_burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_batch_training_loss_dgvae_ema(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
            dgvae=True
        )

        obs_len = 12
        pred_len = 12
        seq_len = obs_len + pred_len

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, seq_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, seq_len, TRAJECTORY_FEATURE_DIM))
        ]).to_padded_tensor(0)

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, seq_len), dtype=torch.bool),
            torch.ones((3, seq_len), dtype=torch.bool)
        ]).to_padded_tensor(0)

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM))
        ]).to_padded_tensor(0)

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32)
        ]).to_padded_tensor(0)

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        graph_info = create_graph(5)

        signals = torch.randint(0, len(EncodedTrafficSignal), (2, seq_len), dtype=torch.int32)

        signal_prediction_loss, encoder_loss, decoder_loss, decoder_burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals = model.batch_training_loss(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            graph_info,
            signals,
            obs_len,
            pred_len,
            include_decoder_burn_in=True,
            put_nan=True,
            debug=True,
            avg_logits='ema',
            ema_alpha=0.9
        )

        loss = signal_prediction_loss + encoder_loss + decoder_loss + decoder_burn_in_loss

        assert not loss.isnan().any()
        loss.backward()

    def test_batch_predict_future(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
        )

        obs_len = 12
        pred_len = 12

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, obs_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, obs_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((1, obs_len, TRAJECTORY_FEATURE_DIM))
        ]).to_padded_tensor(0)

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, obs_len), dtype=torch.bool),
            torch.ones((3, obs_len), dtype=torch.bool),
            torch.ones((1, obs_len), dtype=torch.bool)
        ]).to_padded_tensor(0)

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM)),
            torch.randn((1, AGENT_FEATURE_DIM))
        ]).to_padded_tensor(0)

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (1,), dtype=torch.int32)
        ]).to_padded_tensor(0)

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        trajectory_mask[2][0, 4:] = False
        trajectory_mask[2][0, :1] = False

        signals = torch.randint(0, len(EncodedTrafficSignal), (3, obs_len), dtype=torch.int32)

        predicted, edge_types, edge_masks, pred_signals = model.predict_future(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            signals,
            pred_len,
            put_nan=True
        )

        self.assertFalse(
            edge_types[edge_masks.bool()].isnan().any()
        )

    def test_predict_future_one_agent(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
        )

        obs_len = 12
        pred_len = 12

        trajectories = torch.nested.nested_tensor([
            torch.randn((1, obs_len, TRAJECTORY_FEATURE_DIM))
        ]).to_padded_tensor(0)

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((1, obs_len), dtype=torch.bool)
        ]).to_padded_tensor(0)

        agent_feats = torch.nested.nested_tensor([
            torch.randn((1, AGENT_FEATURE_DIM))
        ]).to_padded_tensor(0)

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (1,), dtype=torch.int32)
        ]).to_padded_tensor(0)

        trajectory_mask[0][0, 4:] = False
        trajectory_mask[0][0, :1] = False

        signals = torch.randint(0, len(EncodedTrafficSignal), (1, obs_len), dtype=torch.int32)

        predicted, edge_types, edge_masks, pred_signals = model.predict_future(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            signals,
            pred_len,
            put_nan=True
        )

        self.assertFalse(
            edge_types[edge_masks.bool()].isnan().any()
        )
        
    def test_batch_predict_future_cuda(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
        )

        obs_len = 12
        pred_len = 12

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, obs_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, obs_len, TRAJECTORY_FEATURE_DIM))
        ]).to_padded_tensor(0).cuda()

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, obs_len), dtype=torch.bool),
            torch.ones((3, obs_len), dtype=torch.bool)
        ]).to_padded_tensor(0).cuda()

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM))
        ]).to_padded_tensor(0).cuda()

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32)
        ]).to_padded_tensor(0).cuda()

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        signals = torch.randint(0, len(EncodedTrafficSignal), (2, obs_len), dtype=torch.int32).cuda()

        model.cuda()

        predicted, edge_types, edge_masks, pred_signals = model.predict_future(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            signals,
            pred_len,
            put_nan=True
        )

        self.assertFalse(
            edge_types[edge_masks.bool()].isnan().any()
        )

    def test_batch_predict_future_dgvae_ema(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
            dgvae=True
        )

        obs_len = 12
        pred_len = 12

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, obs_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, obs_len, TRAJECTORY_FEATURE_DIM))
        ]).to_padded_tensor(0)

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, obs_len), dtype=torch.bool),
            torch.ones((3, obs_len), dtype=torch.bool)
        ]).to_padded_tensor(0)

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM))
        ]).to_padded_tensor(0)

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32)
        ]).to_padded_tensor(0)

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        signals = torch.randint(0, len(EncodedTrafficSignal), (2, obs_len), dtype=torch.int32)

        predicted, edge_types, edge_masks, pred_signals = model.predict_future(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            signals,
            pred_len,
            put_nan=True,
            ema_alpha=0.9
        )

        self.assertFalse(
            edge_types[edge_masks.bool()].isnan().any()
        )

    def test_batch_predict_future_dgvae_ema_cuda(self):
        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
            dgvae=True
        )

        obs_len = 12
        pred_len = 12

        trajectories = torch.nested.nested_tensor([
            torch.randn((5, obs_len, TRAJECTORY_FEATURE_DIM)),
            torch.randn((3, obs_len, TRAJECTORY_FEATURE_DIM))
        ]).to_padded_tensor(0).cuda()

        trajectory_mask = torch.nested.nested_tensor([
            torch.ones((5, obs_len), dtype=torch.bool),
            torch.ones((3, obs_len), dtype=torch.bool)
        ]).to_padded_tensor(0).cuda()

        agent_feats = torch.nested.nested_tensor([
            torch.randn((5, AGENT_FEATURE_DIM)),
            torch.randn((3, AGENT_FEATURE_DIM))
        ]).to_padded_tensor(0).cuda()

        agent_types = torch.nested.nested_tensor([
            torch.randint(0, len(AgentType), (5,), dtype=torch.int32),
            torch.randint(0, len(AgentType), (3,), dtype=torch.int32)
        ]).to_padded_tensor(0).cuda()

        trajectory_mask[0][1, 3:] = False
        trajectory_mask[0][2, :2] = False

        trajectory_mask[1][0, 4:] = False
        trajectory_mask[1][1, :1] = False

        signals = torch.randint(0, len(EncodedTrafficSignal), (2, obs_len), dtype=torch.int32).cuda()

        model.cuda()

        predicted, edge_types, edge_masks, pred_signals = model.predict_future(
            trajectories,
            trajectory_mask,
            agent_feats,
            agent_types,
            signals,
            pred_len,
            put_nan=True,
            ema_alpha=0.9
        )

        self.assertFalse(
            edge_types[edge_masks.bool()].isnan().any()
        )
#endregion


class AgentEmbeddingTest(unittest.TestCase):
    def test_forward_nested(self):
        model = AgentEmbedding(32)

        agent_feats = torch.nested.nested_tensor([
            torch.randn(12, 2),
            torch.randn(10, 2)
        ])

        agent_types = torch.nested.nested_tensor([
            torch.randint(low=0, high=6, size=(12,), dtype=torch.int32),
            torch.randint(low=0, high=6, size=(10,), dtype=torch.int32)
        ])

        agent_embedding = model.forward(agent_feats, agent_types)

    def test_forward_nested_cuda(self):
        model = AgentEmbedding(32)
        model.cuda()

        agent_feats = torch.nested.nested_tensor([
            torch.randn(12, 2),
            torch.randn(10, 2)
        ]).cuda()

        agent_types = torch.nested.nested_tensor([
            torch.randint(low=0, high=6, size=(12,), dtype=torch.int32),
            torch.randint(low=0, high=6, size=(10,), dtype=torch.int32)
        ]).cuda()

        agent_embedding = model.forward(agent_feats, agent_types)

    def test_forward(self):
        model = AgentEmbedding(32)

        agent_feats = torch.randn((3, 12, 2))
        agent_types = torch.randint(low=0, high=6, size=(3, 12,), dtype=torch.int32)

        agent_embedding = model.forward(agent_feats, agent_types)
        
#region TrafficSignalSequenceModel tests
class TrafficSignalSeq2SeqModelTest(unittest.TestCase):
    def test_training_loss(self):
        model = TrafficSignalSeq2SeqModel(16, 2)

        signals = torch.zeros((4, 24), dtype=torch.int32)

        logits, loss = model.training_loss(signals, obs_len=12, pred_len=12)

        self.assertFalse(loss.isnan().any())
        loss.backward()

    def test_autoregressive_training_loss(self):
        model = TrafficSignalSeq2SeqModel(16, 2, autoregression=True)

        signals = torch.zeros((4, 24), dtype=torch.int32)

        logits, loss = model.training_loss(signals, obs_len=12, pred_len=12)

        self.assertFalse(loss.isnan().any())
        loss.backward()

    def test_training_loss_cuda(self):
        model = TrafficSignalSeq2SeqModel(16, 2)
        model.cuda()

        signals = torch.zeros((4, 24), dtype=torch.int32).cuda()

        logits, loss = model.training_loss(signals, obs_len=12, pred_len=12)

        self.assertFalse(loss.isnan().any())
        loss.backward()

    def test_autoregressive_training_loss_cuda(self):
        model = TrafficSignalSeq2SeqModel(16, 2, autoregression=True)
        model.cuda()

        signals = torch.zeros((4, 24), dtype=torch.int32).cuda()

        logits, loss = model.training_loss(signals, obs_len=12, pred_len=12)

        self.assertFalse(loss.isnan().any())
        loss.backward()

    def test_most_probable_outcome(self):
        model = TrafficSignalSeq2SeqModel(16, 2)

        signals = torch.zeros((4, 24), dtype=torch.int32)
        predicted_signals = model.most_probable_outcome(signals, 12).to(torch.int8)

    def test_most_probable_outcome_cuda(self):
        model = TrafficSignalSeq2SeqModel(16, 2)
        model.cuda()

        signals = torch.zeros((4, 24), dtype=torch.int32).cuda()
        predicted_signals = model.most_probable_outcome(signals, 12).to(torch.int8)


class TrafficSignalTransformerModelTest(unittest.TestCase):
    def test_training_loss(self):
        model = TrafficSignalTransformerModel(
            hid_dim=32,
            ffn_dim=64,
            n_head=4,
            max_seq_len=24,
            num_layers=2
        )

        signals = torch.zeros((4, 24), dtype=torch.int32)

        logits, loss = model.training_loss(signals, obs_len=12, pred_len=12)

        self.assertFalse(loss.isnan().any())
        loss.backward()

    def test_training_loss_cuda(self):
        model = TrafficSignalTransformerModel(
            hid_dim=32,
            ffn_dim=64,
            n_head=4,
            max_seq_len=24,
            num_layers=2
        )

        model.cuda()

        signals = torch.zeros((4, 24), dtype=torch.int32).cuda()

        logits, loss = model.training_loss(signals, obs_len=12, pred_len=12)

        self.assertFalse(loss.isnan().any())
        loss.backward()

    def most_probable_outcome(self):
        model = TrafficSignalTransformerModel(
            hid_dim=32,
            ffn_dim=64,
            n_head=4,
            max_seq_len=24,
            num_layers=2
        )

        signals = torch.zeros((4, 24), dtype=torch.int32)
        predicted_signals = model.most_probable_outcome(signals, 12).to(torch.int8)

    def most_probable_outcome_cuda(self):
        model = TrafficSignalTransformerModel(
            hid_dim=32,
            ffn_dim=64,
            n_head=4,
            max_seq_len=24,
            num_layers=2
        )

        model.cuda()

        signals = torch.zeros((4, 24), dtype=torch.int32).cuda()
        predicted_signals = model.most_probable_outcome(signals, 12).to(torch.int8)
#endregion

#region utils tests
class test_avg_logits(unittest.TestCase):
    def test(self):
        logits = torch.randn((4, 4, 4))
        
        partial_mask = torch.tensor([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ], dtype=torch.bool)

        self.assertTrue(
            avg_logits(logits).isclose(
                logits.mean(dim=1, keepdim=True).repeat(1, 4, 1)
            ).all().item()
        )

        self.assertTrue(
            avg_logits(logits, partial_mask).isclose(
                torch.stack([
                    logits[i, partial_mask[i]].mean(dim=0, keepdim=True).repeat(4, 1)
                    for i in range(4)
                ], dim=0)
            ).all().item()
        )

    def test_cuda(self):
        logits = torch.randn((4, 4, 4)).cuda()
        
        partial_mask = torch.tensor([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ], dtype=torch.bool).cuda()

        self.assertTrue(
            avg_logits(logits).isclose(
                logits.mean(dim=1, keepdim=True).repeat(1, 4, 1)
            ).all().item()
        )

        self.assertTrue(
            avg_logits(logits, partial_mask).isclose(
                torch.stack([
                    logits[i, partial_mask[i]].mean(dim=0, keepdim=True).repeat(4, 1)
                    for i in range(4)
                ], dim=0)
            ).all().item()
        )

class test_ema_logits(unittest.TestCase):
    def test(self):
        logits = torch.tensor([
            [
                [0], [1], [2],
            ], [
                [3], [4], [5]
            ], [
                [3], [4], [5]
            ], [
                [3], [4], [5]
            ]
        ], dtype=torch.float32)
        
        partial_mask = torch.tensor([
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ], dtype=torch.bool)

        self.assertTrue(
            ema_logits(logits, 0.9).isclose(
                torch.tensor([
                    [
                        [0], [0.9], [0.9 * 0.1 + 2 * 0.9]
                    ], [
                        [3], [3 * 0.1 + 4 * 0.9], [0.1 * (3 * 0.1 + 4 * 0.9) + 5 * 0.9]
                    ], [
                        [3], [3 * 0.1 + 4 * 0.9], [0.1 * (3 * 0.1 + 4 * 0.9) + 5 * 0.9]
                    ], [
                        [3], [3 * 0.1 + 4 * 0.9], [0.1 * (3 * 0.1 + 4 * 0.9) + 5 * 0.9]
                    ]
                ], dtype=torch.float32)
            ).all().item()
        )

        self.assertTrue(
            ema_logits(logits, 0.9, partial_mask).isclose(
                torch.tensor([
                    [
                        [0], [0.9], [0]
                    ], [
                        [0], [0], [5]
                    ], [
                        [0], [4], [4 * 0.1 + 5 * 0.9]
                    ], [
                        [3], [3 * 0.1 + 4 * 0.9], [0.1 * (3 * 0.1 + 4 * 0.9) + 5 * 0.9]
                    ]
                ], dtype=torch.float32)
            ).all().item()
        )

    def test_cuda(self):
        logits = torch.tensor([
            [
                [0], [1], [2],
            ], [
                [3], [4], [5]
            ], [
                [3], [4], [5]
            ], [
                [3], [4], [5]
            ]
        ], dtype=torch.float32).cuda()
        
        partial_mask = torch.tensor([
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ], dtype=torch.bool).cuda()

        self.assertTrue(
            ema_logits(logits, 0.9).isclose(
                torch.tensor([
                    [
                        [0], [0.9], [0.9 * 0.1 + 2 * 0.9]
                    ], [
                        [3], [3 * 0.1 + 4 * 0.9], [0.1 * (3 * 0.1 + 4 * 0.9) + 5 * 0.9]
                    ], [
                        [3], [3 * 0.1 + 4 * 0.9], [0.1 * (3 * 0.1 + 4 * 0.9) + 5 * 0.9]
                    ], [
                        [3], [3 * 0.1 + 4 * 0.9], [0.1 * (3 * 0.1 + 4 * 0.9) + 5 * 0.9]
                    ]
                ], dtype=torch.float32).cuda()
            ).all().item()
        )

        self.assertTrue(
            ema_logits(logits, 0.9, partial_mask).isclose(
                torch.tensor([
                    [
                        [0], [0.9], [0]
                    ], [
                        [0], [0], [5]
                    ], [
                        [0], [4], [4 * 0.1 + 5 * 0.9]
                    ], [
                        [3], [3 * 0.1 + 4 * 0.9], [0.1 * (3 * 0.1 + 4 * 0.9) + 5 * 0.9]
                    ]
                ], dtype=torch.float32).cuda()
            ).all().item()
        )

#endregion

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True, False) # check_nan dont work with nested tensors
    unittest.main()