import unittest

import torch
from torch.utils.data import DataLoader

from NRI.dataset import SignalizedIntersectionDatasetForNRI, SignalizedIntersectionDatasetConfig

from SinD.config import get_dataset_path
from SinD.dataset.io import get_dataset_records
from SinD.dataset.dataset import SIGNAL_DIM

class dataset_test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        path = get_dataset_path()
        records = get_dataset_records(path)[:2]

        config = SignalizedIntersectionDatasetConfig(
            obs_len=12,
            pred_len=12,
            stride=15,
            padding_value=float('nan')
        )

        cls.dataset = SignalizedIntersectionDatasetForNRI(config)
        cls.dataset.load_records(path, records, False)

    def test_output(self):
        for el in dataset_test.dataset:
            (
                trajectory,
                trajectory_mask,
                agent_mask,
                agent_records,
                agent_flags,
                signals,
                graph_info
            ) = el

            self.assertFalse(torch.any(torch.isnan(trajectory[trajectory_mask])))

            num_agents = agent_records.size(0)
            self.assertEqual(num_agents, agent_flags.size(0))
            self.assertEqual(num_agents, agent_mask.size(0))

            self.assertEqual(signals.shape, (dataset_test.dataset.config.seq_len, SIGNAL_DIM))

            send_edges, recv_edges = graph_info
            self.assertEqual(send_edges.size(0), num_agents * (num_agents - 1))
            self.assertEqual(recv_edges.size(0), num_agents * (num_agents - 1))

    def test_collate_nested(self):
        dataloader = DataLoader(dataset_test.dataset, batch_size=4, shuffle=True, collate_fn=SignalizedIntersectionDatasetForNRI.collate_nested)

        for batch in dataloader:
            (
                trajectory,
                trajectory_mask,
                agent_mask,
                agent_records,
                agent_flags,
                signals,
                graph_info
            ) = batch

            for i in range(agent_records.size(0)):
                self.assertFalse(torch.any(torch.isnan(trajectory[i][trajectory_mask[i]])))

                num_agents = agent_records[i].size(0)
                self.assertEqual(num_agents, agent_flags[i].size(0))
                self.assertEqual(num_agents, agent_mask[i].size(0))

                send_edges, recv_edges = graph_info[i]
                self.assertEqual(send_edges.size(0), num_agents * (num_agents - 1))
                self.assertEqual(recv_edges.size(0), num_agents * (num_agents - 1))
            
            self.assertEqual(signals.size(-2), dataset_test.dataset.config.seq_len)
            self.assertEqual(signals.size(-1), SIGNAL_DIM)

    def test_collate_padded(self):
        dataloader = DataLoader(
            dataset_test.dataset, 
            batch_size=64,
            num_workers=4, 
            shuffle=True, 
            collate_fn=SignalizedIntersectionDatasetForNRI.collate_padded
        )

        for batch in dataloader:
            (
                trajectory,
                trajectory_mask,
                agent_mask,
                agent_records,
                agent_flags,
                signals,
                graph_info
            ) = batch

            self.assertFalse(torch.any(torch.isnan(trajectory[trajectory_mask])))

            num_agents = agent_records.size(1)
            self.assertEqual(num_agents, agent_flags.size(1))
            self.assertEqual(num_agents, agent_mask.size(1))

            send_edges, recv_edges = graph_info
            self.assertEqual(send_edges.size(0), num_agents * (num_agents - 1))
            self.assertEqual(recv_edges.size(0), num_agents * (num_agents - 1))
            
            self.assertEqual(signals.size(-2), dataset_test.dataset.config.seq_len)
            self.assertEqual(signals.size(-1), SIGNAL_DIM)

if __name__ == '__main__':
    unittest.main()