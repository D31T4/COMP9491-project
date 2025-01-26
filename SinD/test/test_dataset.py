from SinD.dataset.dataset import SignalizedIntersectionDataset, SignalizedIntersectionDatasetConfig, SIGNAL_DIM, encode_traffic_signals, decode_traffic_signals, TrafficSignalType, EncodedTrafficSignal
from SinD.dataset.io import get_dataset_records
from SinD.config import get_dataset_path

import unittest

import torch
from torch.utils.data import DataLoader

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

        cls.dataset = SignalizedIntersectionDataset(config)
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
            ) = el

            self.assertFalse(torch.any(torch.isnan(trajectory[trajectory_mask])))

            self.assertEqual(agent_records.size(0), agent_flags.size(0))
            self.assertEqual(agent_records.size(0), agent_mask.size(0))

            self.assertEqual(signals.shape, (dataset_test.dataset.config.seq_len, SIGNAL_DIM))

    def test_collated_output(self):
        dataloader = DataLoader(dataset_test.dataset, batch_size=4, shuffle=True, collate_fn=SignalizedIntersectionDataset.collate_nested)

        for batch in dataloader:
            (
                trajectory,
                trajectory_mask,
                agent_mask,
                agent_records,
                agent_flags,
                signals,
            ) = batch

            for i in range(agent_records.size(0)):
                self.assertFalse(torch.any(torch.isnan(trajectory[i][trajectory_mask[i]])))

                self.assertEqual(agent_records[i].size(0), agent_flags[i].size(0))
                self.assertEqual(agent_records[i].size(0), agent_mask[i].size(0))
            
            self.assertEqual(signals.size(-2), dataset_test.dataset.config.seq_len)
            self.assertEqual(signals.size(-1), SIGNAL_DIM)


class test_encode_decode_traffic_signals(unittest.TestCase):
    def test_encode(self):
       encoded = encode_traffic_signals(torch.tensor([[
            TrafficSignalType.green, 
            TrafficSignalType.red, 
            TrafficSignalType.red, 
            TrafficSignalType.green,
            TrafficSignalType.green,
            TrafficSignalType.red,
            TrafficSignalType.red,
            TrafficSignalType.green
       ]], dtype=torch.int8))

       self.assertEqual(encoded, torch.tensor([EncodedTrafficSignal.GGRR], dtype=torch.int8))

    def test_decode(self):
        decoded = decode_traffic_signals(torch.tensor([[EncodedTrafficSignal.GGRR]], dtype=torch.int8))

        self.assertTrue(
            decoded.eq(torch.tensor([[
                TrafficSignalType.green, 
                TrafficSignalType.red, 
                TrafficSignalType.red, 
                TrafficSignalType.green,
                TrafficSignalType.green,
                TrafficSignalType.red,
                TrafficSignalType.red,
                TrafficSignalType.green
            ]], dtype=torch.int8)).all()
        )

if __name__ == '__main__':
    unittest.main()