import unittest

import torch
from torch.utils.data import DataLoader

from SinD.config import get_dataset_path
from SinD.dataset.io import get_dataset_records

from NRI.dataset import SignalizedIntersectionDatasetForNRI, SignalizedIntersectionDatasetConfig
from NRI.models import DynamicNeuralRelationalInference
from NRI.experiments.config import ExperimentConfig
from NRI.experiments.train import train_one_epoch
from NRI.experiments.valid import valid_one_epoch

class ExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        path = get_dataset_path()
        records = get_dataset_records(path)[:1]

        config = SignalizedIntersectionDatasetConfig(
            obs_len=12,
            pred_len=12,
            stride=15,
            padding_value=0,
            encode_traffic_signals=True
        )

        cls.dataset = SignalizedIntersectionDatasetForNRI(config)
        cls.dataset.load_records(path, records, True)

    def test_train_one_epoch(self):
        config = ExperimentConfig(
            obs_len=12,
            pred_len=12,
            encoder_loss_weight=1.0,
            decoder_loss_weight=1.0,
            signal_loss_weight=1.0,
            n_epoch=2,
            batch_size=16,
            checkpoint_prefix='test',
            use_cuda=True
        )

        loader = DataLoader(
            ExperimentTest.dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            collate_fn=SignalizedIntersectionDatasetForNRI.collate_padded,
            num_workers=4
        )

        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
            dgvae=True
        )

        model.cuda()

        optimizer = torch.optim.Adam(model.parameters())

        stats = train_one_epoch(
            model,
            loader,
            optimizer,
            tqdm_desc='[test]',
            config=config
        )

    def test_valid_one_epoch(self):
        config = ExperimentConfig(
            obs_len=12,
            pred_len=12,
            encoder_loss_weight=1.0,
            decoder_loss_weight=1.0,
            signal_loss_weight=1.0,
            n_epoch=2,
            batch_size=16,
            checkpoint_prefix='test',
            use_cuda=True
        )

        loader = DataLoader(
            ExperimentTest.dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            collate_fn=SignalizedIntersectionDatasetForNRI.collate_padded
        )

        model = DynamicNeuralRelationalInference(
            hid_dim=32,
            n_edges=4,
            dgvae=True
        )

        model.cuda()

        stats = valid_one_epoch(
            model,
            loader,
            tqdm_desc='[test]',
            config=config
        )



if __name__ == '__main__':
    unittest.main()