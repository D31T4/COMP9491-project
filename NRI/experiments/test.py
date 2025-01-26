import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from NRI.dataset import SignalizedIntersectionDatasetForNRI
from NRI.models import DynamicNeuralRelationalInference
from NRI.experiments.config import ExperimentConfig
from NRI.experiments.metrics import classification_accuracy, displacement_error

from SinD.dataset.type import AgentType

import tqdm

from typing import Optional

class TestStatistics:
    def __init__(
        self,
        signal_prediction_acc: float,
        overall_ade: float,
        overall_fde: float,
        ade_by_class: dict[int, float],
        fde_by_class: dict[int, float],
        de_by_step: list[float]
    ):
        self.signal_prediction_acc = signal_prediction_acc
        self.overall_ade = overall_ade
        self.overall_fde = overall_fde

        self.ade_by_class = ade_by_class
        self.fde_by_class = fde_by_class
        self.de_by_step = de_by_step

    def report(self):
        print('\n'.join([
            f'=== Overall Result ===',
            f'Signal Prediction Accuracy: {self.signal_prediction_acc}',
            f'Overall Final Displacement Error: {self.overall_fde}',
            f'Overall Average Displacement Error: {self.overall_ade}',
        ]))

        for agent_class in AgentType:
            print('\n'.join([
                f'=== Displacement Error of Class `{agent_class.name}` ===',
                f'FDE: {self.fde_by_class[agent_class]}',
                f'ADE: {self.ade_by_class[agent_class]}'
            ]))

    def save(self, path: str):
        torch.save({
            'signal_prediction_acc': self.signal_prediction_acc,
            'overall_ade': self.overall_ade,
            'overall_fde': self.overall_fde,
            'ade_by_class': self.ade_by_class,
            'fde_by_class': self.fde_by_class
        }, path)

def test_one_epoch(
    model: DynamicNeuralRelationalInference,
    dataloader: DataLoader,
    config: ExperimentConfig,
    tqdm_desc: Optional[str] = None,
    feed_true_signals: bool = False
):
    model.eval()

    signal_acc: float = 0.0

    overall_fde: list[float] = []
    overall_ade: list[float] = []

    fde_by_class = { int(agent_type): [] for agent_type in AgentType }
    ade_by_class = { int(agent_type): [] for agent_type in AgentType }
    de_by_step = [[] for _ in range(config.pred_len)]

    n_batches = len(dataloader)

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc=tqdm_desc, disable=tqdm_desc is None):
            (
                trajectory,
                trajectory_mask,
                _,
                agent_records,
                agent_flags,
                signals,
                graph_info
            ) = batch

            fde_mask = trajectory_mask[..., config.obs_len:].all(dim=-1)
            batch_sample_size = fde_mask.sum().item()

            if config.mask_traffic_signal:
                signals = torch.zeros_like(signals)

            if config.use_cuda:
                trajectory = trajectory.cuda()
                trajectory_mask = trajectory_mask.cuda()
                
                agent_records = agent_records.cuda()
                agent_flags = agent_flags.cuda()

                signals = signals.cuda()

                graph_info = graph_info.cuda()

                fde_mask = fde_mask.cuda()

            true_signals = None

            if feed_true_signals:
                true_signals = signals[..., config.obs_len:, 0]

            predictions, edge_types, edge_masks, pred_signals = model.predict_future(
                trajectory[..., :config.obs_len, :],
                trajectory_mask[..., :config.obs_len],
                agent_records,
                agent_flags[..., 0],
                signals[..., :config.obs_len, 0],
                n_steps=config.pred_len,
                graph_info=graph_info,
                put_nan=False,
                output_mode='point',
                ema_alpha=config.ema_alpha,
                true_signals=true_signals
            )

            signal_acc += classification_accuracy(
                pred_signals, 
                signals[..., config.obs_len:, 0]
            ) / n_batches

            if batch_sample_size < 1:
                continue

            # [N, pred_len]
            des = displacement_error(
                predictions[..., :2],
                trajectory[..., config.obs_len:, :2]
            )

            fdes = des[..., -1]
            ades = des.mean(dim=-1)

            overall_fde.append(fdes[fde_mask].mean())
            overall_ade.append(ades[fde_mask].mean())

            for batch_index in range(trajectory.size(0)):
                for agent_index in range(trajectory.size(1)):
                    if not fde_mask[batch_index, agent_index].item():
                        continue

                    agent_class = int(agent_flags[batch_index, agent_index, 0].item())
                    
                    fde_by_class[agent_class].append(fdes[batch_index, agent_index].item())
                    ade_by_class[agent_class].append(ades[batch_index, agent_index].item())

            # displacement error by step
            des = des[fde_mask].mean(dim=0).tolist()

            for timestamp in range(config.pred_len):
                de_by_step[timestamp].append(des[timestamp])



    return TestStatistics(
        signal_prediction_acc=signal_acc,
        overall_ade=sum(overall_ade) / n_batches,
        overall_fde=sum(overall_fde) / n_batches,
        ade_by_class={
            agent_class: sum(ades) / max(len(ades), 1)
            for agent_class, ades in ade_by_class.items()
        },
        fde_by_class={
            agent_class: sum(fdes) / max(len(fdes), 1)
            for agent_class, fdes in fde_by_class.items()
        },
        de_by_step=[
            sum(de) / max(len(de), 1)
            for de in de_by_step
        ]
    )

def generate_result(
    model: DynamicNeuralRelationalInference, 
    dataset: SignalizedIntersectionDatasetForNRI,
    config: ExperimentConfig,
    feed_true_signals: bool = False
):
    if config.use_cuda:
        model.cuda()

    dataloader = DataLoader(
        dataset, 
        shuffle=False, 
        batch_size=config.batch_size, 
        collate_fn=SignalizedIntersectionDatasetForNRI.collate_padded
    )

    test_stats = test_one_epoch(model, dataloader, config, '[test] generating result', feed_true_signals)
    test_stats.report()
    return test_stats
