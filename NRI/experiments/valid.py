import torch
from torch.utils.data import DataLoader

import tqdm

from NRI.models import DynamicNeuralRelationalInference
from NRI.experiments.config import ExperimentConfig
from NRI.experiments.metrics import displacement_error, classification_accuracy

class ValidStatistics:
    def __init__(
        self,
        signal_prediction_acc: float,
        fde: float,
        ade: float
    ):
        self.fde = fde
        self.ade = ade
        self.signal_prediction_acc = signal_prediction_acc

    def print(self):
        print('\n'.join([
            f'Signal Prediction Accuracy: {self.signal_prediction_acc}',
            f'Final Displacement Error: {self.fde}',
            f'Average Displacement Error: {self.ade}'
        ]))

def valid_one_epoch(
    model: DynamicNeuralRelationalInference,
    dataloader: DataLoader,
    tqdm_desc: str,
    config: ExperimentConfig
) -> ValidStatistics:
    model.eval()

    fde: list[float] = []
    ade: list[float] = []

    avg_signal_prediction_acc = 0.0
    n_batches = len(dataloader)

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc=tqdm_desc):
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
            true_signals = None


            if config.use_cuda:
                trajectory = trajectory.cuda()
                trajectory_mask = trajectory_mask.cuda()
                
                agent_records = agent_records.cuda()
                agent_flags = agent_flags.cuda()

                signals = signals.cuda()

                graph_info = graph_info.cuda()

                fde_mask = fde_mask.cuda()

            if config.mask_traffic_signal:
                signals = torch.zeros_like(signals)
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

            avg_signal_prediction_acc += classification_accuracy(
                pred_signals, 
                signals[..., config.obs_len:, 0]
            ) / n_batches

            if batch_sample_size < 1:
                continue

            fde.append(displacement_error(
                predictions[fde_mask][:, -1, :2],
                trajectory[fde_mask][:, -1, :2]
            ).mean().item())

            ade.append(displacement_error(
                predictions[fde_mask][..., :2].reshape(-1, 2),
                trajectory[fde_mask][:, config.obs_len:, :2].reshape(-1, 2)
            ).mean().item())
            
    avg_fde = sum(fde) / max(len(fde), 1)
    avg_ade = sum(ade) / max(len(ade), 1)

    return ValidStatistics(
        signal_prediction_acc=avg_signal_prediction_acc,
        fde=avg_fde,
        ade=avg_ade
    )
    