import torch
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

import tqdm

from NRI.models import DynamicNeuralRelationalInference
from NRI.experiments.config import ExperimentConfig

class TrainStatistics:
    def __init__(
        self,
        loss: float,
        encoder_loss: float,
        decoder_loss: float,
        signal_loss: float
    ):
        self.loss = loss
        self.encoder_loss = encoder_loss
        self.decoder_loss = decoder_loss
        self.signal_loss = signal_loss

    def print(self):
        print('\n'.join([
            f'Overall Loss: {self.loss}',
            f'Encoder KL Loss: {self.encoder_loss}',
            f'Decoder NLL Loss: {self.decoder_loss}',
            f'Signal Cross-Entropy Loss: {self.signal_loss}'
        ]))

def train_one_epoch(
    model: DynamicNeuralRelationalInference,
    dataloader: DataLoader,
    optimizer: Optimizer,
    tqdm_desc: str,
    config: ExperimentConfig
) -> TrainStatistics:
    model.train()

    avg_loss: float = 0.0
    avg_encoder_loss: float = 0.0
    avg_decoder_loss: float = 0.0
    avg_signal_loss: float = 0.0
    sample_size = len(dataloader)

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

        if config.mask_traffic_signal:
            signals = torch.zeros_like(signals)

        if config.use_cuda:
            trajectory = trajectory.cuda()
            trajectory_mask = trajectory_mask.cuda()

            agent_records = agent_records.cuda()
            agent_flags = agent_flags.cuda()

            signals = signals.cuda()
            graph_info = graph_info.cuda()

        optimizer.zero_grad()     


        if config.batched_training:
            signal_prediction_loss, encoder_loss, decoder_loss, decoder_burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals = model.batch_training_loss(
                trajectory,
                trajectory_mask,
                agent_records,
                agent_flags[..., 0],
                graph_info,
                signals[..., 0],
                obs_len=config.obs_len,
                pred_len=config.pred_len,
                put_nan=False,
                debug=False,
                include_decoder_burn_in=config.include_burn_in_loss,
                avg_logits=config.avg_logits,
                ema_alpha=config.ema_alpha,
                teacher_forcing_steps=config.teacher_forcing_steps
            )
        else:
            signal_prediction_loss, encoder_loss, decoder_loss, decoder_burn_in_loss, predictions, edge_logits, edge_masks, predicted_signals = model.training_loss(
                trajectory,
                trajectory_mask,
                agent_records,
                agent_flags[..., 0],
                graph_info,
                signals[..., 0],
                obs_len=config.obs_len,
                pred_len=config.pred_len,
                put_nan=False,
                debug=False,
                include_decoder_burn_in=config.include_burn_in_loss,
                avg_logits=config.avg_logits,
                ema_alpha=config.ema_alpha,
                teacher_forcing_steps=config.teacher_forcing_steps
            )


        if config.include_burn_in_loss:
            decoder_loss = decoder_loss + decoder_burn_in_loss

        loss = config.encoder_loss_weight * encoder_loss + config.decoder_loss_weight * decoder_loss
        
        if config.signal_loss_weight > 0.0:
            loss += config.signal_loss_weight * signal_prediction_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        avg_loss += loss.item() / sample_size
        avg_encoder_loss += encoder_loss.item() / sample_size
        avg_decoder_loss += decoder_loss.item() / sample_size
        avg_signal_loss += signal_prediction_loss.item() / sample_size

    return TrainStatistics(
        loss=avg_loss,
        encoder_loss=avg_encoder_loss,
        decoder_loss=avg_decoder_loss,
        signal_loss=avg_signal_loss
    )

